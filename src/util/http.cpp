#include "util/http.hpp"

#include "common.hpp"

#ifdef _WIN32
#include <windows.h>
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")
#ifndef WINHTTP_OPTION_IPV6_FAST_FALLBACK
#define WINHTTP_OPTION_IPV6_FAST_FALLBACK 110
#endif
#else
#include <curl/curl.h>
#include <mutex>
#endif

#include <sstream>
#include <vector>

namespace xn {
namespace {

std::string default_user_agent() {
    return std::string("xnminer-cuda/") + kMinerVersion;
}

#ifdef _WIN32

std::wstring to_wide(const std::string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

std::string to_utf8(const std::wstring& w) {
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), nullptr, 0,
                                nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), s.data(), n, nullptr,
                        nullptr);
    return s;
}

struct ParsedUrl {
    bool https = false;
    std::wstring host;
    INTERNET_PORT port = 0;
    std::wstring path;
    bool ok = false;
};

ParsedUrl parse_url(const std::string& url) {
    ParsedUrl p;
    URL_COMPONENTS uc{};
    uc.dwStructSize = sizeof(uc);
    uc.dwSchemeLength = static_cast<DWORD>(-1);
    uc.dwHostNameLength = static_cast<DWORD>(-1);
    uc.dwUrlPathLength = static_cast<DWORD>(-1);
    uc.dwExtraInfoLength = static_cast<DWORD>(-1);
    auto wurl = to_wide(url);
    if (!WinHttpCrackUrl(wurl.c_str(), 0, 0, &uc)) {
        return p;
    }
    p.https = (uc.nScheme == INTERNET_SCHEME_HTTPS);
    p.host.assign(uc.lpszHostName, uc.dwHostNameLength);
    p.port = uc.nPort;
    p.path.assign(uc.lpszUrlPath, uc.dwUrlPathLength);
    if (uc.dwExtraInfoLength > 0 && uc.lpszExtraInfo) {
        p.path.append(uc.lpszExtraInfo, uc.dwExtraInfoLength);
    }
    if (p.path.empty()) p.path = L"/";
    p.ok = true;
    return p;
}

std::string winhttp_err(const char* what) {
    const DWORD e = GetLastError();
    std::ostringstream oss;
    oss << what << " (winhttp " << e << ")";
    return oss.str();
}

// Per-thread keep-alive: xenblocks.io /difficulty is flaky and match windows
// are only a few minutes. Reusing TCP skips the handshake on every poll/submit.
struct WinHttpKeepAlive {
    HINTERNET session = nullptr;
    HINTERNET conn = nullptr;
    std::wstring host;
    INTERNET_PORT port = 0;
    bool https = false;
    std::wstring user_agent;
    int timeout_ms = 0;

    ~WinHttpKeepAlive() { close(); }

    void close() {
        if (conn) {
            WinHttpCloseHandle(conn);
            conn = nullptr;
        }
        if (session) {
            WinHttpCloseHandle(session);
            session = nullptr;
        }
        host.clear();
        port = 0;
        https = false;
        timeout_ms = 0;
    }

    bool ensure(const ParsedUrl& parsed, int timeout_ms_, const std::wstring& ua) {
        const bool same_peer =
            session && conn && host == parsed.host && port == parsed.port && https == parsed.https;
        if (same_peer) {
            if (timeout_ms != timeout_ms_ && session) {
                timeout_ms = timeout_ms_;
                WinHttpSetTimeouts(session, timeout_ms_, timeout_ms_, timeout_ms_, timeout_ms_);
            }
            return true;
        }
        close();
        // Direct (no system proxy) — VPN/WinHTTP proxy layers have hung HTTP:80
        // while HTTPS still answered.
        session = WinHttpOpen(ua.c_str(), WINHTTP_ACCESS_TYPE_NO_PROXY, WINHTTP_NO_PROXY_NAME,
                              WINHTTP_NO_PROXY_BYPASS, 0);
        if (!session) return false;
        timeout_ms = timeout_ms_;
        WinHttpSetTimeouts(session, timeout_ms_, timeout_ms_, timeout_ms_, timeout_ms_);
        DWORD ipv6_fast = 1;
        WinHttpSetOption(session, WINHTTP_OPTION_IPV6_FAST_FALLBACK, &ipv6_fast, sizeof(ipv6_fast));
        conn = WinHttpConnect(session, parsed.host.c_str(), parsed.port, 0);
        if (!conn) {
            close();
            return false;
        }
        host = parsed.host;
        port = parsed.port;
        https = parsed.https;
        user_agent = ua;
        return true;
    }
};

thread_local WinHttpKeepAlive tls_http;

HttpResponse winhttp_request(const std::string& method, const std::string& url,
                             const std::string& body, int timeout_ms,
                             const std::string& content_type,
                             const std::string& user_agent) {
    HttpResponse out;
    auto parsed = parse_url(url);
    if (!parsed.ok) {
        out.error = "invalid url";
        return out;
    }

    const int t = timeout_ms > 0 ? timeout_ms : 5000;
    if (!tls_http.ensure(parsed, t, to_wide(user_agent))) {
        out.error = winhttp_err("WinHttpOpen/Connect failed");
        return out;
    }

    DWORD flags = parsed.https ? WINHTTP_FLAG_SECURE : 0;
    // Bypass WinHTTP cache so /difficulty is never a stale 1100.
    flags |= WINHTTP_FLAG_REFRESH;
    HINTERNET req = WinHttpOpenRequest(tls_http.conn, to_wide(method).c_str(), parsed.path.c_str(),
                                       nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                                       flags);
    if (!req) {
        out.error = winhttp_err("WinHttpOpenRequest failed");
        tls_http.close();
        return out;
    }

    // Disable redirects — pool 401/3xx must surface as-is.
    DWORD redir = WINHTTP_DISABLE_REDIRECTS;
    WinHttpSetOption(req, WINHTTP_OPTION_DISABLE_FEATURE, &redir, sizeof(redir));

    std::wstring headers = L"Connection: keep-alive\r\n";
    if (!content_type.empty()) {
        headers += L"Content-Type: " + to_wide(content_type) + L"\r\n";
    }

    BOOL ok = WinHttpSendRequest(req, headers.c_str(), static_cast<DWORD>(-1L),
                                 body.empty() ? WINHTTP_NO_REQUEST_DATA
                                              : const_cast<char*>(body.data()),
                                 static_cast<DWORD>(body.size()),
                                 static_cast<DWORD>(body.size()), 0);
    if (!ok) {
        out.error = winhttp_err("WinHttpSendRequest failed");
        WinHttpCloseHandle(req);
        tls_http.close();
        return out;
    }

    if (!WinHttpReceiveResponse(req, nullptr)) {
        out.error = winhttp_err("WinHttpReceiveResponse failed");
        WinHttpCloseHandle(req);
        tls_http.close();
        return out;
    }

    DWORD status = 0;
    DWORD status_size = sizeof(status);
    WinHttpQueryHeaders(req, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                        WINHTTP_HEADER_NAME_BY_INDEX, &status, &status_size, WINHTTP_NO_HEADER_INDEX);
    out.status = static_cast<int>(status);

    std::string response;
    for (;;) {
        DWORD avail = 0;
        if (!WinHttpQueryDataAvailable(req, &avail)) break;
        if (avail == 0) break;
        std::vector<char> buf(avail);
        DWORD read = 0;
        if (!WinHttpReadData(req, buf.data(), avail, &read)) break;
        response.append(buf.data(), read);
    }
    out.body = std::move(response);

    WinHttpCloseHandle(req);
    return out;
}

// Champ miner submit path: new session per POST, system proxy, no keep-alive.
// The keep-alive + custom UA path was getting HTTP 401 on /verify.
HttpResponse winhttp_oneshot(const std::string& method, const std::string& url,
                             const std::string& body, int timeout_ms,
                             const std::string& content_type, const std::string& user_agent) {
    HttpResponse out;
    auto parsed = parse_url(url);
    if (!parsed.ok) {
        out.error = "invalid url";
        return out;
    }

    // Champ path: system proxy, one-shot session. NO_PROXY was 401'ing every
    // /verify (empty body) during m=100 while this path has landed HTTP 200s.
    HINTERNET session = WinHttpOpen(to_wide(user_agent).c_str(), WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                    WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!session) {
        out.error = winhttp_err("WinHttpOpen failed");
        return out;
    }

    int t = timeout_ms > 0 ? timeout_ms : 5000;
    WinHttpSetTimeouts(session, t, t, t, t);
    DWORD ipv6_fast = 1;
    WinHttpSetOption(session, WINHTTP_OPTION_IPV6_FAST_FALLBACK, &ipv6_fast, sizeof(ipv6_fast));

    HINTERNET conn = WinHttpConnect(session, parsed.host.c_str(), parsed.port, 0);
    if (!conn) {
        out.error = winhttp_err("WinHttpConnect failed");
        WinHttpCloseHandle(session);
        return out;
    }

    DWORD flags = parsed.https ? WINHTTP_FLAG_SECURE : 0;
    HINTERNET req = WinHttpOpenRequest(conn, to_wide(method).c_str(), parsed.path.c_str(), nullptr,
                                       WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
    if (!req) {
        out.error = winhttp_err("WinHttpOpenRequest failed");
        WinHttpCloseHandle(conn);
        WinHttpCloseHandle(session);
        return out;
    }

    std::wstring headers = L"Accept: application/json\r\n";
    if (!content_type.empty()) {
        headers += L"Content-Type: " + to_wide(content_type) + L"\r\n";
    }

    BOOL ok = WinHttpSendRequest(req, headers.empty() ? WINHTTP_NO_ADDITIONAL_HEADERS : headers.c_str(),
                                 headers.empty() ? 0 : static_cast<DWORD>(-1L),
                                 body.empty() ? WINHTTP_NO_REQUEST_DATA
                                              : const_cast<char*>(body.data()),
                                 static_cast<DWORD>(body.size()),
                                 static_cast<DWORD>(body.size()), 0);
    if (!ok) {
        out.error = winhttp_err("WinHttpSendRequest failed");
        WinHttpCloseHandle(req);
        WinHttpCloseHandle(conn);
        WinHttpCloseHandle(session);
        return out;
    }

    if (!WinHttpReceiveResponse(req, nullptr)) {
        out.error = winhttp_err("WinHttpReceiveResponse failed");
        WinHttpCloseHandle(req);
        WinHttpCloseHandle(conn);
        WinHttpCloseHandle(session);
        return out;
    }

    DWORD status = 0;
    DWORD status_size = sizeof(status);
    WinHttpQueryHeaders(req, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                        WINHTTP_HEADER_NAME_BY_INDEX, &status, &status_size, WINHTTP_NO_HEADER_INDEX);
    out.status = static_cast<int>(status);

    std::string response;
    for (;;) {
        DWORD avail = 0;
        if (!WinHttpQueryDataAvailable(req, &avail)) break;
        if (avail == 0) break;
        std::vector<char> buf(avail);
        DWORD read = 0;
        if (!WinHttpReadData(req, buf.data(), avail, &read)) break;
        response.append(buf.data(), read);
    }
    out.body = std::move(response);

    WinHttpCloseHandle(req);
    WinHttpCloseHandle(conn);
    WinHttpCloseHandle(session);
    return out;
}

#else  // !_WIN32

void ensure_curl() {
    static std::once_flag once;
    std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

size_t write_cb(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* s = static_cast<std::string*>(userdata);
    s->append(ptr, size * nmemb);
    return size * nmemb;
}

// Per-thread keep-alive: xenblocks.io /difficulty is flaky and match windows
// are only a few minutes. Reusing the TCP/TLS connection skips handshake.
struct CurlKeepAlive {
    CURL* curl = nullptr;
    ~CurlKeepAlive() {
        if (curl) curl_easy_cleanup(curl);
    }
    CURL* get() {
        if (!curl) curl = curl_easy_init();
        return curl;
    }
    void reset() {
        if (curl) curl_easy_reset(curl);
    }
};

thread_local CurlKeepAlive tls_curl;

HttpResponse curl_request(const std::string& method, const std::string& url, const std::string& body,
                          int timeout_ms, const std::string& content_type,
                          const std::string& user_agent, const std::string& extra_header = {}) {
    ensure_curl();
    HttpResponse out;
    CURL* curl = tls_curl.get();
    if (!curl) {
        out.error = "curl_easy_init failed";
        return out;
    }
    tls_curl.reset();
    const int t = timeout_ms > 0 ? timeout_ms : 5000;
    std::string response;
    curl_slist* headers = nullptr;
    if (!content_type.empty()) {
        const std::string h = "Content-Type: " + content_type;
        headers = curl_slist_append(headers, h.c_str());
    }
    if (!extra_header.empty()) {
        headers = curl_slist_append(headers, extra_header.c_str());
    }
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, method.c_str());
    curl_easy_setopt(curl, CURLOPT_USERAGENT, user_agent.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(t));
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, static_cast<long>(t));
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "");
    if (headers) curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    if (!body.empty() || method == "POST" || method == "PUT") {
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
    }
    const CURLcode rc = curl_easy_perform(curl);
    if (rc != CURLE_OK) {
        out.error = curl_easy_strerror(rc);
    } else {
        long status = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
        out.status = static_cast<int>(status);
        out.body = std::move(response);
    }
    if (headers) curl_slist_free_all(headers);
    return out;
}

#endif  // _WIN32

}  // namespace

HttpResponse http_get(const std::string& url, int timeout_ms, const std::string& extra_header) {
#ifdef _WIN32
    return winhttp_request("GET", url, {}, timeout_ms, {}, default_user_agent());
#else
    return curl_request("GET", url, {}, timeout_ms, {}, default_user_agent(), extra_header);
#endif
}

HttpResponse http_post_json(const std::string& url, const std::string& json_body, int timeout_ms,
                            const std::string& user_agent, const std::string& extra_header) {
#ifdef _WIN32
    const std::string ua = user_agent.empty() ? "python-requests/2.31.0" : user_agent;
    return winhttp_oneshot("POST", url, json_body, timeout_ms, "application/json; charset=utf-8",
                           ua);
#else
    const std::string ua = user_agent.empty() ? default_user_agent() : user_agent;
    return curl_request("POST", url, json_body, timeout_ms, "application/json; charset=utf-8", ua,
                        extra_header);
#endif
}

}  // namespace xn
