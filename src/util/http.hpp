#pragma once

#include <cstdint>
#include <string>

namespace xn {

struct HttpResponse {
    int status = 0;
    std::string body;
    std::string error;
};

// timeout_ms applies to connect+receive where supported (WinHTTP).
// GET keeps the keep-alive poller. POST /verify matches the champ miner:
// one-shot WinHTTP, User-Agent xnminer-cuda/4.0 unless overridden.
HttpResponse http_get(const std::string& url, int timeout_ms = 5000,
                      const std::string& extra_header = {});
HttpResponse http_post_json(const std::string& url, const std::string& json_body,
                            int timeout_ms = 10000,
                            const std::string& user_agent = "xnminer-cuda/4.0",
                            const std::string& extra_header = {});

}  // namespace xn
