#include "qallow/env.h"

#include "qallow/logging.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <string>

namespace {
std::string g_last_error;

std::string trim(std::string value) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

void set_error(const std::string& err) {
    g_last_error = err;
    qallow_log_warn("env", "%s", err.c_str());
}
}

extern "C" {

int qallow_env_load(const char* path) {
    const char* effective = path && *path ? path : ".env";
    std::ifstream stream(effective);
    if (!stream.is_open()) {
        set_error(std::string("unable to open env file: ") + effective);
        return -1;
    }

    std::string line;
    size_t line_no = 0;
    while (std::getline(stream, line)) {
        ++line_no;
        std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        auto pos = trimmed.find('=');
        if (pos == std::string::npos) {
            set_error("invalid env line " + std::to_string(line_no));
            continue;
        }
        std::string key = trim(trimmed.substr(0, pos));
        std::string value = trim(trimmed.substr(pos + 1));
        if (key.empty()) {
            set_error("missing key at line " + std::to_string(line_no));
            continue;
        }
        if (setenv(key.c_str(), value.c_str(), 1) != 0) {
            set_error("failed to set env: " + key);
        }
    }

    qallow_log_info("env", "loaded %s", effective);
    return 0;
}

const char* qallow_env_last_error(void) {
    return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

}  // extern "C"
