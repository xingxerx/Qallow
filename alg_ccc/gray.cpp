#include "include/ccc.hpp"
#include <stdexcept>

namespace qallow {
namespace ccc {

int gray2int(uint32_t g) {
    uint32_t value = g;
    uint32_t mask = value >> 1;
    while (mask) {
        value ^= mask;
        mask >>= 1;
    }
    return static_cast<int>(value);
}


int gray2int(uint32_t g, const GrayReviewCallback& reviewer) {
    if (!reviewer) {
        throw std::invalid_argument("gray2int requires a reviewer callback when human approval is enabled");
    }

    const int decoded = gray2int(g);
    if (!reviewer(g, decoded)) {
        throw std::runtime_error("Gray code conversion rejected by human reviewer");
    }
    return decoded;
}

}
}
