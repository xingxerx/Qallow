#include "include/ccc.hpp"

namespace qallow {
namespace ccc {

int gray2int(uint32_t g) {
    int result = 0;
    for (; g; g >>= 1) result ^= g;
    return result;
}

}
}
