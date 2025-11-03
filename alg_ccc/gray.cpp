#include "include/ccc.hpp"

namespace qallow {
namespace ccc {

unsigned int gray2int(unsigned int g) {
    unsigned int result = 0;
    for (; g; g >>= 1) result ^= g;
    return result;
}

}
}
