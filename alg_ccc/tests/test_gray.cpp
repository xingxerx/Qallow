#include "ccc.hpp"
#include <cassert>
#include <stdexcept>
using namespace qallow::ccc;
int main(){
  assert(gray2int(0b000)==0);
  assert(gray2int(0b001)==1);
  assert(gray2int(0b011)==2);
  assert(gray2int(0b010)==3);

  bool review_called = false;
  const auto reviewer = [&](uint32_t gray, int decoded) {
    assert(gray == 0b010);
    assert(decoded == 3);
    review_called = true;
    return true;
  };
  assert(gray2int(0b010, reviewer) == 3);
  assert(review_called);

  bool rejected = false;
  try {
    gray2int(0b001, [](uint32_t, int) { return false; });
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);

  bool invalid = false;
  try {
    gray2int(0b001, GrayReviewCallback{});
  } catch (const std::invalid_argument&) {
    invalid = true;
  }
  assert(invalid);
  return 0;
}
