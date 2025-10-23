#include "ccc.hpp"
#include <cassert>
using namespace qallow::ccc;
int main(){
  assert(gray2int(0b000)==0);
  assert(gray2int(0b001)==1);
  assert(gray2int(0b011)==2);
  assert(gray2int(0b010)==3);
  return 0;
}
