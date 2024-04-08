#include "calc.h"
#include <stdint.h>

uint64_t getCombinationsCount(uint32_t signal_size, uint32_t base)
{
  uint64_t combinations_count = base;

  for (int i = 0; i < signal_size; i++) {
    combinations_count = combinations_count * base;
  }

  return combinations_count;
}

uint32_t upperPowerOfTwo(uint32_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}
