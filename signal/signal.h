#include <stdint.h>

#ifndef SIGNAL_H
#define SIGNAL_H

__global__ void kernel(
  float *c,
  uint64_t offset,
  uint32_t signal_size,
  uint32_t base
);

__global__ void kernel_doppler(
  float *c,
  uint64_t offset,
  uint32_t signal_size,
  uint32_t base
);

#endif
