#include <stdint.h>
#include <stdio.h>

int checkOverflow(uint64_t combinations_count)
{
  if (combinations_count <= 0) {
    printf("result array size error\n");
    return 1;
  }

  return 0;
}