#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>

using namespace std;

typedef float2 Complex;

#define PI 3.14159265358979323846

#define N 64
#define BITS 6

void getSignal(Complex* signal)
{
  for (int i = 0; i < N; i++) {
    signal[i].x = 0;
    signal[i].y = 0;
  }

  signal[0].x = 1;
  signal[1].x = 1;
  signal[2].x = 1;
  signal[3].x = -1;
  signal[4].x = 1;
  signal[5].x = -1;
  signal[6].x = -1;
  signal[7].x = -1;
  signal[8].x = -1;
  signal[9].x = -1;
  signal[10].x = -1;
  signal[11].x = -1;
  signal[12].x = -1;
  signal[13].x = -1;
  signal[14].x = -1;
  signal[15].x = -1;
  signal[16].x = -1;
}

void printSignal(Complex* signal)
{
  for (int i = 0; i < N; i++) {
    printf("%10f %f\n", signal[i].x, signal[i].y);
  }
}

unsigned int reverseBits(unsigned int num, unsigned int numberOfBits)
{
    unsigned int reverse_num = 0;
    int i;
    for (i = 0; i < numberOfBits; i++) {
        if ((num & (1 << i)))
            reverse_num |= 1 << ((numberOfBits - 1) - i);
    }
    return reverse_num;
}

Complex multiplyCompl(Complex first, Complex second)
{
  Complex result = {
    first.x * second.x - first.y * second.y,
    first.x * second.y + first.y * second.x
  };

  return result;
}

Complex sumCompl(Complex first, Complex second)
{
  Complex result = {
    first.x + second.x,
    first.y + second.y
  };

  return result;
}

Complex subCompl(Complex first, Complex second)
{
  Complex result = {
    first.x - second.x,
    first.y - second.y
  };

  return result;
}

void findFFT(Complex* signal, int direction = -1)
{
  for (int i = 1; i < N; i++) {
    unsigned int j = reverseBits(i, BITS);

    if (i < j) {
      Complex temp = signal[i];
      signal[i] = signal[j];
      signal[j] = temp;
    }
  }

  printf("\n");
  for (int i = 0; i < N; i++) {
    printf("index = %d %10f %10f\n", i, signal[i].x, signal[i].y);
  }
  printf("\n");

  unsigned int step = 2;
  while (step <= N) {
    unsigned int halfStep = step / 2;

    float angle = direction * 2 * PI / step;
    Complex delta = {cos(angle), sin(angle)};

    for (int i = 0; i < N; i += step) {
      Complex w = {1,0};
      for (int j = i; j < i + halfStep; j++) {
        Complex u = signal[j];
        Complex v = multiplyCompl(signal[j + halfStep], w);

        // if (direction == -1) printf("%f %f\n", u.x, v.y);

        // if (direction == -1) printf("u = %10f %10f v = %10f %10f | sub = %10f %10f\n", u.x, u.y, v.x, v.y, subCompl(u, v).x, subCompl(u, v).y);

        signal[j] = sumCompl(u, v);
        signal[j + halfStep] = subCompl(u, v);

        if (direction == -1) printf("u = %10f %10f v = %10f %10f step = %4d indexes = %4d %4d | replace = %10f %10f | %10f %10f | angle = %10f %10f\n", u.x, u.y, v.x, v.y, step, j, j + halfStep,
          signal[j].x, signal[j].y,
          signal[j + halfStep].x, signal[j + halfStep].y,
          w.x, w.y
        );
        w = multiplyCompl(w, delta);
      }
    }
    printf("\n");
    for (int i = 0; i < N; i++) {
      printf("index = %d %10f %10f\n", i, signal[i].x, signal[i].y);
    }
    printf("\n");
    step *= 2;
  }
}

int main()
{
  Complex signal[N];

  getSignal(signal);

  findFFT(signal);

  // printSignal(signal);

  return;

  findFFT(signal, 1);

  printSignal(signal);
}
