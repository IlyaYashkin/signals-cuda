#include <iostream>
#include <stdio.h>
#include <cmath>
#include <math.h>

using namespace std;

#define N 15
#define PHASE 45
#define BASE (360 / PHASE)
#define SIGNAL 10544233731

int main()
{
  int signal[N];

  size_t m_trans_size = BASE * sizeof(float);

  float *signal_Re = (float*)malloc(m_trans_size);
  float *signal_Im = (float*)malloc(m_trans_size);

  for (int i = 0; i < BASE; i++) {
    float rad = 2 * atan(1.0) * 4 * i / BASE;
    signal_Im[i] = sin(rad);
    signal_Re[i] = cos(rad);
  }

  for (int i = 0; i < N; i++) {
    unsigned long long signal_part = SIGNAL;

    for (int j = 0; j < i; j++) {
      if (signal_part == 0) { break; }
      signal_part /= BASE;
    }
    signal[i] = signal_part % BASE;
  }

  float akf[N];
  for (int i = 0; i < N; i++) {
    float sum_Re = 0;
    float sum_Im = 0;
    for (int j = 0; j + i < N; j++) {
      int idx = (BASE + signal[j] - signal[i + j]) % BASE;

      sum_Re += signal_Re[idx];
      sum_Im += signal_Im[idx];
    }

    akf[i] = sqrtf(sum_Re * sum_Re + sum_Im * sum_Im);
    cout << akf[i] << ' ';
  }
  cout << endl;
}
