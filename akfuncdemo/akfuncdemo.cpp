#include "akfuncdemo.h"
#include <iostream>
#include <sstream>

using namespace std;

double* ak_func_demo(double* signal, int n)
{
  double* akf = new double[n];
  for (int i = 0; i < n; i++) {
    int k = 0;
    akf[i] = 0.0;
    for (int j = 0; i + j < n; j++) {
      k++;
      akf[i] += signal[i + j] * signal[j];
    }
    akf[i] = abs(akf[i]);
    // akf[i] /= (double)k;
  }
  return akf;
}

double* get_signal_array(string signal)
{
  double *result = (double*)malloc(signal.size() * sizeof(double));

  int i = 0;
  for (char& c : signal) {
    double number = (double) c - '0' == 0 ? -1.0 : 1.0;
    result[i] = number;
    i++;
  }

  return result;
}


void ak_func_demo_test()
{
  string strSignal = "000110001110001101010011011001";

  double* signal = get_signal_array(strSignal);

  double* result = ak_func_demo(signal, strSignal.size());

  for (int i = 0; i < 30; i++) {
    cout << result[i] << " ";
  }

  cout << endl;
}
