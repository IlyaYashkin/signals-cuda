# -*- coding: utf-8 -*-
import numpy as np
import time
from numba import cuda
import warnings
import numba
cuda.select_device(0)
line_lenght_min = 5
line_lenght_max = 40
start_time = time.time()
line_95 = (0, 0, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 2, 3, 3, 3,
 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)

@cuda.jit('void(int8[:], uint64[:], int8[:,:], int32[:], uint64[:])')
def ak_func_demo(d_k, d_j, d_max_line, d_min, d_index_min):

  row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

  m = row + d_j[0] * 2**32 # поправка на батчер (batch = max(0, k - 33)) по формуле d_j[0] * 2 ** [k стартабатчера - 2]

  d_max_line[row][0] = 0
  d_max_line[row][1] = d_k[0]

  line_95 = (0, 0, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 2, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)
  line_max = line_95[d_k[0]]

  flag = 0

  for d in range(1, d_k[0]):
    akf = 0
    g = 0

    while d+g < d_k[0]:
      m_div_pow_2_g = m // pow(2, g)
      akf += ((m_div_pow_2_g // pow(2, d) % 2) * 2 - 1) * ((m_div_pow_2_g % 2) * 2 - 1)
      g += 1

    akf = abs(akf)

    if (akf > line_max) or (flag == 0 and d > 5):
      d_max_line[row] = d_k[0]
      break
    else:
      d_max_line[row][0] = max(akf, d_max_line[row][0])
      if akf == 0:
        d_max_line[row][1] = min(d, d_max_line[row][1])
        flag = 1

  # Поиск оптимального сигнала и его индекса
  cuda.atomic.min(d_min, 0, d_max_line[row][0])
  if d_max_line[row][0] == d_min[0]:
    if d_max_line[row][1] <= d_max_line[d_index_min[0]][1]:
      d_index_min[0] = row


@numba.jit(nopython = True)
def get_bin(count, k):
  line = np.zeros(k, dtype=np.int8)

  for i in range(len(line)):
    line[i] = count % 2 * 2 - 1
    count //= 2

  akf = np.zeros_like(line)

  for d in range(k):
    for g in range(k):
      if d+g < k:
        akf[d] += line[d+g] * line[g]
  return line, akf


batch = 0
tpb = cuda.get_current_device().WARP_SIZE


for k in range(line_lenght_min, line_lenght_max + 1):
  batch = max(0, k - 33)

  batch_pow = pow(2, batch)

  power = k - 2 - batch

  k_pow = pow(2, power)

  bpg = int(np.ceil(k_pow / tpb))

  d_k = cuda.to_device(np.array([k], dtype=np.int8))
  d_max_line = cuda.device_array((k_pow, 2), dtype=np.int8)
  d_index_min = cuda.to_device(np.array([0], dtype=np.uint64))

  best_on_line = 0

  for j in range(batch_pow // 8, max(batch_pow // 8 * 7, batch_pow % 8)):

    d_min = cuda.to_device(np.array([100], dtype=np.int32))

    with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ak_func_demo[bpg, tpb](
      d_k,
      cuda.to_device(np.array([j], dtype=np.uint64)), # d_j
      d_max_line,
      d_min,
      d_index_min
    )
    
    optimal_amp = d_min.copy_to_host()[0]

    best_on_line = k / optimal_amp

    if best_on_line >= k / line_95[k]:
      index_min = int(d_index_min.copy_to_host()[0])
      optimal_line, optimal_akf = get_bin(index_min + j * 2**32, k)

      text_message = f'\n{ optimal_line } \
        \n{ optimal_akf } \
        \n{ k } / { optimal_amp } = { best_on_line } - Batch: { j+1 }/{ batch_pow }, {
        np.round((time.time() - start_time), 2) } sec\n'

      print (text_message)
      f = open('akf_bin.txt', 'a')
      f.write(text_message)
      f.close()

    else:
      print('\rFor {} in batch: {}/{}, no better than {} - {} sec'.format(
        k, j+1, batch_pow, line_95[k], np.round((time.time() - start_time), 2)), end='')

 d_max_line = cuda.device_array(1, dtype=np.int8)