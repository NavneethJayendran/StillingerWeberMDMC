'''
This module contains tools to calculate certain functions in a fast but
memory intensive manner.
'''

import math
import numpy as np

def set_exp_table(x_min, x_max, dx):
  global exp_x_min
  global exp_x_max
  global exp_dx
  global exp_table
  global exp_dx_inv
  exp_x_min = x_min
  exp_x_max = x_max
  exp_dx = dx
  exp_dx_inv = 1.0/dx #divison is MUCH, MUCH slower than multiplication
  exp_table = np.zeros(int((x_max-x_min)/dx)+5)
  #exp_calc = 1+np.zeros_like(exp_table).astype(bool)
  value = exp_x_min + exp_dx/2
  for i in range(len(exp_table)):
    exp_table[i] = math.exp(value)
    value += exp_dx

def exp(x):
  #assert(x >= exp_x_min and x < exp_x_max)
  idx = int((x-exp_x_min)*exp_dx_inv) #get bin index
  #if exp_calc[idx]:
  #    exp_calc[idx] = False #No longer need to recompute this
  #    exp_table[idx] = math.exp(exp_x_min+exp_dx*idx) #Compute exp at left edge
  #else:
  #  exp_hits += 1
  return exp_table[idx]

def exp_interp(x):
  #assert(x >= exp_x_min and x < exp_x_max)
  idx = int((x-exp_x_min)*exp_dx_inv) #get bin index
  bin_dx = x-exp_dx*(idx+0.5)-exp_x_min  #get offset from bin center
  #bin_dx = x-exp_dx*idx-exp_x_min #get offset from bin left edge
  #if exp_calc[idx]:
  #    exp_calc[idx] = False
  #    exp_table[idx] = exp(exp_x_min+exp_dx*idx)
  #else:
  #  exp_hits += 1
  return exp_table[idx]*(1.0+bin_dx) #First degree Taylor approximation


if __name__ == '__main__':
  import timeit
  numbers = np.random.rand(10000)
  set_exp_table(0, 1, 0.01)
  tabulars1 = np.zeros(10000) #use simple table lookup
  tabulars2 = np.zeros(10000) #use first degree Taylor lookup
  actuals  = np.zeros(10000)  #use math.exp()
  
  for i in range(10000):
    tabulars1[i] = exp(numbers[i])
    tabulars2[i] = exp_interp(numbers[i]) 
    actuals[i] = math.exp(numbers[i])
    #print(tabulars1[i], tabulars2[i], actuals[i])
  L1_tab = np.linalg.norm(tabulars1-actuals, ord=1)/10000
  Linf_tab = np.max(np.abs(tabulars1-actuals))
  L1_int = np.linalg.norm(tabulars2-actuals, ord=1)/10000
  Linf_int = np.max(np.abs(tabulars2-actuals))
  setup = "from __main__ import numbers, exp, exp_interp; import math"
  print("NUMERICAL ERROR TESTS")
  print("L1 Norm of difference for ordinary tabular: {} (Max {})".format( 
        L1_tab, Linf_tab))
  print("L1 Norm of difference for interpolated tabular: {} (Max {})".format(
        L1_int, Linf_int))
  print("TIMING TESTS (random access)")
  Tt = timeit.timeit("for num in numbers: exp(num)", number=1, setup=setup)
  Ti = timeit.timeit("for num in numbers: exp_interp(num)", 
                     number=1, setup=setup)
  Te = timeit.timeit("for num in numbers: math.exp(num)", number=1, setup=setup)
  print("math.exp(): {} per 1000000 iterations".format(Te))
  print("tabular.exp(): {} per 1000000 iterations".format(Tt))
  print("tabular.exp_interp(): {} per 1000000 iterations".format(Ti))
  numbers.sort()
  Tt = timeit.timeit("for num in numbers: exp(num)", number=1, setup=setup)
  Ti = timeit.timeit("for num in numbers: exp_interp(num)", 
                     number=1, setup=setup)
  Te = timeit.timeit("for num in numbers: math.exp(num)", number=1, setup=setup)
  print("TIMING TESTS (sequential access)")
  print("math.exp(): {} per 1000000 iterations".format(Te))
  print("tabular.exp(): {} per 1000000 iterations".format(Tt))
  print("tabular.exp_interp(): {} per 1000000 iterations".format(Ti))
  
