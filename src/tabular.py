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
  global exp_calc
  global exp_dx_inv
  global exp_hits
  exp_x_min = x_min
  exp_x_max = x_max
  exp_dx = dx
  exp_dx_inv = 1.0/dx #divison is MUCH, MUCH slower than multiplication
  exp_table = np.zeros(int((x_max-x_min)/dx)+5)
  exp_calc = 1+np.zeros_like(exp_table).astype(bool)
  exp_hits = 0

def exp(x):
  global exp_hits
  assert(x >= exp_x_min and x < exp_x_max)
  idx = int((x-exp_x_min)*exp_dx_inv)
  if exp_calc[idx]:
      exp_calc[idx] = False #No longer need to recompute this
      exp_table[idx] = math.exp(exp_x_min+exp_dx*idx) #Compute exp at left edge
  else:
    exp_hits += 1
  return exp_table[idx]

def exp_interp(x):
  idx = (x-exp_x_min)*exp_dx_inv
  bin_dx = x-exp_dx*idx
  if exp_calc[idx]:
      exp_calc[idx] = False
      exp_table[idx] = exp(exp_x_min+exp_dx*idx)
  else:
    exp_hits += 1
  return exp_table[idx]*(1.0+bin_dx) #First degree Taylor approximation


if __name__ == '__main__':
  numbers = np.random.rand(100000)
  set_exp_table(0, 1, 0.001)
  tabulars = np.zeros(100000)
  actuals = np.zeros(100000)
  for i in range(100000):
    tabulars[i] = exp(numbers[i])
    actuals[i] = math.exp(numbers[i])
  L2 = np.linalg.norm(tabulars-actuals)/math.sqrt(100000)
  print("Norm of", L2)
  print(exp_hits,"cache hits")
