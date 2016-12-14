'''
An example of an array/tree satisfying the heap properity, given that we use
the weights to assign priority.

               10
        0              2
     3     1         5    6
   7 8    9 4


Indices: 1 4 2 3 10 5 6 7 8 9 0
Elems: 0 10 2 3 1 5 6 7 8 9 4
Weights: 0 0 1 0 0 0 0 0 0 10
'''
import numpy as np

class Heap:
  def leftChild(self, currIdx):
    return 2*currIdx+1
  def rightChild(self, currIdx):
    return 2*currIdx+2
  def parent(self, currIdx):
    return (currIdx-1)//2
  def hasAChild(self, currIdx):
    return 2*currIdx < self.size
  def maxPriorityChild(self, currIdx):
   #left = self.leftChild(currIdx)    Inlining to improve performance
   #right = self.rightChild(currIdx)  Inlining to improve performance
   right = (currIdx << 1) + 2
   if right >= self.size:
     return (currIdx << 1) + 1
   left = (currIdx << 1) + 1
   if self.weight[self.elems[right]] > self.weight[self.elems[left]]:
     return right
   else:
     return left

  def swap(self, idx1, idx2):
    elem1 = self.elems[idx1]
    self.indices[elem1] = idx2
    self.elems[idx1] = self.elems[idx2]
    self.elems[idx2] = elem1
    self.indices[self.elems[idx1]] = idx1   

    #elem2 = self.elems[idx2]
    #self.indices[elem1] = idx2
    #self.indices[elem2] = idx1
    #self.elems[idx1] = elem2
    #self.elems[idx2] = elem1

  def heapifyDown(self, currIdx):
    #if self.hasAChild(currIdx):
    if (currIdx << 1) < self.size:
      child = self.maxPriorityChild(currIdx)
      if self.weight[self.elems[child]] > self.weight[self.elems[currIdx]]:
        self.swap(child, currIdx)
        self.heapifyDown(child)
        return True
    return False

  def increased(self, elem):
    idx = self.indices[elem] 
    self.heapifyUp(idx)

  def decreased(self, elem):
    idx = self.indices[elem] 
    self.heapifyDown(idx)



  def heapifyUp(self, currIdx):
    if currIdx == 0:
      return False
    parentIdx = (currIdx-1) >> 1
    if self.weight[self.elems[currIdx]] > self.weight[self.elems[parentIdx]]:
      self.swap(currIdx, parentIdx)
      self.heapifyUp(parentIdx)

  def top2(self):
      best = self.elems[0]
      if self.weight[self.elems[2]] > self.weight[self.elems[1]]:
        return (best, self.elems[2])
      else:
        return (best, self.elems[1])

  def __init__(self, elems, weight):
    assert(elems.shape[0] == weight.shape[0])
    self.weight = weight
    self.elems = elems.copy()
    self.indices = np.arange(self.elems.shape[0]) 
    self.size = self.elems.shape[0]
    it = self.elems.shape[0]-1;
    while it>0:
      self.heapifyDown(self.parent(it));
      it-=2;

import time

def two_max(arr):
  idx1 = 0
  idx2 = 0
  max1 = 0.0
  max2 = 0.0
  for i in range(len(arr)):
    if arr[i] >= max1:
      max2 = max1
      idx2 = idx1
      max1 = arr[i]
      idx1 = i
    elif arr[i] >= max2:
      max2 = arr[i]
      idx2 = i
  return idx1, idx2

if __name__ == "__main__":
  atomic_positions = np.zeros((216,3))
  disp_list = np.zeros((216,3));
  dist_list = np.zeros(216);
  max_heap = Heap(np.arange(216), dist_list)
  heap_time = 0.0
  linear_time = 0.0
  for i in range(1000):
    j = np.random.randint(0,high=216)
    dv = np.random.rand(3)-0.5
    old_distj = dist_list[j]
    disp_list[j] += dv
    dist_list[j] = np.linalg.norm(disp_list[j])
    heap_t = time.time()
    if dist_list[j] > old_distj:
      max_heap.increased(j)
    else:
      max_heap.decreased(j)
    (h1, h2) = max_heap.top2()
    heap_time += time.time()-heap_t
    linear_t = time.time()
    (d1, d2) = two_max(dist_list)
    linear_time += time.time() - linear_t
    print((d1, d2), (h1, h2))
    if (dist_list[d1] != dist_list[h1] or dist_list[d2] != dist_list[h2]):
      print("(*d1, *d2, *h1, *h2) = ({0},{1},{2},{3}))".format(dist_list[d1],
        dist_list[d2], dist_list[h1], dist_list[h2]))
    assert(dist_list[d1] == dist_list[h1] and dist_list[d2] == dist_list[h2])
    if(np.random.rand() > 0.01):
      print("Random redefinition!")
      disp_list = np.zeros((216, 3))
      dist_list[:] = 0
  print("Heap took {0} seconds.".format(heap_time))
  print("Linear search took {0} seconds.".format(linear_time))
