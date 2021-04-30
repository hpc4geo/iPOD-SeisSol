
import numpy as np
from .sobol import i4_sobol as jbpkg_sobol
from .halton import halton as jbpkg_halton


class Sampler:
  def __init__(self, dimension):
    self.type = 'base'
    self.dimension = dimension
    self.ranges = np.zeros((dimension,2))
    for d in range(self.dimension):
      self.ranges[d][0] = 0.0
      self.ranges[d][1] = 1.0
    return
    
  def next(self):
    raise RuntimeError('[next] Not implemented for base class Sampler')

  def init_index(self, index):
    raise RuntimeError('[init_index] Not implemented for base class Sampler')

  def scale(self, vals):
    if self.ranges is not None:
      for d in range(self.dimension):
        e_s = self.ranges[d][1] - self.ranges[d][0]
        vals[d] = vals[d] * e_s + self.ranges[d][0]
    return vals




class SobolSampler(Sampler):
  
  def __init__(self, dimension, r=None):
    Sampler.__init__(self, dimension)
    if r is not None:
      self.ranges = np.copy(r)
    self.type = 'sobol'
    self.seed_in = 0
    self.seed_out = 0

  def next(self):
    self.seed_in = self.seed_out
    [ vals, self.seed_out ] = jbpkg_sobol(self.dimension, self.seed_in )
    
    if self.seed_out != self.seed_in + 1:
      raise RuntimeError('seed out differs from seed in by more than +1. Cannot be restarted!')
    
    vals = self.scale(vals)
    return vals

  def init_index(self, index):
    self.seed_in = index
    self.seed_out = index




class HaltonSampler(Sampler):
  
  def __init__(self, dimension, r=None):
    Sampler.__init__(self, dimension)
    if r is not None:
      self.ranges = np.copy(r)
    self.type = 'halton'
    self.index = 0
  
  def next(self):
    vals = jbpkg_halton(self.index, self.dimension)
    self.index += 1
    vals = self.scale(vals)
    return vals

  def init_index(self, index):
    self.index = index




class TensorProductSampler(Sampler):

  def __init__(self, dimension, ranges=None, partition=None, include_end_points=True):
    Sampler.__init__(self, dimension)
    if ranges is not None:
      self.ranges = np.copy(ranges)
    self.type = 'tensor'
    self.index = 0
    
    self.partition = [2] * self.dimension
    self.include_end_points = include_end_points

    if partition is not None:
      self.partition = partition
      if len(self.partition) != dimension:
        raise RuntimeError('`partition` must have length',str(dimension))

    ds = np.zeros(dimension)
    for d in range(dimension):
      ds[d] = self.ranges[d][1] - self.ranges[d][0]
    #print('ds',ds)

    n = 1
    if include_end_points == False:
      for d in range(dimension):
        n = n * self.partition[d]
    else:
      for d in range(dimension):
        n = n * (self.partition[d] + 1)
    self.npoints = n
    #print('npoints',n)

    if include_end_points == False:
      d = 0
      n0 = self.partition[0]
      h0 = ds[0]
      hs = h0 / float(n0)
      xc = list()
      for i in range(n0):
        xc.append( [self.ranges[d][0] + 0.5 * hs + i * hs] )

      x1 = list()
      for d in range(1,dimension):
        n1 = self.partition[d]
        hs = ds[d] / float(n1)
        x1 = list()
        for i in range(n1):
          x1.append(self.ranges[d][0] + 0.5 * hs + i * hs)
      
        xnew = list()
        for i in range(len(xc)):
          for j in range(n1):
            xnew.append( xc[i] + [x1[j]] )
        xc = list(xnew)
          
    else:
      d = 0
      n0 = self.partition[0]
      hs = ds[0] / float(n0)
      xc = list()
      for i in range(n0+1):
        xc.append( [self.ranges[d][0] + i * hs] )

      x1 = list()
      for d in range(1,dimension):
        n1 = self.partition[d]
        hs = ds[d] / float(n1)
        x1 = list()
        for i in range(n1+1):
          x1.append(self.ranges[d][0] + i * hs)
    
        xnew = list()
        for i in range(len(xc)):
          for j in range(n1+1):
            xnew.append( xc[i] + [x1[j]] )
        xc = list(xnew)

    self.point_set = xc
    #print(xc)


  def next(self):
    vals = None
    if self.index < self.npoints:
      vals = self.point_set[ self.index ]
      self.index += 1
    else:
      print('Exhausted all available tensor product points')
    return vals


  def init_index(self, index):
    self.index = index


  def get_points(self):
    return self.point_set
