
import numpy as np
from sampling import SobolSampler, HaltonSampler, TensorProductSampler

# Tests
def sobol_test05():
  
  dimension = 3
  seed = 0
  for i in range( 0 , 10+1):
    [ r, seed_out ] = jbpkg_sobol.i4_sobol (dimension, seed)
    out = '%6d %6d  '% (seed, seed_out)
    for j in range(dimension):
      out += '%10f  '% r[j]
    print (out)
    seed = seed_out


def sampler_sobol_test05():
  
  sample = SobolSampler(3)
  for i in range(0, 10+1):
    r = sample.next()
    out = '%6d %6d  '% (sample.seed_in, sample.seed_out)
    for j in range(sample.dimension):
      out += '%10f  '% r[j]
    print(out)

  print('restart')
  sample.init_index(4)
  for i in range(4 , 10+1):
    r = sample.next()
    out = '%6d %6d  ' % (sample.seed_in, sample.seed_out)
    for j in range(sample.dimension):
      out += '%10f  '% r[j]
    print(out)


def sampler3_sobol_test05():
  
  sample = SobolSampler(3)
  for i in range(0, 50000):
    r = sample.next()
    if sample.seed_out != sample.seed_in + 1:
      raise RuntimeError('seed out differs from seed in by more than +1')


def sampler2_sobol_test05():
  sample = SobolSampler(3)
  sample.ranges[0][:] = np.asarray([0,4])
  sample.ranges[1][:] = np.asarray([1,2])
  sample.ranges[2][:] = np.asarray([-1,1])
  
  for i in range(0, 20+1):
    r = sample.next()
    out = '%6d %6d  ' % (sample.seed_in, sample.seed_out)
    for j in range(sample.dimension):
      out += '%10f  '% r[j]
    print(out)


def sampler_halton_test05():
  
  sample = HaltonSampler(3)
  for i in range(8):
    r = sample.next()
    out = '%6d  ' % (i)
    for j in range(sample.dimension):
      out += '%10f  '% r[j]
    print(out)


def sampler2_halton_test05():
  
  sample = HaltonSampler(3)
  sample.ranges[0][:] = np.asarray([0,4])
  sample.ranges[1][:] = np.asarray([1,2])
  sample.ranges[2][:] = np.asarray([-1,1])
  for i in range(8):
    r = sample.next()
    out = '%6d  ' % (i)
    for j in range(sample.dimension):
      out += '%10f  '% r[j]
    print(out)


def sampler3_halton_test05():
  
  sample = HaltonSampler(3)
  sample.ranges[0][:] = np.asarray([0,4])
  sample.ranges[1][:] = np.asarray([1,2])
  sample.ranges[2][:] = np.asarray([-1,1])
  for i in range(8):
    r = sample.next()
    out = '%6d  ' % (i)
    for j in range(sample.dimension):
      out += '%10f  '% r[j]
    print(out)

  print('restart')
  sample.init_index(4)
  for i in range(4, 8):
    r = sample.next()
    out = '%6d  ' % (i)
    for j in range(sample.dimension):
      out += '%10f  '% r[j]
    print(out)


def sampler_tensor_test05():
  
  dimension = 1
  ranges = np.zeros((dimension,2))
  ranges[0][:] = np.asarray([0.0, 4.0])
  sample = TensorProductSampler(dimension, ranges=ranges, partition=[3], include_end_points=False)
  
  dimension = 1
  ranges = np.zeros((dimension,2))
  ranges[0][:] = np.asarray([0.0, 4.0])
  sample = TensorProductSampler(dimension, ranges=ranges, partition=[3], include_end_points=True)

  dimension = 1
  ranges = np.zeros((dimension,2))
  ranges[0][:] = np.asarray([3.0, 4.0])
  sample = TensorProductSampler(dimension, ranges=ranges, partition=[4], include_end_points=True)

  dimension = 2
  ranges = np.zeros((dimension,2))
  ranges[0][:] = np.asarray([0.0, 4.0])
  ranges[1][:] = np.asarray([-1.0, 1.0])
  sample = TensorProductSampler(dimension, ranges=ranges, partition=[3, 3], include_end_points=False)

  dimension = 2
  ranges = np.zeros((dimension,2))
  ranges[0][:] = np.asarray([5.0, 10.0])
  ranges[1][:] = np.asarray([0.01, 0.06])
  sample = TensorProductSampler(dimension, ranges=ranges, partition=[10, 2], include_end_points=False)

  points = sample.get_points()
  npoints = len(points)
  for i in range(npoints):
    r = sample.next()
    out = '%6d  ' % (i)
    for j in range(sample.dimension):
      out += '%10f  '% r[j]
    print(out)


if __name__ == "__main__":
  
  #sampler3_sobol_test05()
  
  #print("Sobol <default>")
  #sobol_test05()
  
  print("SobolSampler")
  sampler_sobol_test05()
  
  print("SobolSampler <with non-default range>")
  sampler2_sobol_test05()

  print("HaltonSampler")
  sampler_halton_test05()
 
  print("HaltonSampler <with non-default range>")
  sampler2_halton_test05()

  print("HaltonSampler <with restart>")
  sampler3_halton_test05()

  print("TensorSampler")
  sampler_tensor_test05()
