
import numpy as np

import pod as podtools
from pod import PODUnivariate


def func(a, xp):
  f = np.sin(a*xp)**2
  return f

def test1():
  xp = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
  
  c = {"a": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]}

  v0 = func(c["a"][0],xp)
  v1 = func(c["a"][1],xp)
  v2 = func(c["a"][2],xp)
  v3 = func(c["a"][3],xp)
  v4 = func(c["a"][4],xp)
  v5 = func(c["a"][5],xp)

  pod = PODUnivariate(remove_mean=False)
  pod.database_append(c, [v0, v1, v2, v3, v4, v5])
  pod.setup_basis()
  print('singular values:\n',pod.singular_values)
  ric = pod.get_ric()
  print('RIC:\n',ric)

  pod.setup_interpolant()
  x = pod.evaluate([1.34])
  print('x\n',x)
  print('x_true\n',func(1.34,xp))

  m = podtools.pod_loocv(pod)
  print('measure\n',m)

  print(pod)

def test2():
  xp = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
  
  c = {"a": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]}
  
  v0 = func(c["a"][0],xp)
  v1 = func(c["a"][1],xp)
  v2 = func(c["a"][2],xp)
  v3 = func(c["a"][3],xp)
  v4 = func(c["a"][4],xp)
  v5 = func(c["a"][5],xp)
  
  pod = PODUnivariateB(remove_mean=False)
  pod.database_append(c, [v0, v1, v2, v3, v4, v5])
  pod.setup_basis()
  print('singular values:\n',pod.singular_values)
  ric = pod.get_ric()
  print('RIC:\n',ric)
  
  pod.setup_interpolant()
  x = pod.evaluate([1.34])
  print('x\n',x)
  
  #m = podtools.pod_loocv(pod)
  
  
  index = 0
  xi = np.linspace(1.0,1.5,256)
  fp = open('basis.gp','w')
  for k in range(len(xi)):
    weight = np.zeros(pod.n)
    for j in range(pod.n):
      #weight[j] = interpolate.splev([xi[k]], pod.splinebasis[j]) # Univariate
      weight[j] = pod.splinebasis[j](xi[k]) # UnivariateC, UnivariateB
    weight = weight[pod.iperm]
    fp.write(str(xi[k]) + ' ' + str(weight[0]) + ' ' + str(weight[1]) + ' ' + str(weight[2]) + '\n')
  fp.close()


if __name__ == "__main__":
  test1()
  #test2()

