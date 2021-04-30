
import numpy as np

import pod as podtools

def func(a, xp):
  f = np.sin(a*xp)**2 + 1*a**4 * xp
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
  
  pod = podtools.PODMultivariate(remove_mean=False)
  pod.database_append(c, [v0, v1, v2, v3, v4, v5])
  pod.setup_basis()
  print('singular values:\n',pod.singular_values)
  ric = pod.get_ric()
  print('RIC:\n',ric)
  
  pod.setup_interpolant()
  #pod.setup_interpolant(shape_parameter_opt = True)
  #pod.setup_interpolant(bounds_auto=True)
  lb = pod.interpolant.bound_lower
  ub = pod.interpolant.bound_upper
  
  x = pod.evaluate([1.34])
  x_true = func(1.34,xp)
  print('x\n',x)
  print('x_true\n',x_true)
  linf = np.max(np.absolute(x_true-x))
  print('|x-x_true|_inf\n',('%1.4e'%linf))
  
  
  m = podtools.pod_loocv(pod)
  #m = podtools.pod_loocv(pod, shape_parameter_opt = True)
  #m = podtools.pod_loocv(pod, bound_lower=lb, bound_upper=ub)
  print('measure\n',m)
  
  print(pod)

def test_loocv_pod():
  xp = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
  
  c = {"a": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]}
  
  v0 = func(c["a"][0],xp)
  v1 = func(c["a"][1],xp)
  v2 = func(c["a"][2],xp)
  v3 = func(c["a"][3],xp)
  v4 = func(c["a"][4],xp)
  v5 = func(c["a"][5],xp)
  
  pod = podtools.PODMultivariate(remove_mean=False)
  pod.database_append(c, [v0, v1, v2, v3, v4, v5])
  pod.setup_basis()
  print('singular values:\n',pod.singular_values)
  ric = pod.get_ric()
  print('RIC:\n',ric)
  
  #pod.setup_interpolant()
  #pod.setup_interpolant(shape_parameter_opt = True)
  pod.setup_interpolant(bounds_auto=True)
  lb = pod.interpolant.bound_lower
  ub = pod.interpolant.bound_upper
  
  x = pod.evaluate([1.34])
  x_true = func(1.34,xp)
  print('x\n',x)
  print('x_true\n',x_true)
  linf = np.max(np.absolute(x_true-x))
  print('|x-x_true|_inf\n',('%1.4e'%linf))
  
  
  #m = podtools.pod_loocv(pod)
  #m = podtools.pod_loocv(pod, shape_parameter_opt = True)
  m = podtools.pod_loocv(pod, bound_lower=lb, bound_upper=ub)
  print('measure\n',m)

  m = podtools.rbf_loocv(pod)
  print('measure\n',m)


  print(pod)

def test1HR():
  xp = np.linspace(0.0, 0.5, 1280000)

  avals = np.linspace(1.0, 1.5, 50)

  c = {"a": list(avals)}
  
  snaps = list()
  for cval in avals:
    v = func(cval,xp)
    snaps.append(v)

  pod = podtools.PODMultivariate(remove_mean=True)
  pod.database_append(c, snaps)
  pod.setup_basis()
  #print('singular values:\n',pod.singular_values)
  #ric = pod.get_ric()
  #print('RIC:\n',ric)
  
  pod.setup_interpolant()
  x = pod.evaluate([1.34])
  x_true = func(1.34,xp)
  linf = np.max(np.absolute(x_true-x))
  print('|x-x_true|_inf\n',('%1.4e'%linf))
  
  m = podtools.pod_loocv(pod)
  #print('measure\n',m)
  #print('||measure||\n',np.linalg.norm(m))
  #m = podtools.rbf_loocv(pod)
  #print('measure\n',m)
  print('||measure||\n',np.linalg.norm(m))
  
  print(pod)

def test_compress_pod():
  M = 30*10
  xp = np.linspace(0.0, 0.5, M)
  N = 60
  avals = np.linspace(1.0, 1.5, N)
  avls = 0.5 * np.random.rand(N) + 1.0
  print(avls)
  
  c = {"a": list(avals)}
  
  snaps = list()
  for cval in avals:
    v = func(cval,xp)
    snaps.append(v)
  
  pod = podtools.PODMultivariate(remove_mean=False)
  pod.database_append(c, snaps)
  pod.setup_basis()
  print('singular values:\n',pod.singular_values)
  ric = pod.get_ric()
  print('RIC:\n',ric)
  ric_range = ric[ric <= 0.9999]
  print('RIC[99.99%]:\n',ric_range)
  n_ric = len(ric_range) + 1 # +1 is for safety
  if n_ric < 4:
    n_ric = 4

  #n_ric = 8
  print('======== #RIC',n_ric,'========')

  pod.setup_interpolant()
  #pod.setup_interpolant(shape_parameter_opt = True)
  #pod.setup_interpolant(bounds_auto=True)
  #lb = pod.interpolant.bound_lower
  #ub = pod.interpolant.bound_upper
  
  P = 1.34
  #P = avals[9]

  x = pod.evaluate([P])
  x_true = func(P,xp)
  #print('x\n',x)
  #print('x_true\n',x_true)
  linf = np.max(np.absolute(x_true-x))
  print('|x-x_true|_inf\n',('%1.10e'%linf))
  
  
  maxerr = 0
  t = np.linspace(1.0, 1.5, 4000, dtype=np.float32)
  for time in t:
    x_predict = pod.evaluate([time])
    x_true = func(time,xp)
    linf = np.max(np.absolute(x_true-x_predict))
    if linf > maxerr:
      maxerr = linf
  print('max error <full>',('%1.4e'%maxerr))

  
  m = podtools.pod_loocv(pod)
  #m = podtools.pod_loocv(pod, shape_parameter_opt = True)
  #m = podtools.pod_loocv(pod, bound_lower=lb, bound_upper=ub)
  print('measure\n',m)
  linf = np.max(np.absolute(m))
  print('|measure|_inf <pod>\n',('%1.4e'%linf))
  m = podtools.rbf_loocv(pod)
  linf = np.max(np.absolute(m))
  print('|measure|_inf <pod-rbf>\n',('%1.4e'%linf))

  idx = np.argsort(m)
  print('idx',idx)

  ms = m[idx]
  print('measure(sort)\n',ms)

  ms_s =  ms[ms > 1.0e-5]
  n_ms = len(ms_s)
  print('ms_s',ms_s)

  #n_basis = N-12
  n_basis = N - n_ric
  n_basis = N - n_ms
  n_basis = 15         # if n_basis > 15, the loocv closesly tracks track max error. Why?


  sizes = pod.coeff.shape
  
  #x = pod.evaluate_rank([P], n_ric)
  #x_true = func(P,xp)
  #linf = np.max(np.absolute(x_true-x))
  #print('|x-x_true|_inf <low rank>\n',('%1.10e'%linf))


  chop = False
  if chop:
    for i in range(sizes[0]-n_basis, sizes[0]):
      #pod.coeff[i, :] = 0.0 # Don't require this
      pod.coeff[:, i] = 0.0

    x = pod.evaluate([P])
    x_true = func(P,xp)
    linf = np.max(np.absolute(x_true-x))
    print('|x-x_true|_inf <chopped basis>\n',('%1.10e'%linf))


  #keepers = idx[-n_basis:]
  keepers = idx[n_basis:N]
  print('[keep]',keepers)
  print('[keep]',np.sort(keepers),'(sorted)')


  snaps = list()
  for i in range(len(keepers)):
    x = pod.snapshot[keepers[i]]
    snaps.append(x)

  cr = {"a": list(avals[keepers])}
  print('cr',cr)

  podr = podtools.PODMultivariate(remove_mean=False)
  podr.database_append(cr, snaps)

  podr.setup_basis()
  print('singular values:\n',podr.singular_values)


  podr.setup_interpolant()
  #podr.setup_interpolant(shape_parameter_opt = True)
  #podr.setup_interpolant(bound_lower = lb, bound_upper = ub)

  x = podr.evaluate([P])
  x_true = func(P,xp)
  #print('x\n',x)
  #print('x_true\n',x_true)
  linf = np.max(np.absolute(x_true-x))
  print('|x-x_true|_inf\n',('%1.4e'%linf))
  
  
  #m = podtools.pod_loocv(podr)
  #m = podtools.pod_loocv(podr, shape_parameter_opt = True)
  #m = podtools.pod_loocv(podr, bound_lower=lb, bound_upper=ub)
  m = podtools.rbf_loocv(podr)
  print('measure\n',m)
  linf = np.max(np.absolute(m))
  print('|measure|_inf <podr-rbf>\n',('%1.4e'%linf))

  maxerr = 0
  t = np.linspace(1.0, 1.5, 4000, dtype=np.float32)
  for time in t:
    x_predict = podr.evaluate([time])
    x_true = func(time,xp)
    linf = np.max(np.absolute(x_true-x_predict))
    if linf > maxerr:
      maxerr = linf
  print('max error <red>',('%1.4e'%maxerr))


  ex = [12, 9]
  ex = [12,  9, 10,  7,  8, 11,  6, 13]
  #ex = [12,  9, 10,  7,  8, 11,  6, 13, 14,  5, 15,  4, 16,  3, 17,  2]
  ex = list()
  for i in range(N):
    ex.append(i)
  for i in range(len(keepers)):
    ex[keepers[i]] = -1
  ex2 = [i for i in ex if i >= 0]
  ex = ex2

  m = podtools.rbf_loocv_n(pod, ex)
  print('measure[filter]\n',m)
  linf = np.max(np.absolute(m))
  print('|measure|_inf <pod-rbf-exclude>\n',('%1.4e'%linf))


  #
  probe = False
  if probe:
    print('=== scan ===')
    for b in range(1,N-2+1):
      keepers = idx[b:N]

      ex = list()
      for i in range(N):
        ex.append(i)
      for i in range(len(keepers)):
        ex[keepers[i]] = -1
      ex2 = [i for i in ex if i >= 0]
      ex = ex2
      
      m = podtools.rbf_loocv_n(pod, ex)
      linf = np.max(np.absolute(m))
      print('  keeping',str(N-b),'basis; |measure|_inf',('%1.4e'%linf))

  nsamples = 5000
  bin_size = 2
  vv = np.zeros(nsamples)
  #avg = 0.0
  idx = 0
  for bin in range(nsamples):
    exclude = np.arange(N)
    np.random.shuffle(exclude)
    ex = exclude[:bin_size]

    m = podtools.rbf_loocv_n(pod, ex)
    linf = np.linalg.norm(np.absolute(m))
    vv[idx] = linf
    #avg += linf
    idx += 1
  #avg /= float(nsamples)
  #print('  k-fold mean, sigma |measure|_inf',('%1.4e'%avg))
  avg = np.mean(vv)
  dev = np.std(vv)
  print('  k-fold mean, sigma |measure|_inf',('%1.4e'%avg),'+/-',('%1.4e'%dev))


if __name__ == "__main__":
  #test1()
  #test1HR()
  #test_loocv_pod()
  test_compress_pod()
