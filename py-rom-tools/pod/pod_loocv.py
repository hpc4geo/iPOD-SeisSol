
import numpy as np

from .pod_base import PODBase

def pod_loocv(pod_full, norm_type="linf", **kwargs):
  from time import perf_counter
  
  # Check pod is instance of PODBase
  if isinstance(pod_full, PODBase) == False:
    raise RuntimeError("[Error][pod_loocv] Arg `pod_full` must inherit from PODBase")

  norm_valid = ["l2", "linf"]
  if norm_type not in norm_valid:
    msg = "[Error][pod_loocv] Arg `norm_type` must one of [" + ", ".join(norm_valid) +"]. Found " + norm_type
    raise RuntimeError(msg)
  ntype_l2 = False
  ntype_linf = False
  if norm_type == "l2":
    ntype_l2 = True
  if norm_type == "linf":
    ntype_linf = True

  dimension = len(pod_full.control)
  measure = np.zeros(pod_full.n)
  control_key = list(pod_full.control.keys())

  _pod_class_constructor = pod_full.__class__

  time_pod = 0.0
  time_rbf = 0.0

  print('pod-loocv <init>')
  t0 = perf_counter()
  for k in range(pod_full.n):
  
    control = dict()
    snaps = list()
    for key in control_key:
      control[key] = list()
      if k != 0:
        control[key] += pod_full.control[key][0:k]
        snaps        += pod_full.snapshot    [0:k]
      control[key] += pod_full.control[key][k+1:pod_full.n+1]
      snaps        += pod_full.snapshot    [k+1:pod_full.n+1]

    tA = perf_counter()
    # Determine class for pod_full, define the constructor and then call it
    pod = _pod_class_constructor(control=control, snapshot=snaps, remove_mean=pod_full.remove_mean)
    #pod = PODModel(control, snaps)
    
    pod.setup_basis()
    tB = perf_counter()

    pod.setup_interpolant(**kwargs)

    x_true = pod_full.snapshot[k]
    
    params = np.zeros(dimension)
    for i in range(dimension):
      params[i] = pod_full.control[control_key[i]][k]
    
    x_predict = pod.evaluate(params)
    tC = perf_counter()

    diff = x_true - x_predict

    if norm_type == "l2":
      err = np.linalg.norm(diff)
    if norm_type == "linf":
      err = np.max(np.absolute(diff))

    measure[k] = err
    
    time_pod += (tB - tA)
    time_rbf += (tC - tB)
    
    if k != 0 and k % 25 == 0:
      print('pod-loocv processed snapshot',k)
      t1 = perf_counter()
      print("   time:", t1-t0)
      print("   [pod]:", time_pod)
      print("   [rbf]:", time_rbf)

  t1 = perf_counter()
  print(" total time:", t1-t0)
  
  return measure

def pod_loocv2(pod_full, norm_type="linf", **kwargs):
  from time import perf_counter
  
  # Check pod is instance of PODBase
  if isinstance(pod_full, PODBase) == False:
    raise RuntimeError("[Error][pod_loocv] Arg `pod_full` must inherit from PODBase")
  
  norm_valid = ["l2", "linf"]
  if norm_type not in norm_valid:
    msg = "[Error][pod_loocv] Arg `norm_type` must one of [" + ", ".join(norm_valid) +"]. Found " + norm_type
    raise RuntimeError(msg)
  ntype_l2 = False
  ntype_linf = False
  if norm_type == "l2":
    ntype_l2 = True
  if norm_type == "linf":
    ntype_linf = True
  
  measure = np.zeros(pod_full.n)
  control_key = list(pod_full.control.keys())

  # flatten
  _control = np.zeros((len(control_key),pod_full.n))
  idx = 0
  for key in control_key:
    _control[idx][:] = np.asarray(pod_full.control[key])
    idx += 1

  control = dict()

  _pod_class_constructor = pod_full.__class__

  print('pod-loocv <init>')
  t0 = perf_counter()

  time_pod = 0.0
  time_rbf = 0.0

  for k in range(pod_full.n):

    r0 = np.arange(0,k,1,np.int32)
    r1 = np.arange(k+1,pod_full.n,1,np.int32)
    rrange = np.concatenate([r0,r1])

    control = dict()
    idx = 0
    for key in control_key:
      vals = _control[idx][ rrange ]
      control[key] = list(vals)
      idx += 1
    
    #snaps = pod_full.snapshot[ np.array(rrange) ]
    snaps = [pod_full.snapshot[i] for i in rrange]

    tA = perf_counter()
    # Determine class for pod_full, define the constructor and then call it
    pod = _pod_class_constructor(control=control, snapshot=snaps, remove_mean=pod_full.remove_mean)
    #pod = PODModel(control, snaps)
    
    pod.setup_basis()
    tB = perf_counter()
    pod.setup_interpolant(**kwargs)
    
    x_true = pod_full.snapshot[k]
    
    params = np.zeros(len(control_key))
    for i in range(len(control_key)):
      params[i] = _control[i][k]

    x_predict = pod.evaluate(params)
    tC = perf_counter()
  
    diff = x_true - x_predict
    
    if norm_type == "l2":
      err = np.linalg.norm(diff)
    if norm_type == "linf":
      err = np.max(np.absolute(diff))
    
    measure[k] = err

    time_pod += (tB - tA)
    time_rbf += (tC - tB)

    if k % 25 == 0:
      print('pod-loocv iteration',k)
      t1 = perf_counter()
      print("   time:", t1-t0)
      print("   [pod]:", time_pod)
      print("   [rbf]:", time_rbf)

  t1 = perf_counter()
  print(" total time:", t1-t0)
  
  return measure

