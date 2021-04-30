
import numpy as np
from .pod_base import PODBase
from . import rbf_affine as rbfaffine

def evaluate_weight_loocv(pod_full, k, cval):
  
  r0 = np.arange(0, k, 1, np.int32)
  r1 = np.arange(k+1, pod_full.n, 1, np.int32)
  insert = np.arange(0, pod_full.n-1, 1, np.int32)
  select = np.concatenate([r0, r1])
  #print('insert',insert)
  #print('select',select)

  dimension = len(pod_full.control)
  npoints = pod_full.n
  control_key = list(pod_full.control.keys())

  #params = np.zeros((dimension,npoints))
  #index = 0
  #for key in pod_full.control.keys():
  #  v = pod_full.control[key] # use key/value from active_control
  #  for j in range(npoints):
  #    params[index,j] = v[j]
  #  index += 1

  npoints_red = npoints - 1
  params_red = np.zeros((dimension,npoints_red))
  for d in range(dimension):
    v = np.asarray(pod_full.control[control_key[d]]) # use key/value from active_control
    params_red[d, insert] = v[select]

  values_red = np.zeros(npoints_red)
  values_red[:] = 1.0


  #interp = rbfaffine.RBFAffine(dimension, rbftype = 'polyh', solvertype = 'svd', perturb = 0.0e-12)
  #interp.set_bounds(low, high)
  #interp.setup(npoints_red, params_red, values_red)

  interp = rbfaffine.RBFAffine(dimension, rbftype = pod_full.interpolant.basis_type, solvertype = 'svd', perturb = pod_full.interpolant.diag_perturb)

  interp.shape_parameter = pod_full.interpolant.shape_parameter

  interp.set_bounds(pod_full.interpolant.bound_lower, pod_full.interpolant.bound_upper)

  interp.setup(npoints_red, params_red, values_red)

  nsample = 1
  iparams = np.zeros((dimension, nsample))
  for d in range(dimension):
    iparams[d,0] = cval[d]

  D = interp.distance_matrix_general(nsample, iparams)
  Dp = np.zeros((nsample, npoints_red+1))
  Dp[0:nsample, 0:npoints_red] = D
  A = interp.build_rbf(0, interp.shape_parameter, Dp)
  A[:, npoints_red] = 1
  
  #Nt = np.matmul(A, interp.Ainv)
  #
  #weight_loocv = np.zeros(npoints_red)
  #for j in range(npoints_red):
  #  weight_loocv[j] = Nt[0,j]

  rhs = np.zeros(npoints_red+1)
  for i in range(npoints_red+1):
    rhs[i] = A[0][i]
  x = interp.action_A_inverse(rhs)
  weight_loocv = x[0:interp.npoints]


  weight = np.zeros(pod_full.n)
  weight[select] = weight_loocv[insert]
  
  return weight


def evaluate(self, weight, vars=None):
  if vars is None:
    variation = np.matmul(self.phi_normalized, self.coeff)
  else:
    variation = vars

  prediction = np.matmul(variation, weight)
  if self.Umean is not None:
    prediction += self.Umean

  return prediction



def rbf_loocv(pod_full, norm_type="linf", **kwargs):
  from time import perf_counter
  
  # Check pod is instance of PODBase
  if isinstance(pod_full, PODBase) == False:
    raise RuntimeError("[Error][pod_loocv] Arg `pod_full` must inherit from PODBase")
  
  norm_valid = ["l2", "linf", "rms"]
  if norm_type not in norm_valid:
    msg = "[Error][pod_loocv] Arg `norm_type` must one of [" + ", ".join(norm_valid) +"]. Found " + norm_type
    raise RuntimeError(msg)

  dimension = len(pod_full.control)
  measure = np.zeros(pod_full.n)
  control_key = list(pod_full.control.keys())

  time_w = 0.0
  time_e = 0.0

  variation = np.matmul(pod_full.phi_normalized, pod_full.coeff)

  print('rbf_loocv <init>')
  t0 = perf_counter()
  for k in range(pod_full.n):

    x_true = pod_full.snapshot[k]

    cval = np.zeros(dimension)
    for d in range(dimension):
      cval[d] = pod_full.control[control_key[d]][k]


    tA = perf_counter()
    weight = evaluate_weight_loocv(pod_full, k, cval)
    tB = perf_counter()
    time_w += tB - tA

    tA = perf_counter()
    x_predict = evaluate(pod_full, weight, vars=variation)
    tB = perf_counter()
    time_e += tB - tA

    diff = x_true - x_predict

    if norm_type == "l2":
      err = np.linalg.norm(diff)
    if norm_type == "linf":
      err = np.max(np.absolute(diff))
      #location = np.argsort(np.absolute(diff))
      #print(k,'location',location[-1])
    if norm_type == "rms":
      err = np.linalg.norm(diff) / np.sqrt(float(len(diff)))

    measure[k] = err
    
    if k != 0 and k % 25 == 0:
      print('rbf_loocv processed snapshot',k)
      t1 = perf_counter()
      print("   time:", t1-t0)
      print("     weight:", time_w)
      print("     eval  :", time_e)

  t1 = perf_counter()
  print(" total time:", t1-t0)
  
  return measure






def evaluate_weight_loocv_n(pod_full, k_exclude, cval):

  # Create two arrays, containing indices for the reference and those to exclude
  ridx = np.arange(0, pod_full.n, 1, np.int32)  #[0, 1, 2, 3, 4]
  ridx += 1 #[1, 2, 3, 4, 5]
  ridx_ex = np.zeros(pod_full.n, dtype=np.int32)
  for k in k_exclude:
    ridx_ex[k] = k + 1 #k_exclude = [3, 1] ==> ridx_ex = [0, 2, 0, 4, 0]
  select = ridx - ridx_ex - 1 #[1, 2, 3, 4, 5] - [0, 2, 0, 4, 0] - 1 => [0, -1, 2, -1, 4]
  select = select[select >= 0] # < 0 => index removed

  #print('[selected]',select)

  dimension = len(pod_full.control)
  npoints = pod_full.n
  control_key = list(pod_full.control.keys())

  np_exclude = len(k_exclude)
  npoints_red = npoints - np_exclude
  params_red = np.zeros((dimension,npoints_red))
  for d in range(dimension):
    v = np.asarray(pod_full.control[control_key[d]]) # use key/value from active_control
    params_red[d, :] = v[select]
  
  values_red = np.zeros(npoints_red)
  values_red[:] = 1.0
  
  interp = rbfaffine.RBFAffine(dimension, rbftype = pod_full.interpolant.basis_type, solvertype = 'svd', perturb = pod_full.interpolant.diag_perturb)
  interp.shape_parameter = pod_full.interpolant.shape_parameter
  interp.set_bounds(pod_full.interpolant.bound_lower, pod_full.interpolant.bound_upper)
  interp.setup(npoints_red, params_red, values_red)

  weights = list()
  for k in range(np_exclude):
    nsample = 1
    iparams = np.zeros((dimension, nsample))
    for d in range(dimension):
      iparams[d,0] = cval[k][d]
    
    D = interp.distance_matrix_general(nsample, iparams)
    Dp = np.zeros((nsample, npoints_red+1))
    Dp[0:nsample, 0:npoints_red] = D
    A = interp.build_rbf(0, interp.shape_parameter, Dp)
    A[:, npoints_red] = 1
    
    rhs = np.zeros(npoints_red+1)
    for i in range(npoints_red+1):
      rhs[i] = A[0][i]
    x = interp.action_A_inverse(rhs)
    weight_loocv = x[0:interp.npoints]

    weight = np.zeros(pod_full.n)
    weight[select] = weight_loocv[:]
    weights.append(weight)

  return weights

###
def rbf_loocv_n(pod_full, k_exclude, norm_type="linf", **kwargs):
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
  
  view = False
  if "view" in kwargs:
    view = True
  
  dimension = len(pod_full.control)
  control_key = list(pod_full.control.keys())

  variation = np.matmul(pod_full.phi_normalized, pod_full.coeff)

  if view:
    print('rbf_loocv_n <init>')
    print('[exclude]',k_exclude)

  cvals = list()
  for k in k_exclude:
    _cval = np.zeros(dimension)
    for d in range(dimension):
      _cval[d] = pod_full.control[control_key[d]][k]
    cvals.append(_cval)

  weights = evaluate_weight_loocv_n(pod_full, k_exclude, cvals)

  measure = np.zeros(len(k_exclude))
  idx = 0
  for k in k_exclude:
    x_true = pod_full.snapshot[k]
    x_predict = evaluate(pod_full, weights[idx], vars=variation)
    diff = x_true - x_predict
    if norm_type == "l2":
      err = np.linalg.norm(diff)
    if norm_type == "linf":
      err = np.max(np.absolute(diff))
    #print('k',k,'err',('%1.6e'%err))
    measure[idx] = err
    idx += 1

  return measure



