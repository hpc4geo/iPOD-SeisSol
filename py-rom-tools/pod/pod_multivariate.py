
import numpy as np

from .pod_base  import PODBase
from .rbf_affine import RBFAffine
from .pod_rbf_loocv import rbf_loocv, rbf_loocv_n


class PODMultivariate(PODBase):
  
  def __init__(self, control=None, snapshot=None, remove_mean=True):
    PODBase.__init__(self, control, snapshot, remove_mean)
  
  
  def setup_interpolant(self, **kwargs):
    if self.issetup_basis is False:
      raise RuntimeError('POD basis not setup - must call pod.setup_basis() first')

    dimension = len(self.control)
    for key in self.control.keys():
      npoints = len( self.control[key] ) # use key/value from active_control

    params = np.zeros((dimension,npoints))
    index = 0
    for key in self.control.keys():
      v = self.control[key] # use key/value from active_control
      for j in range(npoints):
        params[index,j] = v[j]
      index += 1

    values = np.zeros(npoints)
    values[:] = 1.0

    low = None
    high = None
    try:
      flg = kwargs['bounds_auto']
      #print('Using auto bounds')
      if flg == True:
        low = np.zeros(dimension)
        high = np.zeros(dimension)
        index = 0
        for key in self.control.keys():
          v = self.control[key] # use key/value from active_control
          low[index] = np.min(v)
          high[index] = np.max(v)
          index += 1
    except:
      pass

    try:
      low = kwargs['bound_lower']
      #print('Using user lower bound')
    except:
      pass
    try:
      high = kwargs['bound_upper']
      #print('Using user upper bound')
    except:
      pass

    rbf_type = 'mq'
    try:
      rbf_type = kwargs['rbf_type']
    except:
      pass

    #interp = RBFAffine(dimension, rbftype = 'polyh', solvertype = 'svd', perturb = 0.0e-12)

    #interp = RBFAffine(dimension, rbftype = 'mq', solvertype = 'svd', perturb = 0.0e-12)

    interp = RBFAffine(dimension, rbftype = rbf_type, solvertype = 'svd', perturb = 0.0e-12)
  

    try:
      eps = kwargs['shape_parameter']
      interp.shape_parameter = eps
    except:
      pass
      
    try:
      flg = kwargs['shape_parameter_opt']
      if flg == True and interp.basis_type != 'polyh':
        alpha = alpha_optimzer_bf(params)
        print('[PODMultivariate][setup_interpolant] Using optimized alpha =',alpha)
        interp.shape_parameter = alpha
    except:
      pass

    if low is not None and high is not None:
      interp.set_bounds(low, high)
    interp.setup(npoints, params, values)

#    interp = list()
#    for k in range(self.n):
#      vals = np.zeros(self.n)
#      vals[k] = 1.0
#      rbf = RBFAffine(dimension, rbftype = 'polyh', solvertype = 'svd', perturb = 0.0e-12)
#      #rbf.set_bounds(low, high)
#      rbf.setup(npoints, params, vals)
#      interp.append(rbf)


    self.interpolant = interp # store

    self.issetup_interpolant = True

  def get_w(self, rbf, iparams):
    D = rbf.distance_matrix_general(1, iparams)
    Dp = np.zeros((1, rbf.npoints+1))
    Dp[0:1,0:rbf.npoints] = D
    A = rbf.build_rbf(0, rbf.shape_parameter, Dp)
    A[:, rbf.npoints] = 1

    rhs = np.zeros(rbf.npoints+1)
    for i in range(rbf.npoints+1):
      rhs[i] = A[0][i]

    #Nt = np.matmul(A, rbf.Ainv)
    #u, s, vh = np.linalg.svd(rbf.A, full_matrices=True)
    #u, s, vh = rbf.U, rbf.S, rbf.Vh
    #x = np.zeros(rbf.npoints+1)
    #for i in range(0,len(s)):
    #  fac = np.dot(u[:,i],rhs) / s[i]
    #  x = x + fac * vh[i,:]
    x = rbf.action_A_inverse(rhs)

    w = x[0:rbf.npoints]

    return w


  def evaluate(self, cval):
    if self.issetup_interpolant is False:
      raise RuntimeError('Interpolant not setup - must call pod.setup_interpolant() first')
    
    rbf = self.interpolant
    dimension = rbf.dimension
    if len(cval) != dimension:
      raise RuntimeError('Control vector must have length ' + str(dimension) + '. Found ' + str(len(cval)))

    npoints   = rbf.npoints
    nsample = 1
    iparams = np.zeros((dimension, nsample))
    for d in range(dimension):
      iparams[d,0] = cval[d]

    weight = self.get_w(rbf, iparams)

    variation = np.matmul(self.phi_normalized, self.coeff)
    prediction = np.matmul(variation, weight)
    if self.Umean is not None:
      prediction += self.Umean

    return prediction


  def evaluate_rank(self, cval, N):
    if self.issetup_interpolant is False:
      raise RuntimeError('Interpolant not setup - must call pod.setup_interpolant() first')
    
    rbf = self.interpolant
    dimension = rbf.dimension
    
    if len(cval) != dimension:
      raise RuntimeError('Control vector must have length ' + str(dimension) + '. Found ' + str(len(cval)))

    npoints   = rbf.npoints
    nsample = 1
    iparams = np.zeros((dimension, nsample))
    for d in range(dimension):
      iparams[d,0] = cval[d]
  
    weight = self.get_w(rbf, iparams)
    
    T_recon = np.zeros(self.m)
    for j in range(0,N):
      Tb = np.matmul(self.phi_normalized , self.coeff[:,j])
      T_recon = T_recon + weight[j] * Tb
    if self.Umean is not None:
      T_recon += self.Umean

    return T_recon


