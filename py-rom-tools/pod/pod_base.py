
import numpy as np

class PODBase:

  def setup_interpolant(self, **kwargs):
    raise RuntimeError("Base class does not implement method setup_interpolant()")
  
  def evaluate(self, cvals):
    """
    cvals: Control parameters.
    """
    raise RuntimeError("Base class does not implement method evaluate()")
  
  def evaluate_gradient(self, cvals, cindex):
    """
    cvals: Control parameters.
    cindex: Index of control we want the derivative with respect to.
    """
    raise RuntimeError("Base class does not implement method evaluate_gradient()")
  
  def evaluate_at(self, cvals, index):
    """
    cvals: Control parameters.
    index: Index of point in the POD basis we want to extract.
    """
    raise RuntimeError("Base class does not implement method evaluate_at()")

  def tabulate_basis(self, nsamples, prefix):
    """
    nsamples: Number of points used to sample the interpolation basis.
    prefix: Name we append to each file.
    """
    raise RuntimeError("Base class does not implement method tabulate_basis()")


  def database_append(self, c_dict, s_list):
    
    # Check sizes
    N = len(s_list)
    for k in c_dict.keys():
      if len(c_dict[k]) != N:
        raise RuntimeError('Key',k,'maps to list of length',str(len(c_dict[k])),', from `s_list` the expected length is',str(N))
    
    self.snapshot += s_list
    
    if len(self.control) == 0:
      for k in c_dict.keys():
        self.control[k] = list()
  
    for k in self.control.keys():
      try:
        self.control[k] += c_dict[k]
      except:
        raise RuntimeError('Key',k,'not found in `c_dict`')

    self.m = len(self.snapshot[0])
    self.n = len(self.snapshot)


  def __init__(self, control=None, snapshot=None, remove_mean=True):
    self.m = 0
    self.n = 0
    self.snapshot = list()
    self.control = dict()
    self.Umean = None
    self.phi_normalized = None
    self.coeff = None
    self.remove_mean = remove_mean
    self.singular_values = None
    self.issetup_basis = False
    self.issetup_interpolant = False

    if control is not None and snapshot is not None:
      self.database_append(control, snapshot)


  def __str__(self):
    block =  "POD class: " + type(self).__name__ + "\n"
    block += "POD basis length: m = " + str(self.m) + "\n"
    block += "Number of snapshots: n = " + str(self.n) + "\n"
    block += "Snapshot matrix forced to have zero mean: " + str(self.remove_mean) + "\n"
    block += "Dimension of parameter space: d = " + str(len(self.control)) + "\n"
    block += "Basis setup: " + str(self.issetup_basis) + "\n"
    block += "Interpolant setup: " + str(self.issetup_interpolant)
    return block


  def setup_basis(self):
    from time import perf_counter
    
    rm_mean = self.remove_mean

    Ut =  np.zeros((self.n, self.m))
    for i in range(self.n):
      Ut[i,:] = self.snapshot[i]
    U = np.copy(np.transpose(Ut)) # Copy since svd will need a copy anyway
    del Ut
    
    # For U to have zero mean (row-wise)
    if rm_mean:
      self.Umean = U.mean(1)
      U = U - self.Umean[:, None]

    t0 = perf_counter()
    u, s, _ = np.linalg.svd(U, full_matrices=False)
    t1 = perf_counter()
    #print("[POD][SVD factorisation] time:", t1-t0)

    self.singular_values = np.copy(s)
    
    t0 = perf_counter()
    np.multiply(u, s, out=u) # column wise scaling by s
    cnorm = np.sum(u**2, axis=0)**(1./2) # column wise norm
    np.multiply(u, 1/cnorm, out=u) # 2x memory for u required - should be no
    self.phi_normalized = u
    del u, s
    t1 = perf_counter()
    #print("[POD][Build \hat{phi}] time:", t1-t0)
    
    t0 = perf_counter()
    #self.coeff = np.matmul(self.phi_normalized.transpose(), U)
    self.coeff = np.matmul(np.transpose(self.phi_normalized), U)
    del U
    t1 = perf_counter()
    #print("[POD][Build coefficients] time:", t1-t0)

    self.issetup_basis = True


  # Compute Relative Information Content (RIC)
  def get_ric(self):
    s2 = np.sqrt(self.singular_values)  # eigenvalues are sqrt(s)
    s2sum = s2.sum()
    energy = np.zeros(self.n)
    for k in range(self.n-1):
      k_s2sum = s2[0:k+1].sum()
      energy[k] = k_s2sum/s2sum
    energy[self.n-1] = 1.0
    return energy


