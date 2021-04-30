
import numpy as np

from .pod_base import PODBase


from scipy import interpolate

class PODUnivariate(PODBase):
  
  def __init__(self, control=None, snapshot=None, remove_mean=True):
    PODBase.__init__(self, control, snapshot, remove_mean)


  def setup_interpolant(self, **kwargs):
    if self.issetup_basis is False:
      raise RuntimeError('POD basis not setup - must call pod.setup_basis() first')

    keys = list(self.control.keys())
    control = self.control[keys[0]]
    
    # splrep() requires unique and sorted input values
    control_ = np.asarray(control)
    control_.sort()
    
    perm = np.argsort(control)
    iperm = np.zeros(perm.size, dtype=np.int32)
    for k in np.arange(perm.size):
      iperm[perm[k]] = k
    self.iperm = iperm # store
    
    basis = list()
    for k in range(self.n):
      vals = np.zeros(self.n)
      vals[k] = 1.0
      tck = interpolate.splrep(control_, vals, k=3)
      basis.append(tck)
    
    self.splinebasis = basis # store
    self.issetup_interpolant = True


  def evaluate(self, cval):
    if self.issetup_interpolant is False:
      raise RuntimeError('Spline interpolant not setup - must call pod.setup_interpolant() first')

    weight = np.zeros(self.n)
    for j in range(self.n):
      weight[j] = interpolate.splev(cval[0], self.splinebasis[j])
    weight = weight[self.iperm]
    
    tp = np.matmul(self.phi_normalized, self.coeff)
    prediction = np.matmul(tp, weight)
    if self.Umean is not None:
      prediction += self.Umean

    return prediction


from scipy.interpolate import CubicSpline

class PODUnivariateC(PODBase):
  
  def __init__(self, control=None, snapshot=None, remove_mean=True):
    PODBase.__init__(self, control, snapshot, remove_mean)
  
  
  def setup_interpolant(self, **kwargs):
    if self.issetup_basis is False:
      raise RuntimeError('POD basis not setup - must call pod.setup_basis() first')
    
    keys = list(self.control.keys())
    control = self.control[keys[0]]
    
    # splrep() requires unique and sorted input values
    control_ = np.asarray(control)
    control_.sort()
    
    perm = np.argsort(control)
    iperm = np.zeros(perm.size, dtype=np.int32)
    for k in np.arange(perm.size):
      iperm[perm[k]] = k
    self.iperm = iperm # store
    
    basis = list()
    for k in range(self.n):
      vals = np.zeros(self.n)
      vals[k] = 1.0
      tck = CubicSpline(control_, vals)
      basis.append(tck)

    grad_basis = list()
    for k in range(self.n):
      tck = basis[k].derivative(nu=1)
      grad_basis.append(tck)

    self.splinebasis = basis # store
    self.splinebasisgrad = grad_basis # store
    self.issetup_interpolant = True
  
  
  def evaluate(self, cval):
    if self.issetup_interpolant is False:
      raise RuntimeError('Spline interpolant not setup - must call pod.setup_interpolant() first')
    
    weight = np.zeros(self.n)
    for j in range(self.n):
      weight[j] = self.splinebasis[j](cval[0])
    weight = weight[self.iperm]
    
    tp = np.matmul(self.phi_normalized, self.coeff)
    prediction = np.matmul(tp, weight)
    if self.Umean is not None:
      prediction += self.Umean
    
    return prediction


  def evaluate_derivative(self, cval):
    if self.issetup_interpolant is False:
      raise RuntimeError('Spline interpolant not setup - must call pod.setup_interpolant() first')
    
    weight = np.zeros(self.n)
    for j in range(self.n):
      weight[j] = self.splinebasis[j](cval[0])
    weight = weight[self.iperm]
    
    tp = np.matmul(self.phi_normalized, self.coeff)
    prediction = np.matmul(tp, weight)

    return prediction


class PODUnivariateB(PODBase):
  
  def __init__(self, control=None, snapshot=None, remove_mean=True):
    PODBase.__init__(self, control, snapshot, remove_mean)
  
  
  def setup_interpolant(self, **kwargs):
    if self.issetup_basis is False:
      raise RuntimeError('POD basis not setup - must call pod.setup_basis() first')
    
    keys = list(self.control.keys())
    control = self.control[keys[0]]
    
    # splrep() requires unique and sorted input values
    control_ = np.asarray(control)
    control_.sort()
    
    perm = np.argsort(control)
    iperm = np.zeros(perm.size, dtype=np.int32)
    for k in np.arange(perm.size):
      iperm[perm[k]] = k
    self.iperm = iperm # store
    
    basis = list()
    for k in range(self.n):
      vals = np.zeros(self.n)
      vals[k] = 1.0
      tck = interpolate.BSpline(control_, vals, 2)
      basis.append(tck)
    
    self.splinebasis = basis # store
    self.issetup_interpolant = True
  
  
  def evaluate(self, cval):
    if self.issetup_interpolant is False:
      raise RuntimeError('Spline interpolant not setup - must call pod.setup_interpolant() first')
    
    weight = np.zeros(self.n)
    for j in range(self.n):
      weight[j] = self.splinebasis[j](cval[0])
    weight = weight[self.iperm]
    
    tp = np.matmul(self.phi_normalized, self.coeff)
    prediction = np.matmul(tp, weight)
    if self.Umean is not None:
      prediction += self.Umean
    
    return prediction
