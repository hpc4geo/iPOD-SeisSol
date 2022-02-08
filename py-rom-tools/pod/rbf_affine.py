
import os
import numpy as np
import time

class RBFAffine:

  def __init__(self,dimension,**kwargs):
    self.npoints = 0
    self.dimension = dimension
    self.points = None
    self.coeff = None
    self.coeff_constant = 0.0
    self.shape_parameter = 18.1*18.1
    self.Ainv = None
    self.bound_lower = np.zeros(self.dimension)
    self.bound_upper = np.zeros(self.dimension)
    self.bound_upper[:] = 1.0
    self.basis_type = 'polyh'
    self.solver_type = 'lu'
    self.diag_perturb = None
    for key, value in kwargs.items():
      if key == "rbftype":
        self.basis_type = value
      if key == "solvertype":
          self.solver_type = value
      if key == "perturb":
        self.diag_perturb = value


  # -----------------------------------------------------------------------------------
  def set_bounds(self,low,high):
    self.bound_lower = np.copy(low)
    self.bound_upper = np.copy(high)


  # -----------------------------------------------------------------------------------
  def build_distance_matrix(self):
    
    self_points = np.copy(self.points)
    for d in range(self.dimension):
      for i in range(self.npoints):
        self_points[d,i] = (self_points[d,i] - self.bound_lower[d]) / (self.bound_upper[d] - self.bound_lower[d])
    
    D = np.zeros((self.npoints+1,self.npoints+1))
    # fill row wise
    for d in range(self.dimension):
      pset = self_points[d,:]
      for i in range(self.npoints):
        distance = pset[i] - pset
        D[i,0:self.npoints] += distance*distance

    D = np.sqrt(D)
    #print(D)

    return D


  # -----------------------------------------------------------------------------------
  def build_gaussian_rbf_v1(self,npoints,c,A):
    B = np.zeros(A.shape)
    for i in range(A.shape[0]):
      for j in range(A.shape[1]):
        B[i,j] = np.exp( - c * A[i,j]*A[i,j] )
    return B


  # -----------------------------------------------------------------------------------
  def build_polyharmonic_rbf(self,npoints,c,A):
    k = 4.0 # must be even power
    B = np.zeros(A.shape)
    #for i in range(A.shape[0]):
    #  for j in range(A.shape[1]):
    #    r = A[i,j]
    #    B[i,j] = np.power(r,k-1.0) * np.log( np.power(r,r) )
    B = np.power(A,k-1.0) * np.log( np.power(A,A) )
    return B


  # -----------------------------------------------------------------------------------
  def build_tps_rbf(self,npoints,c,A):
    k = 2.0 # polyharmonic with k = 2
    #tau = 1.0
    B = np.zeros(A.shape)
    for i in range(A.shape[0]):
      for j in range(A.shape[1]):
        r = A[i,j]
        #        if r == 0.0:
        #          B[i,j] = 0.0
        #        else:
        #          ff = r * r * np.log(r/tau)
        #          B[i,j] = ff
        B[i,j] = r * np.log( np.power(r,r) )
    return B


  # -----------------------------------------------------------------------------------
  def build_mq_rbf(self,npoints,c,A):
    B = np.zeros(A.shape)
    for i in range(A.shape[0]):
      for j in range(A.shape[1]):
        dist = A[i,j]
        B[i,j] = np.sqrt( 1.0 + c * dist*dist )
    return B


  # -----------------------------------------------------------------------------------
  def build_rbf(self,npoints,c,A):
    if self.basis_type == 'gauss':
      B = self.build_gaussian_rbf_v1(npoints,c,A)
    elif self.basis_type == 'tps':
      B = self.build_tps_rbf(npoints,c,A)
    elif self.basis_type == 'polyh':
      B = self.build_polyharmonic_rbf(npoints,c,A)
    elif self.basis_type == 'mq':
      B = self.build_mq_rbf(npoints,c,A)
    else:
      raise RuntimeError('[RBF] Unknown RBF basis type. Found ' + self.basis_type)
    
    return B


  # -----------------------------------------------------------------------------------
  # points is tuple of length dimension
  def setup(self,npoints,points,values):
    compute_store_Ainv = False
    
    s = points.shape
    if self.dimension != s[0]:
      print('Dimension does not match. Expected',str(self.dimension),'found',str(s[0]))
      return
    for d in range(0,self.dimension):
      if npoints != len(points[d]):
        print('Length of array',d,'must',str(npoints),'found',len(points[d]))
        return

    self.npoints = npoints
    self.points = points
    #print('[RBF] npoints',npoints)

    # Build distance matrix
    t0 = time.time()
    A = self.build_distance_matrix()
    t1 = time.time()
    #print('[RBF][Build distance matrix] time',t1-t0)

    # Build RBF basis from distance matrix
    t0 = time.time()
    A = self.build_rbf(self.npoints,self.shape_parameter,A)
    t1 = time.time()
    #print('[RBF][Build basis matrix] time',t1-t0)
    
    # Impose constraints to enforce constant functions can be reproduced
    A[:, npoints] = 1.0
    A[npoints, :] = 1.0
    A[npoints, npoints] = 0.0
    
    # Optionally perturb diagonal
    if self.diag_perturb is not None:
      #print('[RBF] shifting diagonal by',('%1.2e' % self.diag_perturb))
      #np.fill_diagonal(A,self.diag_perturb)
      for i in range(self.npoints): # Do not perturb the variable associated with enforcing the polynomial constraint!
        A[i,i] += self.diag_perturb

    # swtich off constraint
    #A[:,npoints] = 0.0
    #A[npoints,:] = 0.0
    #A[npoints,npoints] = 1.0

    # Define the right hand side. The last equation is associated with the constant reproduciblilty constraint
    rhs = np.zeros(npoints+1)
    rhs[0:npoints] = values
    rhs[npoints]   = 0.0

    # Solve
    x = np.zeros(npoints+1)
  
    # LU
    if self.solver_type == 'lu':
      try:
        x = np.linalg.solve(A, rhs)
        
        if compute_store_Ainv:
          Ainv = np.zeros((self.npoints+1,self.npoints+1))
          for j in range(self.npoints+1):
            col = np.zeros(self.npoints+1)
            col[j] = 1.0
            Ainv[:,j] = np.linalg.solve(A, col)
      
      except:
        raise RuntimeError('[RBF] LU factorization failed')

    # SVD
    elif self.solver_type == 'svd':
      try:
        t0 = time.time()
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        t1 = time.time()
        #print('[RBF][SVD factorisation] time',t1-t0)
        self.U = u
        self.S = s
        self.Vh = vh
      

        x = np.zeros(npoints+1)
        for i in range(0,len(s)):
          fac = np.dot(u[:,i],rhs) / s[i]
          x = x + fac * vh[i,:]

        if compute_store_Ainv:
          t0 = time.time()
          for j in range(self.npoints+1):
            col = np.zeros(self.npoints+1)
            col[j] = 1.0
            for i in range(0,len(s)):
              fac = np.dot(u[:,i],col) / s[i]
              Ainv[:,j] = Ainv[:,j] + fac * vh[i,:]
          t1 = time.time()
          #print('[RBF][Assemble inv(A)] time',t1-t0)
    
    
      except:
          raise RuntimeError('[RBF] SVD factorization failed')

    else:
      raise RuntimeError('[RBF] Unknown solver type. Found ' + self.solver_type)

    # Store result
    self.coeff          = x[0:npoints]
    self.coeff_constant = x[npoints]

    #print('coeff',self.coeff)
    #print('sum(coeff)',np.sum(self.coeff))
    #print('coeff_const',self.coeff_constant)

    # Monitors
    #Binv = np.linalg.inv(A)
    #diff = Binv - Ainv
    #diff = np.absolute(diff)
    #print('[RBF] max(Ainv[numpy] - Ainv[mine])   ',('%1.6e' % diff.max()))
    if compute_store_Ainv:
      self.Ainv = Ainv

    # compute residuals
    ierr = 0

    rr = np.matmul(A,x) - rhs
    if np.linalg.norm(rr) > 1.0e-6:
      #print('[RBF] l2(A x - b) [my inverse]   ',('%1.6e' % np.linalg.norm(rr)))
      #print('rr',np.absolute(rr))
      ierr = 1
      pass


    return ierr


  # -----------------------------------------------------------------------------------
  def action_A_inverse(self, rhs):
    u, s, vh = self.U, self.S, self.Vh
    return vh.T @ (u.T @ rhs / s)
        


  # -----------------------------------------------------------------------------------
  def distance_matrix_general(self, ni, xi_points):
    self_xi_points = np.copy(xi_points)
    
    #for d in range(self.dimension):
    #  for i in range(ni):
    #    self_xi_points[d,i] = (self_xi_points[d,i] - self.bound_lower[d]) / (self.bound_upper[d] - self.bound_lower[d])
    #for d in range(self.dimension):
    #  self_xi_points[d,:] = (self_xi_points[d,:] - self.bound_lower[d]) / (self.bound_upper[d] - self.bound_lower[d])

    self_xj_points = np.copy(self.points)
    #for d in range(self.dimension):
    #  for i in range(self.npoints):
    #    self_xj_points[d,i] = (self_xj_points[d,i] - self.bound_lower[d]) / (self.bound_upper[d] - self.bound_lower[d])
    #for d in range(self.dimension):
    #  self_xj_points[d,:] = (self_xj_points[d,:] - self.bound_lower[d]) / (self.bound_upper[d] - self.bound_lower[d])

    for d in range(self.dimension):
      self_xi_points[d,:] = (self_xi_points[d,:] - self.bound_lower[d]) / (self.bound_upper[d] - self.bound_lower[d])
      self_xj_points[d,:] = (self_xj_points[d,:] - self.bound_lower[d]) / (self.bound_upper[d] - self.bound_lower[d])

    nj = self.npoints
    D = np.zeros((ni,nj))
    for d in range(self.dimension):
      ipset = self_xi_points[d,:]
      jpset = self_xj_points[d,:]
      for i in range(ni):
        distance = ipset[i] - jpset
        D[i,:] += distance * distance
    return np.sqrt(D)

#   for d in range(self.dimension):
#     self_xi_points[d,:] -= self.bound_lower[d]
#     self_xj_points[d,:] -= self.bound_lower[d]
#
#    nj = self.npoints
#    D = np.zeros((ni,nj))
#    for d in range(self.dimension):
#      da = np.zeros((ni,nj))
#      ipset = self_xi_points[d,:]
#      jpset = self_xj_points[d,:]
#      for i in range(ni):
#        distance = ipset[i] - jpset
#        da[i,:] = distance*distance
#      delta = self.bound_upper[d] - self.bound_lower[d]
#      fac = 1.0/(delta*delta)
#      D[:,:] += fac * da[:,:]
#    return np.sqrt(D)


  # -----------------------------------------------------------------------------------
  def evaluate(self,ni,xi_points):
    xi_points = np.array(xi_points,ndmin=2)
    A = self.distance_matrix_general(ni,xi_points)
    A = self.build_rbf(ni,self.shape_parameter,A)
    ivalue = np.matmul(A,self.coeff)
    ivalue += self.coeff_constant
    return np.asarray(ivalue)

  # D_ij = (D2)^{1/2}
  def d_grad_d_matrix_general(self, ni, xi_points, p):
    if p < 0 or p > self.dimension:
      raise RuntimeError('Derivative must be >= 0 and < ' + str(self.dimension))

    self_xi_points = np.copy(xi_points)
    self_xj_points = np.copy(self.points)
    
    for d in range(self.dimension):
      self_xi_points[d,:] = (self_xi_points[d,:] - self.bound_lower[d]) / (self.bound_upper[d] - self.bound_lower[d])
      self_xj_points[d,:] = (self_xj_points[d,:] - self.bound_lower[d]) / (self.bound_upper[d] - self.bound_lower[d])
    
    nj = self.npoints
    D2 = np.zeros((ni,nj))
    for d in range(self.dimension):
      ipset = self_xi_points[d,:]
      jpset = self_xj_points[d,:]
      for i in range(ni):
        distance = ipset[i] - jpset
        D2[i,:] += distance * distance
    
    gD2 = np.zeros((ni,nj))
    d = p
    factor_d = self.bound_upper[d] - self.bound_lower[d]
    ipset = self_xi_points[d,:]
    jpset = self_xj_points[d,:]
    for i in range(ni):
      sep = ipset[i] - jpset
      gD2[i,:] += 2.0 * sep / factor_d
    D = np.sqrt(D2)
    return D, 0.5 * gD2 / D


  def build_polyharmonic_rbf_deriv(self,npoints,c,A,gradA,p):
    k = 4.0 # must be even power
    B = np.zeros(A.shape)
    #for i in range(A.shape[0]):
    #  for j in range(A.shape[1]):
    #    r = A[i,j]
    #    B[i,j] = np.power(r,k-1.0) * np.log( np.power(r,r) )
    #B = np.power(A,k-1.0) * np.log( np.power(A,A) )

    for i in range(A.shape[0]):
      for j in range(A.shape[1]):
        d = A[i,j]
        grad_p_d = gradA[i,j]
        B[i,j] = (k - 1.0)*np.power(d, k - 1.0)*np.log(np.power(d, d))*grad_p_d/d + (np.log(d)*grad_p_d + grad_p_d)*np.power(d, k - 1.0);
    
    return B

  def build_rbf_derivative(self,npoints,c,A,grad_A,p):
    if self.basis_type == 'gauss':
      raise NotImplementedError('[RBF] basis type \"gauss\" does not support derivatives')
    elif self.basis_type == 'tps':
      raise NotImplementedError('[RBF] basis type \"tps\" does not support derivatives')
    elif self.basis_type == 'polyh':
      dPhi = self.build_polyharmonic_rbf_deriv(npoints,c,A,grad_A,p)
    elif self.basis_type == 'mq':
      raise NotImplementedError('[RBF] basis type \"mq\" does not support derivatives')
    else:
      raise RuntimeError('[RBF] Unknown RBF basis type. Found ' + self.basis_type)
    
    return dPhi

  def evaluate_derivative(self,ni,xi_points,p_i=-1):
    if p_i < 0 or p_i > self.dimension:
      raise RuntimeError('Derivative must be >= 0 and < ' + str(self.dimension))
    
    xi_points = np.array(xi_points,ndmin=2)
    A, grad_A = self.d_grad_d_matrix_general(ni, xi_points, p_i)
    dPHI = self.build_rbf_derivative(ni, self.shape_parameter, A, grad_A, p_i)
    ivalue = np.matmul(dPHI,self.coeff)
    #ivalue += self.coeff_constant
    return np.asarray(ivalue)
