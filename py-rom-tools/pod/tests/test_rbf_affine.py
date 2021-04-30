
import os
import numpy as np
import scipy as sip
from scipy import spatial
import time

from pod import RBFAffine


def test2_pod_dim_1():
  dimension = 1
  
  #platespeeds = np.array([1., 1.157895, 1.315789, 1.473684, 3.0,3.5, 3.6,3.7,4.,4.2,4.5  ])
  platespeeds = np.array([1.,2.421053, 2.578947, 3.052632, 1.473684, 1.157895, 2.894737, 4., 2.105263, 2.263158, 1.947368, 1.631579, 1.789474, 3.368421, 3.210526, 1.315789, 2.736842])
  #platespeeds = np.array([1.,2.421053, 2.578947, 3.052632, 1.473684,4.0])

  npoints = len(platespeeds)
  print(npoints)
  
  values = np.zeros(npoints)
  for i in range(npoints):
    values[i] = np.exp(platespeeds[i])*np.sin(3.14*platespeeds[i]/0.7)
  
  low = np.zeros(dimension)
  high = np.zeros(dimension)
  low[0] = 1.0
  high[0] = 4.0
#low[0] = 0.0
#high[0] = 1.0

  param_1 = platespeeds
  params = np.zeros((dimension,npoints))
  params[0,:] = param_1
  
  rbf = RBFAffine(dimension)
  #rbf.shape_parameter = (3.0/3.75)*(3.0/3.75)
  #rbf.shape_parameter = (1.0/3.75)*(1.0/3.75)
  rbf.shape_parameter = 4.5**2
  rbf.set_bounds(low,high)
  rbf.setup( npoints, params, values )
  
  
  nsample = 120
  iparams_1 = np.linspace(1.0, 4.0, nsample)
  iparams = np.zeros((dimension,nsample))
  iparams[0,:] = iparams_1

  D = rbf.distance_matrix_general(nsample,iparams)
  Dp = np.zeros((nsample,npoints+1))
  Dp[0:nsample,0:npoints] = D
  A = rbf.build_rbf(npoints,rbf.shape_parameter,Dp)
  A[:,npoints] = 1
  
  #print('params',params)
  #print(rbf.Ainv.shape)
  #print(A.shape)
  Nt = np.matmul(A,rbf.Ainv)
  
  U = np.matmul(Nt[:,0:npoints],values)
  file = open('Uinterp.gp','w')
  for i in range(nsample):
    line = ('%1.4e' % iparams_1[i]) + ' ' + ('%1.4e' % U[i] ) + '\n'
    file.write(line)
  file.close()
  
  
  for k in range(npoints):
    file = open('b-'+str(k)+'.gp','w')
    for i in range(nsample):
      line = ('%1.4e' % iparams[0,i]) + ' ' + ('%1.4e' % Nt[i,k] ) + '\n'
      file.write(line)
    file.close()

  idata = rbf.evaluate(nsample,iparams)
  file = open('interp.gp','w')
  for i in range(nsample):
    line = ('%1.4e' % iparams[0,i]) + ' ' + ('%1.4e' % idata[i] ) + '\n'
    file.write(line)
  file.close()

def test2_high_ar_dim_2(nx):

  def func_2(x,y):
    f = np.cos(np.pi*(x-4.4)/9.6)*np.exp(y*1.3)*np.cos(np.pi*(y-0.3))
    return f
  

  dimension = 2
  
  npoints = nx * nx
  params = np.zeros((dimension,npoints))
  cnt = 0
  for i in range(nx):
    for j in range(nx):
      params[0,cnt] = 0.0 + i * 20.0/float(nx-1)
      params[1,cnt] = 0.0 + j * 1.0/float(nx-1)
      cnt += 1

  values = np.zeros(npoints)
  for i in range(npoints):
    values[i] = func_2(params[0][i],params[1][i])
  
  low = np.zeros(dimension)
  high = np.zeros(dimension)
  low[0]  = 0.0
  low[1]  = 0.0
  high[0] = 20.0
  high[1] = 1.0
  
  rbf = RBFAffine(dimension, rbftype = 'polyh', solvertype = 'svd', perturb = 1.0e-6)
  rbf.set_bounds(low,high)
  rbf.shape_parameter = 10.1
  rbf.setup(npoints,params,values)
  
  nsample = 40*40
  iparams = np.zeros((dimension,nsample))
  iparams[0,:] = 20.0 * np.random.rand(nsample)
  iparams[1,:] = 1.0 * np.random.rand(nsample)
  
  idata = rbf.evaluate(nsample,iparams)
  
  maxerr = 0.0
  for i in range(nsample):
    a = iparams[0,i]
    b = iparams[1,i]
    fexact = func_2(a,b)
    err = np.absolute(fexact - idata[i])
    if err > maxerr:
      maxerr = err

  print('max|fexact - f_rbf| (basis type: ' + rbf.basis_type + ')' ,('%1.4e' % maxerr))

  file = open('interp_dim2.gp','w')
  for i in range(nsample):
    line = ('%1.4e' % iparams[0,i]) + ' ' + ('%1.4e' % iparams[1,i]) + ' ' + ('%1.4e' % idata[i] ) + '\n'
    file.write(line)
  file.close()


def sampler(nx):
  import halton as sequence
  
  
  def func_2(x,y):
    f = np.cos(np.pi*(x-4.4)/9.6)*np.exp(y*1.3)*np.cos(np.pi*(y-0.3))
    return f
  
  def test_samples(rbf,points):
    ns = len(points)
    residuals = np.zeros(ns)

    iparams = np.zeros((dimension,ns))
    for i in range(ns):
      iparams[0,i] = points[i][0]
      iparams[1,:] = points[i][1]
  
    idata = rbf.evaluate(ns,iparams)
    for i in range(ns):
      a = iparams[0,i]
      b = iparams[1,i]
      fexact = func_2(a,b)
      residuals[i] = np.absolute(fexact - idata[i])

    return residuals


  dimension = 2
  shift = 0.0
  kernel = 'polyh'
  NW = 5
  NT = 100
  hidx = 0

  low = np.zeros(dimension)
  high = np.zeros(dimension)
  low[0]  = 0.0
  low[1]  = 0.0
  high[0] = 20.0
  high[1] = 1.0


  # intial param space
  points = list()
  for i in range(nx):
    for j in range(nx):
      x = 0.0 + i * 20.0/float(nx-1)
      y = 0.0 + j * 1.0/float(nx-1)
      points.append(np.array([x,y]))
  npoints = len(points)

  hidx = npoints

  # ===============================
  # insert from list into array for rbf
  params = np.zeros((dimension,npoints))
  for i in range(npoints):
    params[0,i] = points[i][0]
    params[1,i] = points[i][1]

  values = np.zeros(npoints)
  for i in range(npoints):
    values[i] = func_2(params[0][i],params[1][i])

  rbf = RBFAffine(dimension, rbftype = kernel, solvertype = 'svd', perturb = shift)
  rbf.set_bounds(low,high)
  rbf.setup(npoints,params,values)


  # Define NT halton points
  # Test them
  new_points = list()
  for i in range(NT):
    s = sequence.halton(i+hidx,2)
    s[0] = s[0] * 20.0
    s[1] = s[1] * 1.0
    new_points.append( s )
  hidx += NT

  residuals = test_samples(rbf,new_points)
  print('max residual',np.ndarray.max(residuals))
  print('max at index',np.argmax(residuals))
  worst = np.argmax(residuals)
  #print(residuals)
  # Get the worst 10
  indices = np.argsort(residuals)[-NW:]
  #print(indices)
  w_point = list()
  for idx in indices:
    w_point.append(np.array([ new_points[idx][0], new_points[idx][1] ]))

  points = points + w_point
  npoints = len(points)


  if 1 == 1:

    # insert from list into array for rbf
    params = np.zeros((dimension,npoints))
    for i in range(npoints):
      params[0,i] = points[i][0]
      params[1,i] = points[i][1]

    values = np.zeros(npoints)
    for i in range(npoints):
      values[i] = func_2(params[0][i],params[1][i])

    rbf = RBFAffine(dimension, rbftype = kernel, solvertype = 'svd', perturb = shift)
    rbf.set_bounds(low,high)
    rbf.setup(npoints,params,values)

    # Define 30 halton points
    # Test them
    new_points = list()
    for i in range(NT):
      s = sequence.halton(i+hidx,2)
      s[0] = s[0] * 20.0
      s[1] = s[1] * 1.0
      new_points.append( s )
    hidx += NT

    residuals = test_samples(rbf,new_points)
    print('max residual',np.ndarray.max(residuals))
    print('max at index',np.argmax(residuals))
    worst = np.argmax(residuals)
    #print(residuals)
    # Get the worst 10
    indices = np.argsort(residuals)[-NW:]
    #print(indices)
    w_point = list()
    for idx in indices:
      w_point.append(np.array([ new_points[idx][0], new_points[idx][1] ]))

    points = points + w_point
    npoints = len(points)
  


  if 2 == 2:
    
    # insert from list into array for rbf
    params = np.zeros((dimension,npoints))
    for i in range(npoints):
      params[0,i] = points[i][0]
      params[1,i] = points[i][1]
      #print('i',i,'x',points[i][0],'y',points[i][1])
      #print(i,points[i][0],points[i][1])

    values = np.zeros(npoints)
    for i in range(npoints):
      values[i] = func_2(params[0][i],params[1][i])



    rbf = RBFAffine(dimension, rbftype = kernel, solvertype = 'svd', perturb = shift)
    rbf.set_bounds(low,high)
    rbf.setup(npoints,params,values)
      
    # Define 30 halton points
    # Test them
    new_points = list()
    for i in range(NT):
      s = sequence.halton(i+hidx,2)
      s[0] = s[0] * 20.0
      s[1] = s[1] * 1.0
      new_points.append( s )
    hidx += NT

    residuals = test_samples(rbf,new_points)
    print('max residual',np.ndarray.max(residuals))
    print('max at index',np.argmax(residuals))
    worst = np.argmax(residuals)
    #print(residuals)
    # Get the worst 10
    indices = np.argsort(residuals)[-NW:]
    #print(indices)
    w_point = list()
    for idx in indices:
      w_point.append(np.array([ new_points[idx][0], new_points[idx][1] ]))
    
    points = points + w_point
    npoints = len(points)




  if 3 == 3:
  
    # insert from list into array for rbf
    params = np.zeros((dimension,npoints))
    for i in range(npoints):
      params[0,i] = points[i][0]
      params[1,i] = points[i][1]

    values = np.zeros(npoints)
    for i in range(npoints):
      values[i] = func_2(params[0][i],params[1][i])


    rbf = RBFAffine(dimension, rbftype = kernel, solvertype = 'svd', perturb = shift)
    rbf.set_bounds(low,high)
    rbf.setup(npoints,params,values)
    
    # Define 30 halton points
    # Test them
    new_points = list()
    for i in range(NT):
      s = sequence.halton(i+hidx,2)
      s[0] = s[0] * 20.0
      s[1] = s[1] * 1.0
      new_points.append( s )
    hidx += NT

    residuals = test_samples(rbf,new_points)
    print('max residual',np.ndarray.max(residuals))
    print('max at index',np.argmax(residuals))
    worst = np.argmax(residuals)
    #print(residuals)
    # Get the worst 10
    indices = np.argsort(residuals)[-NW:]
    #print(indices)
    w_point = list()
    for idx in indices:
      w_point.append(np.array([ new_points[idx][0], new_points[idx][1] ]))

    points = points + w_point
    npoints = len(points)





  # insert from list into array for rbf
  np.random.seed(0)
  params = np.zeros((dimension,npoints))
  for i in range(npoints):
    params[0,i] = points[i][0]
    params[1,i] = points[i][1]
    
  values = np.zeros(npoints)
  for i in range(npoints):
    values[i] = func_2(params[0][i],params[1][i])
    
  rbf = RBFAffine(dimension, rbftype = kernel, solvertype = 'svd', perturb = shift)
  rbf.set_bounds(low,high)
  rbf.setup(npoints,params,values)


  file = open('input_dim2.gp','w')
  for i in range(npoints):
    line = ('%1.4e' % params[0,i]) + ' ' + ('%1.4e' % params[1,i]) + ' ' + ('%1.4e' % values[i] ) + '\n'
    file.write(line)
  file.close()


  # Test space
  nsample = 100*100
  iparams = np.zeros((dimension,nsample))
  iparams[0,:] = 18.0 * np.random.rand(nsample)
  iparams[1,:] = 0.9 * np.random.rand(nsample)
  
  idata = rbf.evaluate(nsample,iparams)
  
  maxerr = 0.0
  for i in range(nsample):
    a = iparams[0,i]
    b = iparams[1,i]
    fexact = func_2(a,b)
    err = np.absolute(fexact - idata[i])
    if err > maxerr:
      maxerr = err

  print('max|fexact - f_rbf| (basis type: ' + rbf.basis_type + ')' ,('%1.4e' % maxerr))
  
  file = open('interp_dim2.gp','w')
  for i in range(nsample):
    line = ('%1.4e' % iparams[0,i]) + ' ' + ('%1.4e' % iparams[1,i]) + ' ' + ('%1.4e' % idata[i] ) + '\n'
    file.write(line)
  file.close()


def test4_grad(nx):
  
  def func_2(x,y):
    f = x*x + 2.0*y*y
    return f
  def dx_func_2(x,y):
    f = 2.0*x
    return f
  def dy_func_2(x,y):
    f = 4.0*y
    return f
  
  
  dimension = 2
  
  npoints = nx * nx
  params = np.zeros((dimension,npoints))
  cnt = 0
  for i in range(nx):
    for j in range(nx):
      params[0,cnt] = 0.0 + i * 1.0/float(nx-1)
      params[1,cnt] = 0.0 + j * 1.0/float(nx-1)
      cnt += 1

  values = np.zeros(npoints)
  for i in range(npoints):
    values[i] = func_2(params[0][i],params[1][i])

  rbf = RBFAffine(dimension, rbftype = 'polyh', solvertype = 'svd', perturb = 0.0e-10)
  rbf.setup(npoints,params,values)
  
  nsample = 40*40
  iparams = np.zeros((dimension,nsample))
  iparams[0,:] = 1.0 * np.random.rand(nsample)
  iparams[1,:] = 1.0 * np.random.rand(nsample)
  
  idata = rbf.evaluate(nsample,iparams)
  
  maxerr = 0.0
  for i in range(nsample):
    a = iparams[0,i]
    b = iparams[1,i]
    fexact = func_2(a,b)
    err = np.absolute(fexact - idata[i])
    if err > maxerr:
      maxerr = err
  print('max|fexact - f_rbf| (basis type: ' + rbf.basis_type + ')' ,('%1.4e' % maxerr))
  
  file = open('interp4_dim2.gp','w')
  for i in range(nsample):
    line = ('%1.4e' % iparams[0,i]) + ' ' + ('%1.4e' % iparams[1,i]) + ' ' + ('%1.4e' % idata[i] ) + ' ' + ('%1.4e' % func_2(iparams[0,i],iparams[1,i]) ) + '\n'
    file.write(line)
  file.close()


  idata = rbf.evaluate_derivative(nsample,iparams,p_i=0)

  maxerr = 0.0
  for i in range(nsample):
    a = iparams[0,i]
    b = iparams[1,i]
    dfexact = dx_func_2(a,b)
    err = np.absolute(dfexact - idata[i])
    if err > maxerr:
      maxerr = err
  print('max|f,x - f_rbf,x| (basis type: ' + rbf.basis_type + ')' ,('%1.4e' % maxerr))


  file = open('interp4_x_dim2.gp','w')
  for i in range(nsample):
    line = ('%1.4e' % iparams[0,i]) + ' ' + ('%1.4e' % iparams[1,i]) + ' ' + ('%1.4e' % idata[i] ) + ' ' + ('%1.4e' % dx_func_2(iparams[0,i],iparams[1,i]) ) + '\n'
    file.write(line)
  file.close()

  idata = rbf.evaluate_derivative(nsample,iparams,p_i=1)
  
  maxerr = 0.0
  for i in range(nsample):
    a = iparams[0,i]
    b = iparams[1,i]
    dfexact = dy_func_2(a,b)
    err = np.absolute(dfexact - idata[i])
    if err > maxerr:
      maxerr = err
  print('max|f,y - f_rbf,y| (basis type: ' + rbf.basis_type + ')' ,('%1.4e' % maxerr))


  file = open('interp4_y_dim2.gp','w')
  for i in range(nsample):
    line = ('%1.4e' % iparams[0,i]) + ' ' + ('%1.4e' % iparams[1,i]) + ' ' + ('%1.4e' % idata[i] ) + ' ' + ('%1.4e' % dy_func_2(iparams[0,i],iparams[1,i]) ) + '\n'
    file.write(line)
  file.close()


def test5_grad(nx):
  
  def func_2(x,y):
    f = x*x + 2.0*y*y
    return f
  def dx_func_2(x,y):
    f = 2.0*x
    return f
  def dy_func_2(x,y):
    f = 4.0*y
    return f
  
  low = np.zeros(2)
  high = np.zeros(2)
  low[0]  = 0.0
  low[1]  = 0.0
  high[0] = 1.0
  high[1] = 2.0

  
  dimension = 2
  
  npoints = nx * nx
  params = np.zeros((dimension,npoints))
  cnt = 0
  for i in range(nx):
    for j in range(nx):
      params[0,cnt] = 0.0 + i * 1.0/float(nx-1)
      params[1,cnt] = 0.0 + j * 1.0/float(nx-1)
      cnt += 1

  values = np.zeros(npoints)
  for i in range(npoints):
    values[i] = func_2(params[0][i],params[1][i])

  rbf = RBFAffine(dimension, rbftype = 'polyh', solvertype = 'svd', perturb = 1.0e-10)
  rbf.set_bounds(low,high)
  rbf.setup(npoints,params,values)
  
  nsample = 40*40
  iparams = np.zeros((dimension,nsample))
  iparams[0,:] = 1.0 * np.random.rand(nsample)
  iparams[1,:] = 1.0 * np.random.rand(nsample)
  
  idata = rbf.evaluate(nsample,iparams)
  
  maxerr = 0.0
  for i in range(nsample):
    a = iparams[0,i]
    b = iparams[1,i]
    fexact = func_2(a,b)
    err = np.absolute(fexact - idata[i])
    if err > maxerr:
      maxerr = err
  print('max|fexact - f_rbf| (basis type: ' + rbf.basis_type + ')' ,('%1.4e' % maxerr))
  
  
  D = rbf.distance_matrix_general(nsample,iparams)
  Dp = np.zeros((nsample,npoints+1))
  Dp[0:nsample,0:npoints] = D
  A = rbf.build_rbf(0,rbf.shape_parameter,Dp)
  A[:,npoints] = 1
  
  Nt = np.matmul(A,rbf.Ainv)
  rhs = np.zeros(npoints+1)
  rhs[0:npoints] = values
  rhs[npoints] = 0
  idata = np.matmul(Nt,rhs)

  maxerr = 0.0
  for i in range(nsample):
    a = iparams[0,i]
    b = iparams[1,i]
    fexact = func_2(a,b)
    err = np.absolute(fexact - idata[i])
    if err > maxerr:
      maxerr = err
  print('max|fexact - f_rbf| (basis type: ' + rbf.basis_type + ')' ,('%1.4e' % maxerr))


  # ==========================
  idata = rbf.evaluate_derivative(nsample,iparams,p_i=0)
  maxerr = 0.0
  for i in range(nsample):
    a = iparams[0,i]
    b = iparams[1,i]
    dfexact = dx_func_2(a,b)
    err = np.absolute(dfexact - idata[i])
    if err > maxerr:
      maxerr = err
  print('max|f,x - f_rbf,x| (basis type: ' + rbf.basis_type + ')' ,('%1.4e' % maxerr))


  file = open('interp5_x_dim2.gp','w')
  for i in range(nsample):
    line = ('%1.4e' % iparams[0,i]) + ' ' + ('%1.4e' % iparams[1,i]) + ' ' + ('%1.4e' % idata[i] ) + ' ' + ('%1.4e' % dx_func_2(iparams[0,i],iparams[1,i]) ) + '\n'
    file.write(line)
  file.close()


  idata = rbf.evaluate_derivative(nsample,iparams,p_i=1)
  maxerr = 0.0
  for i in range(nsample):
    a = iparams[0,i]
    b = iparams[1,i]
    dfexact = dy_func_2(a,b)
    err = np.absolute(dfexact - idata[i])
    if err > maxerr:
      maxerr = err
  print('max|f,y - f_rbf,y| (basis type: ' + rbf.basis_type + ')' ,('%1.4e' % maxerr))


  file = open('interp5_y_dim2.gp','w')
  for i in range(nsample):
    line = ('%1.4e' % iparams[0,i]) + ' ' + ('%1.4e' % iparams[1,i]) + ' ' + ('%1.4e' % idata[i] ) + ' ' + ('%1.4e' % dy_func_2(iparams[0,i],iparams[1,i]) ) + '\n'
    file.write(line)
  file.close()

  #idata = rbf.evaluate_derivative(nsample,iparams,p_i=0)

#  D = rbf.distance_matrix_general(nsample,iparams)
#  Dp = np.zeros((nsample,npoints+1))
#  Dp[0:nsample,0:npoints] = D
#  A = rbf.build_rbf(0,rbf.shape_parameter,Dp)
#  A[:,npoints] = 1
#
#  Nt = np.matmul(A,rbf.Ainv)
#  rhs = np.zeros(npoints+1)
#  rhs[0:npoints] = values
#  rhs[npoints] = 0
#  idata = np.matmul(Nt,rhs)


  D, gradD = rbf.d_grad_d_matrix_general(nsample,iparams,0)
  dPhi = rbf.build_rbf_derivative(0,rbf.shape_parameter,D,gradD,0)
  dPhi_1 = np.zeros((nsample,npoints+1))
  dPhi_1[0:nsample,0:npoints] = dPhi
  dNt = np.matmul(dPhi_1,rbf.Ainv)

  rhs = np.zeros(npoints+1)
  rhs[0:npoints] = values
  rhs[npoints] = 0

  idata = np.matmul(dNt,rhs)

  maxerr = 0.0
  for i in range(nsample):
    a = iparams[0,i]
    b = iparams[1,i]
    dfexact = dx_func_2(a,b)
    err = np.absolute(dfexact - idata[i])
    if err > maxerr:
      maxerr = err
  print('max|f,x - f_rbf,x| (basis type: ' + rbf.basis_type + ')' ,('%1.4e' % maxerr))


def test4_f1d(nx):
  
  def func_2(x):
    f = np.exp(3.3 * x)
    return f
  
  dimension = 1
  npoints = nx
  params = np.zeros((dimension,npoints))
  cnt = 0
  for i in range(nx):
    params[0,cnt] = 0.0 + i * 1.0/float(nx-1)
    cnt += 1

  values = np.zeros(npoints)
  for i in range(npoints):
    values[i] = func_2(params[0][i])

  rbf = RBFAffine(dimension, rbftype = 'polyh', solvertype = 'svd', perturb = 0.0e-10)
  rbf.setup(npoints,params,values)
  
  nsample = 100
  iparams = np.zeros((dimension,nsample))
  np.random.seed(0)
  iparams[0,:] = 1.0 * np.random.rand(nsample)
  idata = rbf.evaluate(nsample,iparams)
  
  maxerr = 0.0
  for i in range(nsample):
    a = iparams[0,i]
    fexact = func_2(a)
    err = np.absolute(fexact - idata[i])
    if err > maxerr:
      maxerr = err
  print('max|fexact - f_rbf| (basis type: ' + rbf.basis_type + ')' ,('%1.4e' % maxerr))
  
  file = open('interp4_dim1.gp','w')
  for i in range(nsample):
    line = ('%1.4e' % iparams[0,i]) + ' ' + ('%1.4e' % idata[i] ) + ' ' + ('%1.4e' % func_2(iparams[0,i]) ) + '\n'
    file.write(line)
  file.close()
  

# Executes only if run as a script
if __name__ == "__main__":
  np.random.seed(0)
  #test2_pod_dim_1()
  #test2_high_ar_dim_2(10)
  #sampler(4)

  #test4_grad(40)
  #test5_grad(10)
  test4_f1d(40)


