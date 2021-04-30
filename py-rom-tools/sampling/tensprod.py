
import numpy as np

#
# Tensor product
#   X = (x0,x1), Y = (y0,y1), Z = (z0,z1)
# product = X \cross Y \cross Z
#
# Concatenate entries dimension-wise
#
# => r = ([x0],[x1])
#
# => r = r(i) * Y(j) = ([x0,y0],[x0,y1],[x1,y0],[x1,y1])
#
# => r = r(i) * Z(j) = ([x0,y0,z0],[x0,y0,z1],
#                       [x0,y1,z0],[x0,y1,z1],
#                       [x1,y0,z0],[x1,y0,z1],
#                       [x1,y1,z0],[x1,y1,z1])


def tp_compute_subdivisions_centroid(cell,nsub=None):
  if nsub is None:
    nsub = cell.dimension * [2]
  else:
    if cell.dimension != len(nsub):
      raise RuntimeError('dimension does not match len(nsub)')

  n0 = nsub[0]
  #h0 = cell.t[0][1] - cell.t[0][0]
  h0 = cell.t[0][1]
  hs = h0 / float(n0)
  xc = list()
  for i in range(n0):
    xc.append( [cell.t[0][0] + 0.5 * hs + i * hs] )
  #print('x0',xc)

  x1 = list()
  for d in range(1,cell.dimension):
    n1 = nsub[d]
    #hs = (cell.t[d][1] - cell.t[d][0]) / float(n1)
    hs = cell.t[d][1] / float(n1)
    x1 = list()
    for i in range(n1):
      x1.append(cell.t[d][0] + 0.5 * hs + i * hs)

    xnew = list()
    for i in range(len(xc)):
      for j in range(n1):
        xnew.append( xc[i] + [x1[j]] )

    xc = list(xnew)

  return xc


def tp_cell_split(cell,nsub=None):

  #print('tp_cell_split')
  if nsub is None:
    nsub = cell.dimension * [2]
  else:
    if cell.dimension != len(nsub):
      raise RuntimeError('dimension does not match len(nsub)')
  
  xc = tp_compute_subdivisions_centroid(cell,nsub)
  
  n = 1
  for d in range(cell.dimension):
    n = n * nsub[d]

  if n != len(xc):
    raise RuntimeError('dimension mismatch')

  # create, set in list and init level
  x0 = cell.dimension * [0]
  h0 = cell.dimension * [0]

  for d in range(cell.dimension):
    x0[d] = cell.t[d][0]
    #h0[d] = cell.t[d][1] - cell.t[d][0]
    h0[d] = cell.t[d][1]
    h0[d] = h0[d] / float( nsub[d] )

  children = list()
  for k in range(n):
    sub = ACell(cell.dimension,x0,h0)
    sub.level = cell.level + 1
    children.append( sub )

  for k in range(n):
    for d in range(cell.dimension):
      h_k = children[k].t[d][1]
      start_k = xc[k][d] - 0.5 * h_k

      r = tuple([start_k,h_k])
      children[k].t[d] = r

  cell.active = False

  return children


def tp_cell_centroid(cell):
  xc = list()
  for d in range(cell.dimension):
    h_k = cell.t[d][1]
    start_k = cell.t[d][0]
    xc.append( start_k + 0.5 * h_k )
  return tuple(xc)


def tp_build_tree(cells):
  maxlevels = 0
  for c in cells:
    if c.level > maxlevels:
      maxlevels = c.level
  maxlevels += 1
  
  tree = list()
  for m in range(maxlevels):
    tree.append( [] )
  
  for c in cells:
    tree[ c.level ].append(c)

  return tree


class ACell:
  def __init__(self,dimension,x0,h):
    # verify input
    if dimension != len(x0):
      raise RuntimeError('dimension does not match len(x0)')
    if dimension != len(h):
      raise RuntimeError('dimension does not match len(h)')
    
    self.dimension = dimension
    self.level = 0
    self.active = True
    self.t = list()
    for i in range(len(x0)):
      start = float(x0[i])
      end = float(x0[i]) + float(h[i])
      #r = tuple([start,end])
      r = tuple([start,h[i]])
      self.t.append(r)
