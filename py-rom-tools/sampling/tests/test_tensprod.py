
import numpy as np

from sampling import *

def test1():
  c = ACell(1,[0.0],[1.0])
  centroid = tp_compute_subdivisions_centroid(c)
  print('[1d] centroids of split cell',centroid)
  
  c = ACell(2,2*[0.0],2*[1.0])
  centroid = tp_compute_subdivisions_centroid(c)
  print('[2d] centroids of split cell(1)',centroid)
  
  c = ACell(2,[0.0,0.5],[1.0,0.5])
  centroid = tp_compute_subdivisions_centroid(c)
  print('[2d] centroids of split cell(2)',centroid)
  
  children = tp_cell_split(c)
  for cc in children:
    print(cc)
    for d in range(cc.dimension):
      print('  d',d,'box',cc.t[d][0],cc.t[d][0]+cc.t[d][1])
    xc = tp_cell_centroid(cc)
    print('  centroid',xc)


def test2():
  ndim = 10
  c = ACell(ndim, ndim * [0.0], ndim * [1.0])
  children = tp_cell_split(c)
  print('len ',len(children))


def test3():
  import pickle
  
  def view_tree(t,filename):
    fp = open(filename,"w")
    for s in range(len(tree)):
      for scell in tree[s]:
        xc = tp_cell_centroid(scell)
        fp.write(str(xc[0]) + ' ' + str(xc[1]) + '\n')
    fp.close()
  
  # init tree
  celllist = []
  ndim = 2
  c = ACell(ndim, ndim * [0.0], ndim * [1.0])
  celllist.append(c)
  tree = tp_build_tree(celllist)
  view_tree(tree,'t0.gp')
  print('[init tree]')
  print('======')
  print('len(tree)',len(tree))
  for s in range(len(tree)):
    for scell in tree[s]:
      xc = tp_cell_centroid(scell)
      print('  -->','[level ',str(s),'] xc',xc[0],xc[1])

  # split root cell
  children = tp_cell_split(c)
  celllist += children
  tree = tp_build_tree(celllist)
  view_tree(tree,'t1.gp')
  print('[ref 1 tree]')
  print('======')
  print('len(tree)',len(tree))
  for s in range(len(tree)):
    for scell in tree[s]:
      xc = tp_cell_centroid(scell)
      print('  -->','[level ',str(s),'] xc',xc[0],xc[1])

  # split all leaves
  leaves = tree[-1]
  for l in leaves:
    children = tp_cell_split(l)
    celllist += children
  tree = tp_build_tree(celllist)

  # split leaves (conditional)
  leaves = tree[-1]
  for l in leaves:
    xc = tp_cell_centroid(l)
    if xc[0] < 0.5 and xc[1] < 0.25:
      children = tp_cell_split(l)
      celllist += children
  tree = tp_build_tree(celllist)

  # split leaves (conditional)
  leaves = tree[-1]
  for l in leaves:
    xc = tp_cell_centroid(l)
    if xc[0] < 0.5 and xc[1] < 0.25:
      children = tp_cell_split(l)
      celllist += children
  tree = tp_build_tree(celllist)
  view_tree(tree,'t2.gp')

  print('======')
  print('len(tree)',len(tree))
  for s in range(len(tree)):
    for scell in tree[s]:
      xc = tp_cell_centroid(scell)
      print('  -->','[level ',str(s),'] xc',xc[0],xc[1])


  with open("celllist.pkl", "wb") as fp:
    pickle.dump(celllist, fp)

  celllist = []
  with open("celllist.pkl", "rb") as fp:
    celllist = pickle.load(fp)
  print('celllist[from disk]')
  for c in celllist:
    xc = tp_cell_centroid(c)
    print('  -->','[level ',str(c.level),'] xc',xc[0],xc[1])


# Check defining a nsub = 1 works (e.g. does not split)
def test4():
  c = ACell(2,2*[0.0],2*[1.0])
  centroid = tp_compute_subdivisions_centroid(c,[2,1])
  print('[2d] centroids of split cell(1)',centroid)
  
  children = tp_cell_split(c,[2,1])
  for cc in children:
    print(cc)
    for d in range(cc.dimension):
      print('  d',d,'box',cc.t[d][0],cc.t[d][0]+cc.t[d][1])
    xc = tp_cell_centroid(cc)
    print('  centroid',xc)


if __name__ == '__main__':
  #test1()
  #test2()
  test3()
  #test4()
