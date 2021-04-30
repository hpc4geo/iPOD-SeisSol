
import numpy as np

from romutils import *

def test1():
  v = np.zeros(3)
  v[0] = 1.1
  v[1] = 2.2
  v[2] = 3.3
  r = ndarray_to_list_of_dicts(v)
  print(r)


def test2():
  v = np.zeros((1, 3))
  v[0][0] = 1.1
  v[0][1] = 2.2
  v[0][2] = 3.3
  r = ndarray_to_list_of_dicts(v)
  print(r)


def test3():
  v = np.zeros((2, 3))
  v[0][0] = 1.1
  v[0][1] = 2.2
  v[0][2] = 3.3
  v[1][0] = 3.1
  v[1][1] = 4.2
  v[1][2] = 5.3
  r = ndarray_to_list_of_dicts(v, param_name=['q', 'f'])
  print(r)


def test3a():
  v = np.zeros((2, 3))
  v[0][0] = 1.1
  v[0][1] = 2.2
  v[0][2] = 3.3
  v[1][0] = 3.1
  v[1][1] = 4.2
  v[1][2] = 5.3
  r = ndarray_to_dict(v, param_name=['q', 'f'])
  print(r)


def test4():
  v = np.zeros((2, 3))
  v[0][0] = 1.1
  v[0][1] = 2.2
  v[0][2] = 3.3
  v[1][0] = 3.1
  v[1][1] = 4.2
  v[1][2] = 5.3
  print('v', v)
  r1 = ndarray_to_list_of_dicts(v)
  print(r1)

  r2 = list_of_dicts_to_ndarray(r1)
  print('r2', r2)


def test5():
  v = ( {'a': 1}, {'a': 2}, {'a': 3} )
  pvals, pnames = list_of_dicts_to_ndarray(v)
  print('pvals', pvals)
  print('pnames', pnames)

  v2 = list_of_dicts_to_dict(v)
  print('v2', v2)


def test6():
  v = ( {'a': 1, 'b': 2.2, 'c': 3}, {'a': 1, 'b': 2.4, 'c': 3}, {'a': 1, 'b': 2.8, 'c': 3} )
  
  # Remove all keys named "a" if want to fine grained control
  #for d in v:
  #  d.pop("a",None)
  #v = list_of_dicts_purge(v, ['a'])
  
  info = list_of_dicts_info(v)
  print('info\n',info)
  print('info["b"]["min"] =',info["b"]["min"])
  
  vf = list_of_dicts_filter(v, {"b":(2.4, 3.0)})
  print('v filtered\n',vf)
  
  pvals, pnames = list_of_dicts_to_ndarray(v)
  print('pvals (all)\n', pvals)
  print('pnames (all)\n', pnames)
  print('parameter : values')
  for i in range(len(pnames)):
    print('"' + pnames[i] + '"' + ':',pvals[i][:])

  # Extract index of the parameters we want to isolate (here "b" and "c")
  index_b = pnames.index("b")
  index_c = pnames.index("c")
  
  # Define a dict() which we use to select "b" and "c" from the full parameter set
  select = dict()
  select[ pnames[index_b] ] = index_b
  select[ pnames[index_c] ] = index_c

  # split
  vv = np.zeros((len(select), pvals.shape[1]))
  vv[0][:] = pvals[select["b"]][:]
  vv[1][:] = pvals[select["c"]][:]
  print('pvals\n', vv)
  print('pnames\n', select)


if __name__ == "__main__":
  #test1()
  #test2()
  test3()
  #test3a()
  #test4()
  test5()
  #test6()

