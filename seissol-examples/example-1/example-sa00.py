
import os
import numpy as np
import pandas as pd
import xmltodict
import h5py

import pod as podtools

#
# Experiment with package xml
# Synposis: Akward to use
#
def test_1():
  import xml.etree.ElementTree as ET

  df = pd.read_csv('mun-GME.csv')
  print(df.columns)

  time_vals = df['time'].values
  print(time_vals)

  dir = df['directory'].values[0]
  xmf_file = df['output-file'].values[0]

  xmf_file = os.path.join(dir, xmf_file)
  
  mytree = ET.parse(xmf_file)
  myroot = mytree.getroot()

  attr_list = []
  for t in time_vals:
    attribute_name = 'SA00.' + str(int(t)) + 's'
    print('  scan for Attribute', attribute_name)
    attr_list.append(attribute_name)

  for x in myroot.findall('.Domain/Grid/Grid/Attribute'):
    print(x.tag, x.text)
    for y in x:
      print(y.tag)



#
# Use xmltodict for simpler parsing of XDMF file
#
def test_2():
  
  df = pd.read_csv('mun-GME.csv')
  print(df.columns)
  
  time_vals = df['time'].values
  print(time_vals)
  
  dir = df['directory'].values[0]
  xmf_file = df['output-file'].values[0]
  
  xmf_file = os.path.join(dir, xmf_file)
  
  with open(xmf_file) as fp:
    doc = xmltodict.parse(fp.read())
  
  
  # Define attribute names we want to extract
  attr_list = []
  for t in time_vals:
    attribute_name = 'SA00.' + str(int(t)) + 's'
    print('  will scan for Attribute with name', attribute_name)
    attr_list.append(attribute_name)

  h5_list = []


  # Determine the H5 file names containing the attributes we seek
  data_root = doc['Xdmf']['Domain']['Grid']['Grid']
  x_attr = data_root['Attribute']
  for item in x_attr:
    
    #@Name @Center DataItem
    #for child in item:
    #  print(child)

    attr_name = item['@Name']
    if attr_name in attr_list:
      print('Loading', attr_name)

      hyperslab_di = item['DataItem']
      #for y in hyperslab_di:
      #  print(y)
      fields = hyperslab_di['DataItem']
      for f in fields:
        if f['@Format'] == 'HDF':
          print('  found h5 file', f['#text'])
          h5_list.append(f['#text'])


  print(attr_list)
  print(h5_list)

  h5_file = h5_list[0].split(':')[0]
  h5_file = os.path.join(dir, h5_file)
  print('H5 file to load', h5_file)

  controls = dict()
  controls['time'] = list()
  for t in time_vals:
    controls['time'].append(t)
  print(controls)


  snapshots = list()
  h5f = h5py.File(h5_file, 'r')
  for name in h5_list:
    h5_field = name.split(':')[1]
    print('  h5_field', h5_field)
    
    _snap = h5f[(h5_field)]
    #print(type(_snap), _snap.shape, _snap.dtype)
    snap = np.array(_snap)
  
    #snap = h5f.get(h5_field).value # `data` is now an ndarray.
    print('   ',type(snap), snap.shape, snap.dtype)
  
    snapshots.append(snap)

  h5f.close()


  # Build the POD reduced order model
  pod = podtools.PODMultivariate(remove_mean=False)
  pod.database_append(controls, snapshots)
  pod.setup_basis()
  pod.setup_interpolant(rbf_type='polyh', bounds_auto=True)

  print('Singular values:', pod.singular_values)
  e = pod.get_ric()
  print('RIC:', e)

  # LOOCV measures
  measure = podtools.rbf_loocv(pod, norm_type="linf")
  measure = np.absolute(measure)
  
  ordering = np.argsort(measure)
  print('m[smallest][Linf] =',('%1.4e' % measure[ordering[0]]))
  print('m[largest ][Linf] =',('%1.4e' % measure[ordering[-1]]))


  measure = podtools.rbf_loocv(pod, norm_type="rms")
  measure = np.absolute(measure)

  ordering = np.argsort(measure)
  print('m[smallest][rms] =',('%1.4e' % measure[ordering[0]]))
  print('m[largest ][rms] =',('%1.4e' % measure[ordering[-1]]))


  # Evaluate the POD model at an arbitrary instant in time
  x0 = pod.evaluate([220.0])
  #print(x0)

  # Push POD data into a new H5 file
  h5f = h5py.File("pod_surface_cell.h5", "w")
  grp = h5f.create_group("mesh0/")
  #dset = h5f.create_dataset("mydataset", (100,), dtype='i')
  dset = grp.create_dataset('pod', data=x0)
  h5f.close()


if __name__ == '__main__':
  
  #test_1()

  test_2()
