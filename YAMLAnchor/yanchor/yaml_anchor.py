
import os, sys
import datetime as ts

try:
  from ruamel.yaml import YAML
except:
  raise ImportError('Require ruamel package (https://pypi.org/project/ruamel.yaml) to be installed!')


# ================================
def __walk_anchor_find(d, target):
  """
  Walk through the dictionary `d` recursively and look for the first instance of a value named `target`.
  If a matching value is found, return both the dictionary and the key associated with the value `target`.
  """
  
  res_k = None
  res_d = None
  for key, value in d.items():
    if value == target:
      return key,d

    if isinstance(value, dict):
      #print('found a value which is a dict()')
      res_k, res_d = __walk_anchor_find(value,target)
      if res_k is not None:
        return res_k, res_d
  
    elif isinstance(value,list):
      #print('found a value which is a list()')
      for i in value:
        #print('iterate',i)
        if isinstance(i, dict):
          res_k, res_d = __walk_anchor_find(i,target)
          if res_k is not None:
            return res_k, res_d
    else:
      pass

  return res_k, res_d

def anchor_find(d, target, summary = False):
  if summary:
    print('[yaml anchor find]')
    print('  * sarching for:',target)
  tk,td = __walk_anchor_find(d,target)
  target_dict = td
  target_key  = tk
  if summary:
    print('  * found dict:  ',td)
    print('  * use key:     ',tk)

  return target_key, target_dict


# ===============================
def __walk_anchor_scan(d, alist):
  """
  Walk through the dictionary `d` recursively and look for values which have the first characeter `@`.
  If a matching value is found, append the value into the list `alist`.
  """

  res_k = None
  res_d = None
  for key, value in d.items():
    try:
      if value[0] == '@':
        alist.append(value)
        return key,d
    except:
      pass

    if isinstance(value, dict):
      res_k, res_d = __walk_anchor_scan(value,alist)
  
    elif isinstance(value,list):
      for i in value:
        if isinstance(i, dict):
          res_k, res_d = __walk_anchor_scan(i,alist)
    else:
      pass
  
  return res_k, res_d


def anchor_scan(d):
  alist = list()
  k, d = __walk_anchor_scan(d,alist)
  return alist


# ==============================
def __walk_anchor_count(d, cnt):
  """
  Walk through the dictionary `d` recursively and look for values which have the first characeter `@`.
  Count these instances and store the results in `cnt`.
  """
  
  cnt_k = cnt
  res_d = None
  for key, value in d.items():
    try:
      if value[0] == '@':
        cnt_k += 1
    except:
      pass

    if isinstance(value, dict):
      res_d, cnt_k = __walk_anchor_count(value,cnt_k)
    elif isinstance(value,list):
      for i in value:
        if isinstance(i, dict):
          res_d, cnt_k = __walk_anchor_count(i,cnt_k)
    else:
      pass
  
  return res_d, cnt_k

def anchor_count(d):
  a = 0
  rd, a = __walk_anchor_count(d, a)
  return a


# ===============
class YAMLAnchor:
  
  
  def __init__(self,input_filename):

    self.input_filename = input_filename
    print('+ Opening file \"' + self.input_filename + '\"')
    with open(self.input_filename, "r") as fp:
      ystring = fp.read()

    # parse
    print('+ Parsing template yaml file \"' + self.input_filename + '\"')
    self.yaml = YAML()
    self.ydata = self.yaml.load(ystring)

    # count anchors (store)
    self.nanchor_init = anchor_count(self.ydata)
    print('+ Scanning input yaml data for anchors: found',str(self.nanchor_init))
    if self.nanchor_init == 0:
      print('[Error] Input file contans zero anchors')
      print('[Error] No yaml file emitted')
      return 1

    # scan for anchors (store)
    self.anchor_list = anchor_scan(self.ydata)
    print('+ anchor list:')
    for a in self.anchor_list:
      print('    ' + str(a))

    # check for duplicates
    self.duplicates_found = False
    c_dict = { i:self.anchor_list.count(i) for i in self.anchor_list }
    for key, value in c_dict.items():
      if value > 1:
        self.duplicates_found = True
    if self.duplicates_found:
      print('+ Input file defines duplicate anchors')
    else:
        print('+ Input file does not define duplicate anchors')


  # =================================
  def subs(self, ydata, anchor_vals):
    
    y_dict = dict(ydata)
    nanchor_init = anchor_count(y_dict)
    
    # Scan inputs and check if they are defined in the file
    # Report any found but unused (undefined in file) anchors
    # We do not throw the error here - just a warning
    # If anchors in file are missing at the end, an error will be thrown
    # Ignoring anchors not found (c.f. throwing an error) will be more convienent for the user
    missing = 0
    for key, value in anchor_vals.items():
      if key not in self.anchor_list:
        print('[Warning] Input anchor \"' + key + '\" not found in file')
        missing += 1
    if missing != 0:
      print('[Warning] Found',str(missing),'input anchor names not in file - these will be ignored')
      print('[Warning] Review anchor list above')

    for key, value in anchor_vals.items():
      k,d = anchor_find(y_dict,key)
      try: # Gaurd this to allow scenario that input anchors are not present in file
        d[k] = value
      except:
        pass

    replaced = 0
    nanchor = anchor_count(y_dict)
    replaced = nanchor_init - nanchor

    return y_dict, replaced


  # =====================================================================
  def subs_emit(self, a_dict, output = None, allow_partial_subs = False, overwrite = False):

    # check for duplicates (dis-allowed in this function)
    if self.duplicates_found:
      print('[Error] Use subs_with_duplicates_emit()')
      return self.nanchor_init

    y_dict, replaced = self.subs(self.ydata, a_dict)

    if allow_partial_subs == False:
      # count anchors, throw error if count != 0
      nanchor = anchor_count(y_dict)
      print('+ Scanning processed yaml data for anchors: found',str(nanchor))
      if nanchor != 0:
        print('[Error] ' + str(nanchor) + ' anchors found in file \"' + self.input_filename + '\" have NOT been replaced')
        found = anchor_scan(y_dict)
        print('+ anchor list:')
        for a in found:
          print('    ' + str(a))
        print('[Error] Review output from anchor list above')
        print('[Error] No yaml file emitted')
        return nanchor

    # dump output yaml file
    filename = output
    if output is None: # create file name
      filename = self.input_filename.replace('.yaml','-anchor-subs.yaml')
    if overwrite == False and os.path.isfile(filename):
        raise RuntimeError('[Error] File name \"' +  filename + '\" already exists')
    print('+ Emitting file \"' + filename + '\"')
    with open(filename, "w") as fp:
      self.yaml.dump(y_dict, fp)
    with open(filename, "a") as fp:
      fp.write('# Generated from ' + self.input_filename + ' on ' + str(ts.datetime.now()))
    
    return 0


  # ============================================
  def subs_duplicates(self, ydata, anchor_vals):
    sum = 0
    ydata0 = dict(ydata)
    y_dict, replaced = self.subs(ydata0, anchor_vals)
    sum += replaced
    while replaced != 0:
      ydata0 = y_dict
      y_dict, replaced = self.subs(ydata0, anchor_vals)
      sum += replaced
    
    return y_dict, sum


  # ================================================================================
  def subs_duplicates_emit(self, a_dict, output = None, allow_partial_subs = False, overwrite = False):
  
    y_dict, replaced = self.subs_duplicates(self.ydata, a_dict)
    
    if allow_partial_subs == False:
      # count anchors, throw error if count != 0
      nanchor = anchor_count(y_dict)
      print('+ Scanning processed yaml data for anchors: found',str(nanchor))
      if nanchor != 0:
        print('[Error] ' + str(nanchor) + ' anchors found in file \"' + self.input_filename + '\" have NOT been replaced')
        found = anchor_scan(y_dict)
        print('+ anchor list:')
        for a in found:
          print('    ' + str(a))
        print('[Error] Review output from anchor list above')
        print('[Error] No yaml file emitted')
        return nanchor

    # dump output yaml file
    filename = output
    if output is None: # create file name
      filename = self.input_filename.replace('.yaml','-anchor-subs.yaml')
    if overwrite == False and os.path.isfile(filename):
      raise RuntimeError('[Error] File name \"' +  filename + '\" already exists')
    print('+ Emitting file \"' + filename + '\"')
    with open(filename, "w") as fp:
      self.yaml.dump(y_dict, fp)
    with open(filename, "a") as fp:
      fp.write('# Generated from ' + self.input_filename + ' on ' + str(ts.datetime.now()))

    return 0
