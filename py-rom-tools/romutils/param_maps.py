
import numpy as np

# Methods to manipulate lists of dicts and 2D numpy.ndarray's defining mulitvariate parameter values


def ndarray_to_list_of_dicts(params, param_name=None):
  """
  Convert a 2D array of parameter values into a list of dict()'s.
  
  params (input)
    - A numpy.ndarray of shape M x N where M = #parameters and N = #parameters_values.
  param_name (input, optional / keyword argument)
    - A list of strings defining the M textual names for each parameter.
      Parameter names must be unique.
      
  result (output)
    - A list of dict()'s in which each key, value pair is given by key = param_name[i], value = params[i][j].
      If param_name was not provided, default parameter names "p[1,...,M]" will be used.
  """
  
  sizes = params.shape
  if len(sizes) == 1:
    raise RuntimeError('Expected `params` to be a 2D array of shape [number_parameters][parameter_values]')
  
  if param_name is None:
    param_name = list()
    for i in range(sizes[0]):
      param_name.append("p" + str(i+1))
  else:
    # Check sizes
    if len(param_name) != sizes[0]:
      raise RuntimeError('Expected `param_name` contain ' + str(sizes[0]) + ' entries. Found ' + str(len(param_name)) )
    # Check for duplicates
    if len(param_name) != len(set(param_name)):
      raise RuntimeError('`param_name` must define unique parameter names')

  result = list()
  for j in range(sizes[1]):
    item = dict()
    for i in range(sizes[0]):
      item[param_name[i]] = params[i][j]
    result.append(item)
  return result


def list_of_dicts_to_ndarray(params):
  """
  Convert a list of N dict's into a 2D array of parameter values.
  Performs the inverse operation of ndarray_to_list_of_dicts().
  
  params (input)
    - A list of dict()'s defining parameter name / value pairs.
    
  result (output)
    - A numpy.ndarray of shape M x N, where M = #parameters and N = #parameters_values.
  keys (output)
    - A list of M unique parameter names (e.g. strings).
  """
  
  # Check input is list/tuple/iteratable of dicts
  for j in range(len(params)):
    d = params[j]
    if not isinstance(d, dict):
      raise RuntimeError('Value of `params` at index ' + str(j) + ' must be a dict()')
  
  d0 = params[0]
  keys = list(d0.keys())
  M = len(d0)

  # Check each dict contains the same number of items
  for j in range(1, len(params)):
    if len(params[j]) != M:
      raise RuntimeError('Dictionary at index ' + str(j) + ' must contain ' + str(M) + ' entries. Found ' + str(len(params[j])))

  # Check values in dict(0) are defined in every other dict()
  for j in range(1, len(params)):
    keys_j = list(params[j].keys())
    for k in keys:
      if k not in keys_j:
        raise RuntimeError('Key \"' + str(k) + '\" not found in dictionary at index ' + str(j))

  # Note - we don't need this test as a dict cannot be defined to have duplicate keys
  # Check keys are actually unique
  #if len(keys) != len(set(keys)):
  #  raise RuntimeError('The keys of each dictionary must define unique parameter names. Found ' + keys)

  # Allocate space and fill
  N = len(params)
  result = np.zeros((M, N))
  for i in range(M):
    for j in range(N):
      result[i][j] = params[j][keys[i]]

  return result, keys


def ndarray_to_dict(params, param_name=None):
  """
  Convert a 2D array of parameter values into a dict()
    
  params (input)
    - A numpy.ndarray of shape M x N where M = #parameters and N = #parameters_values.
  param_name (input, optional / keyword argument)
    - A list of strings defining the M textual names for each parameter.
      Parameter names must be unique.
    
  result (output)
    - A dict() in which each key, value pair is given by key = param_name[i], value = params[i][:].
      If param_name was not provided, default parameter names "p[1,...,M]" will be used.
  """
  
  sizes = params.shape
  if len(sizes) == 1:
    raise RuntimeError('Expected `params` to be a 2D array of shape [number_parameters][parameter_values]')
  
  if param_name is None:
    param_name = list()
    for i in range(sizes[0]):
      param_name.append("p" + str(i+1))
  else:
    # Check sizes
    if len(param_name) != sizes[0]:
      raise RuntimeError('Expected `param_name` contain ' + str(sizes[0]) + ' entries. Found ' + str(len(param_name)) )
    # Check for duplicates
    if len(param_name) != len(set(param_name)):
      raise RuntimeError('`param_name` must define unique parameter names')

  result = dict()
  for i in range(sizes[0]):
    result[param_name[i]] = params[i][:]
  return result


def list_of_dicts_to_dict(params):
  param_vals, param_names = list_of_dicts_to_ndarray(params)
  d = ndarray_to_dict(param_vals, param_name=param_names)
  return d

def list_of_dicts_purge(result, values_to_remove):
  """
  Remove all keys from `result` which are listed in `values_to_remove`.
  No error is thrown if any key in values_to_remove[] is not found in any dict().
  
  result (input)
    - List of dict()'s
  values_to_remove (input)
    - List of keys you wish to remove from `result`
    
  result (output)
    - Modified `result`
  """
  
  for d in result:
    for p in values_to_remove:
      d.pop(p,None)
  return result


def list_of_dicts_info(result):
  """
  Reports bounds and variations of parameters.
  
  result (intput)
    - List of dict()'s defining parameter (key) / parmeter values (value) pairs.
    
  info (output)
    - A dict of dict()'s defining global information for each parameter.
      Parameter info is accessed via the same keys defined in `result`.
      Global information provided includes:
        min, max, range, avg.
      Global information is accessed via the keys:
        "min", "max", "range", "avg"
        
      e.g.
      list_of_dicts = [ {"a": 1.2}, {"a": 3.2}, {"a": 33.2} ]
      
      info = list_of_dicts_info(list_of_dicts)
      
      info["a"]["min"]   -> returns 1.2
      info["a"]["range"] -> returns 32.0
      
  """

  v, k = list_of_dicts_to_ndarray(result)
  info = dict()
  for i in range(len(k)):
    min_i = np.min(v[i][:])
    max_i = np.max(v[i][:])
    range_i = np.abs(max_i - min_i)
    avg_i = np.average(v[i][:])
    info[k[i]] = { "min": min_i, "max": max_i, "range": range_i, "avg": avg_i }

  return info


def list_of_dicts_filter(ld, range):
  """
  Filter entries from ld which are outside the bounds specified by range.
  
  range (input)
    - A dict() specifying parameter (key) and its allowed [min, max] values (value)
    
  result (output)
    - A list containing entries from `ld` which are contained within the bounds specified by `range`.
  """
  
  result = list()

  # Get keys and check that at least one key in range is found in ld
  keys = list(ld[0].keys())
  key_filter = list(range.keys())
  matches = False
  for kf in key_filter:
    if kf in keys:
      matches = True
  if matches == False:
    msg = 'The range filter did not contain any parameters (keys) defined in `ld`. This likely indicates an error.\n'
    msg += 'Please inspect input:\n' + '  `ld` = ' + str(ld) + '\n' + '  `range` = ' + str(range)
    raise RuntimeError(msg)

  for d in ld:
    keep_item = True
    # Test parameter against all specified bounds.
    # If a value violates any bound, mark dict() not to be kept (i.e. keep_item = False)
    for kf in key_filter:
      if kf in keys:
        value = d[kf]
        bounds = range[kf]
        if value < bounds[0]:
          keep_item = False
          break
        if value > bounds[1]:
          keep_item = False
          break
    if keep_item:
      result.append(d)

  return result
