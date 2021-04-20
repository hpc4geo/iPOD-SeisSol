
import sys
#import yaml_anchor as yanchor
from yancher import yaml_anchor


# =======================================================
def parse_emit_fault(filename, anchor_vals = None):
  template = yaml_anchor.YAMLAnchor(filename)
  
  # test 1
  # python yaml_anchor.py fault.yaml
  #ierr = template.subs_emit( {'w':3.3} )
  
  # test 2
  # python yaml_anchor.py fault.yaml
  #ierr = template.subs_emit( {'w':3.3}, allow_partial_subs=True )
  
  # test 3
  # python yaml_anchor.py fault.yaml
  ierr = template.subs_emit( {'@scalar:const:d_c':3.3, '@tensor_comp:inside_patch:s_zz':[0,1],  '@tensor_comp:outside_patch:s_yz':[2,3]}, overwrite = True )
  
  # test 4
  # python yaml_anchor.py fault-with-dups.yaml
  #ierr = template.subs_duplicates_emit( {'@scalar:const': 333.333}, allow_partial_subs=True)
  
  if ierr != 0:
    print('anchors remaining',ierr)

if __name__ == '__main__':
  nargs = len(sys.argv[1:])
  if nargs == 1:
    args = sys.argv[1:]
  else:
    raise RuntimeError('Expected a single command line argument defining a SeisSol YAML file')
  
  ierr = parse_emit_fault(args[0])


