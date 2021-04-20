import numpy as np
#import yaml_anchor as yanchor
from yanchor import yaml_anchor

# Create an instance of YAMLAnchor, indicating where to find your template YAML file
template = yaml_anchor.YAMLAnchor("template_F.yaml")

# Define the input parameter values
a = np.asarray( [1.0, 5.5, 6.0] )
b = np.asarray( [4.4, 99.0, -23.3] )
N = len(a)

# Loop over a, b pairs
for i in range(N):
  # Define an output file name using the param names (a,b) and the index i
  output_yaml = 'F-' + 'a_'+ str(i) + '-b_' + str(i) + '.yaml'
	
  # Substitute and emit a YAML files for each a[i], b[i] pair
  ierr = template.subs_emit( {'@scalar:parameter_a': float(a[i]), '@scalar:parameter_b': float(b[i])}, output = output_yaml, overwrite = True )
