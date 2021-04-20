#import yaml_anchor as yanchor
from yanchor import yaml_anchor

# Create an instance of YAMLAnchor, indicating where to find your template YAML file
template = yaml_anchor.YAMLAnchor("template_F.yaml")

# Call the method subs_emit() which will replace anchors with user provided values, here a = 1.0, b = 4.4
ierr = template.subs_emit( {'@scalar:parameter_a':1.0, '@scalar:parameter_b': 4.4} )
