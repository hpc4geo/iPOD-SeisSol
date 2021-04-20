### What is YAMLAnchor?

YAMLAnchor is simple python class which will parse a YAML file and replace special anchors (values) with user provided values.



### Why?

YAML is a format which is used to express data in a structured and hierarchical manner.

As such, YAML is a good choice to use for describing input files associated with computational science tools.

Suppose you have a forward model `F(a,b)` which depends on two input parameters `a` and` b`. Assume that the values of `a` and `b` are provided to `F()` via a YAML file, e.g. 

```
# F.yaml
myForwardModel_F: 
  - a: 33.3
  - b: 66.6
```

Furthermore suppose the execution of the forward model is given by the following command:

 `./foward_model_F.app -input F.yaml`. 

Parameter studies are frequently conducted in which one wants to execute `F()` with a various input 
values for `a` and `b`. Let suppose we wish to conduct 3 experiments using `a = (1.0, 5.5, 6.0)` 
and `b = (4.4, 99.0, -23.3)`.



One could simply use `regex()` to parse `F.yaml` and perform the appropriate substitution for `a` and `b`. 
This approach has the downside that one apriori cannot safely verify (after calling `regex()`) that 
the resulting file (post substitution) is in fact a valid YAML file. 
One would only discover this when they execute the forward model and wait for the YAML parser within F() 
to produce a run-time error - that is simply annoying and completely avoidable if we use YAMLAnchor. 

YAMLAnchor relies on the python package `ruamel` which is a dedicated YAML parsers. 
By using a YAML parser, we ensured that both the input and output YAML files are valid YAML files.



### How?

Modify `F.yaml` and replace values which you want to vary in your parameter study with an anchor. 
Anchor's are strings (single quoted - backtick) and start with the symbol `@`.  
We will call this a template file. Below is an example

```
# template_F.yaml (modified from F.yaml)
myForwardModel_F: 
  - a: '@scalar:parameter_a'   # 33.3 (reference / default value)
  - b: '@scalar:parameter_b'   # 66.6 (reference / default value)
```

* Anchors must start with `@` 
* What follows the `@` is arbitrary, but as a recommendation the strings for each parameter should be given meaningful names (easy to identify). I included the term scalar here to indicate that the parameter being described is a scalar, as opposed to a vector or tensor.

In a python script, do the following (see `example-1/demo-1.py`)

```
import yaml_anchor as yanchor

# Create an instance of YAMLAnchor, indicating where to find your template YAML file
template = yanchor.YAMLAnchor("template_F.yaml")

# Call the method subs_emit() which will replace anchors with user provided values, here a = 1.0, b = 4.4
ierr = template.subs_emit( {'@scalar:parameter_a':1.0, '@scalar:parameter_b': 4.4} )
```

Calling `subs_emit()` will produce by default a file named `template_F-anchor-subs.yaml`. 
This can be over-ridden via the optional argument `output`.

The argument to `subs_emit()` is a dictionary. 
The keys in the dictionary should correspond to anchor names, the values in the dictionary should 
correspond to the values you wish to replace your anchor with. Dictionary keys which do not 
match an anchor in your input YAML file will be ignored.

As a safety measure, `subs_emit()` will not over-write an output file if it already exists. 
Use the optional argument `overwrite = True` if this default annoys you.

A better version of the above code is (see `example-1/demo-2.py`)

```
import numpy as np
import yaml_anchor as yanchor

# Create an instance of YAMLAnchor, indicating where to find your template YAML file
template = yanchor.YAMLAnchor("template_F.yaml")

# Define the input parameter values
a = np.asarray( [1.0, 5.5, 6.0] )
b = np.asarray( [4.4, 99.0, -23.3] )
N = len(a)

# Loop over a, b pairs
for i in range(N):
  # Define an output file name using the param names (a,b) and the index i
  output_yaml = 'F-' + 'a_'+ str(i) + '-b_' + str(i) + '.yaml'

  # Substitute and emit a YAML files for each a[i], b[i] pair
  ierr = template.subs_emit( {'@scalar:parameter_a': float(a[i]), '@scalar:parameter_b': float(b[i])}, output = output_yaml, , overwrite = True )
```



### Notes

* YAMLAnchor will by default assume that the generated output is valid if all anchors found are replaced by the user. This can be over-ridden via the optional argument, `allow_partial_subs = True` provided to `subs_emit()`.
* The method `subs_emit()` cannot be used if you have duplicate anchors (anchors with identical names) defined in a single YAML input file. Duplicate anchors are detected and an error will be thrown if you try to use `subs_emit()`.  If you defines duplicate anchors, use the method `subs_duplicates_emit()` instead of `subs_emit()`.
* `subs_emit()` and `subs_duplicates_emit()` will over write (clobber) existing files if you tell it to. Use the optional argument `overwrite = True` if you want this behaviour.