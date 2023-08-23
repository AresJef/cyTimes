## Easy management of python datetime & pandas time Series.

Created to be used in a project, this package is published to github 
for ease of management and installation across different modules.

### Installation
Install from `github`

``` bash
pip install git+https://github.com/AresJef/cyTimes.git
```

### Compatibility
Only support for python 3.10 and above.

### Acknowledgements
cyTimes is based on several open-source repositories.
- [numpy](https://github.com/numpy/numpy)
- [pandas](https://github.com/pandas-dev/pandas)

cyTimes makes modification of the following open-source repositories:
- [dateutil](https://github.com/dateutil/dateutil)

This package created a Cythonized version of dateutil.parser (cyparser) and
dateutil.relativedelta (cytimedelta). As a result, these two modules in
this package have sacrificed flexibility and readability in exchange for
enhancements in performance. All credits go to the original authors and
contributors of dateutil.
