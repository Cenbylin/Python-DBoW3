# Python wrapper for [DBoW3](https://github.com/rmsalinas/DBow3)

Originally used to implement ORBSLAM2 with Python. 

This repo will consider supporting more general application scenarios.


## Prerequisites

* A compiler with C++11 support
* CMake >= 2.8.12
* Numpy
* OpenCV 4 (not opencv-python)


## Installation
```bash
pip install DBoW3Py
```

Just clone this repository and pip install. Note the `--recursive` option which is
needed for the pybind11 submodule:

```bash
git clone --recursive https://github.com/Cenbylin/Python-DBoW3.git
cd ./Python-DBoW3
pip install .
```

## Usage

```python
import DBoW3Py as dbow
```

more examples are comming 

## License

PyDBoW3 is provided under a BSD-style license that can be found in the LICENSE
file. By using, distributing, or contributing to this project, you agree to the
terms and conditions of this license.