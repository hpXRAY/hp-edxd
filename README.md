# hp-edxd
 
Python-based suite of programs for data collection, viewing and analysis of energy-dispersive x-ray diffraction. The software has specialized features for experiments on materials under high pressure. The software is currently used at beamline 16-BM-B, Advanced Photon Source to acquire, view, and process the x-ray diffraction data.

The programs work with Anaconda Python 3.7 <br>
required python packages: <br>
PyQt6 5.9.2 (Some compatiblity issues with newer versions)<br>
pyqtgraph 0.11.0 (Not compatible with 0.13)<br> 
pyepics 3.4.0<br>
burnman 0.9.0<br>
PyCifRW 4.4.1 <br>

> export PYEPICS_LIBCA=/usr/local/epics/base-7.0.4/lib/linux-x86_64/libca.so
To find out which CA library will be used by pyepics, use:
>>> import epics
>>> epics.ca.find_libca()

### Executables

Executable versions for Windows, Mac OS (all 64bit) can be downloaded from:

https://github.com/hrubiak/hp-edxd/releases

The executable versions are self-contained and do not need a python installation.

The documentation is hosted at https://hp-edxd.readthedocs.io/.