# -*- coding: utf8 -*-

__version__ = "0.6.4"


import sys
import os
import time

import PyQt5
import pyqtgraph
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtWidgets



'''0) Redo spline functionality
1) g(r) -> coordination #'s
            - implement all options
2) option to pick lorch damping/lp filter
    lowpall_FIR.py (devel, Tyler), Fig. 3.1 in report
    * make sure input params are clear
3)Kaplow-style low-r correction (lok for Yu Shu code?)
                        |--> Tyler code
4) MonteCarlo white beam estimate optimization
    * develop convergence criteria
    * examples using Fe & SiO2 using
    iterative "code running"'''

resources_path = os.path.join(os.path.dirname(__file__), 'resources')
calibrants_path = os.path.join(resources_path, 'calibrants')
icons_path = os.path.join(resources_path, 'icons')
data_path = os.path.join(resources_path, 'data')
output_path = os.path.join(resources_path, 'output')
style_path = os.path.join(resources_path, 'style')
aEDXD = os.path.join(resources_path, 'aEDXD')

def main():
    
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    from axd.controllers.aEDXD_controller import aEDXDController
    controller = aEDXDController(app,1)

    Fe_test=  os.path.normpath( os.path.join(aEDXD,'P3.3-GPa-T1950-408psa-Fe-scan_ab5.cfg'))
    controller.config_controller.load_config_file(filename=Fe_test)

    app.exec_()
    del app
    