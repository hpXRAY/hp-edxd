from ast import Global
from itertools import count
import numpy as np
from scipy import interpolate
import multiprocessing



import time
CAL_OFFSET = 0.12
CAL_SLOPE = .8


x_size = 4096


X = np.asarray(range(x_size))
Y = np.sin(X/21)


now = time.time()
f = interpolate.interp1d(X, Y, assume_sorted=True)
after = time.time()
diff = after - now
print('1dinterp ' + str(diff))

now = time.time()



R = np.linspace(X[0],X[-1],num=int(len(X)*.79))
R_vals = f(R)

#_vals = R_vals #/ CAL_SLOPE

after = time.time()
diff = after - now
print('remap ' +str(diff)) 

from pyqtgraph.Qt import QtGui, QtCore

import pyqtgraph as pg

#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])
#mw = QtGui.QMainWindow()
#mw.resize(800,800)

win = pg.GraphicsWindow(title="Basic plotting examples")
win.resize(1000,600)
win.setWindowTitle('pyqtgraph example: Plotting')


# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

p4 = win.addPlot(title="Parametric, grid enabled")
x = X
y = Y
p4.plot(x, y)
p4.showGrid(x=True, y=True)


p4.plot(R, R_vals, pen=(200,200,200), symbolBrush=(255,0,0), symbolSize = 3, symbolPen='w')



## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec()
