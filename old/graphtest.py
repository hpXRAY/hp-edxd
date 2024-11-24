import pyqtgraph as pg
import numpy as np 
data = np.random.normal(size=1000)
pg.plot(data, title="Simplest example")
if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec()
        