import glob




import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np


# Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder='row-major')

pg.mkQApp()
win = pg.GraphicsLayoutWidget()
win.setWindowTitle('pyqtgraph example: Image Analysis')

# A plot area (ViewBox + axes) for displaying the image
p1 = win.addPlot()

# Item for displaying image data
img = pg.ImageItem()
p1.addItem(img)

# Custom ROI for selecting an image region
roi = pg.ROI([0, 0], [2, 2])
roi.addScaleHandle([0.5, 1], [0.5, 0.5])
roi.addScaleHandle([0, 0.5], [0.5, 0.5])
#p1.addItem(roi)
roi.setZValue(10)  # make sure ROI is drawn above image

# Isocurve drawing
iso = pg.IsocurveItem(level=0.8, pen='g')
iso.setParentItem(img)
iso.setZValue(5)

# Contrast/color control
hist = pg.HistogramLUTItem()
hist.setImageItem(img)
win.addItem(hist)

# Draggable line for setting isocurve level
isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
hist.vb.addItem(isoLine)
hist.vb.setMouseEnabled(y=False) # makes user interaction a little easier
isoLine.setValue(0.8)
isoLine.setZValue(1000) # bring iso line above contrast controls


win.resize(1000, 800)
win.show()


# Generate image data
directory_name = "/Users/hrubiak/Library/CloudStorage/Box-Box/0 File pass/e249953-Hrubiak/Data analysis/cell4/Pressure/S1/P2"
files = glob.glob(glob.escape(directory_name) + "/*.dat")

# prefix the string with r (to produce a raw string):
array2d = np.loadtxt(files[0],)
mn = np.amin(array2d)
mask = array2d > mn

include = len(files)
for i in range(include-1):
    fname = files[i+1]
    new_frame = np.loadtxt(fname)
    array2d += new_frame
    mn = np.amin(new_frame)
    mask = mask * (new_frame > mn)
unmask = np.invert( mask)

array2d [unmask] = 0
array2d = array2d / include
data = np.transpose( array2d)

img.setImage(data)

H_range=(3,5)

hist.setLevels(*H_range)
hist.setHistogramRange(*H_range)


# zoom to fit imageo
p1.autoRange()  


# Callbacks for handling user interaction


h = np.histogram(data, bins=10, range=H_range)
x = h[1][:-1]
y = h[0]
pg.plot(x,y, title="histogram")
for i in range (len(x)):
    print( str(round(x[i],2)) +'\t'+str(y[i]))

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec()
