import pyqtgraph as pg
import numpy as np
from pyqtgraph.exporters.SVGExporter import SVGExporter
from pyqtgraph.exporters.ImageExporter import ImageExporter
from PyQt6 import QtGui

if __name__ == '__main__':
    app = QtGui.QApplication([])
    win = pg.GraphicsWindow("PNG export")
    win.resize(1000, 600)
    p1 = win.addPlot()
    p1.plot(x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], y=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    scatter = pg.ScatterPlotItem(pen=(200, 200, 200),
                                 symbolBrush=(255, 0, 0),
                                 symbolPen='w',
                                 size=5,
                                 )
    p1.addItem(scatter)
    scatter.setData(np.arange(10), np.arange(10))

    QtGui.QApplication.processEvents()
    QtGui.QApplication.processEvents()

    '''
    pngexp = ImageExporter(p1)
    pngexp.params["width"] = int(pngexp.params["width"])
    pngexp.export("176.png")

    pngexp = ImageExporter(p1)
    pngexp.params["width"] = int(pngexp.params["width"] * 3)
    pngexp.export("176_scaled.png")
    '''

    svgexp = SVGExporter(p1)
    svgexp.export("176.svg")