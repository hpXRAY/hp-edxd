# -*- coding: utf8 -*-

# DISCLAIMER
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from PyQt6 import QtCore, QtGui, QtWidgets

from  PyQt6.QtWidgets import QMainWindow, QApplication, QInputDialog, QWidget, QLabel
from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight

from functools import partial

import numpy as np
import pyqtgraph as pg


class sxdmWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        #self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.initUI()
        self.resize(600,600)
        
        
    def initUI(self):
        self.setWindowTitle('SXDM')
        #self.main_widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QVBoxLayout()
        self.make_img_plot()
        self.layout.addWidget(self.win)
        self.bottom_layout = QtWidgets.QHBoxLayout()
        self.btn1 = QtWidgets.QPushButton('Load data')
        self.btn2 = QtWidgets.QPushButton('btn 2')
        self.btn3 = QtWidgets.QPushButton('btn 3')
        self.number = DoubleSpinBoxAlignRight()
        self.number.setDecimals(0)
        self.number.setMaximum(3999)
        self.bottom_layout.addWidget(self.btn1)
        self.bottom_layout.addWidget(self.btn2)
        self.bottom_layout.addWidget(self.btn3)
        self.bottom_layout.addWidget(self.number)
        self.bottom_layout.addSpacerItem(HorizontalSpacerItem())
        self.layout.addLayout(self.bottom_layout)
        self.setLayout(self.layout)
        #self.setCentralWidget(self.main_widget)

    def set_view_range(self, left, top, right, bottom):
        self.view.setRange(QtCore.QRectF(left, top, right, bottom))

    def make_img_plot(self):
        ## Create window with GraphicsView widget
        self.win = pg.GraphicsLayoutWidget(self)
        self.win.setWindowTitle('pyqtgraph example: ImageItem')
        self.view = self.win.addViewBox()
        ## lock the aspect ratio so pixels are always square
        self.view.setAspectLocked(True)
        self.view.setMouseMode(pg.ViewBox.RectMode)
        ## Create image item
        self.img = pg.ImageItem(border='w')
        self.img.setScaledMode()
        
        self.view.addItem(self.img)

   
        
    def updateView(self, data):
        ## Display the data
        self.img.setImage(data)
 

    def raise_widget(self):
        self.show()
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowState.WindowMinimized | QtCore.Qt.WindowState.WindowActive)
        self.activateWindow()


    



