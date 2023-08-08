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

# Principal author: R. Hrubiak (hrubiak@anl.gov)
# Copyright (C) 2018-2019 ANL, Lemont, USA

from functools import partial
from PyQt5.QtCore import QObject, pyqtSignal, Qt
from PyQt5 import QtWidgets, QtCore
import copy
import numpy as np
import pyqtgraph as pg
from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight
from hpm.widgets.PltWidget import PltWidget
from hpm.widgets.MaskWidget import MaskWidget
from hpm.widgets.plot_widgets import ImgWidget2

class CustomImageWidget(pg.GraphicsLayoutWidget):
    plotMouseMoveSignal = pyqtSignal(float)  
    plotMouseCursorSignal = pyqtSignal(list)
    def __init__(self):
        super().__init__()
        
        self.p1 : pg.PlotWidget
        #self.setWindowTitle('Multiple spectra view')

        self.make_img_plot()
        

        self.current_scale = {'label': 'Channel', 'scale': [1,0]}
        self.current_row_scale = {'label': 'Index', 'scale': [1,0]}

    def set_spectral_data(self, data):
        if len(data):
            data_positive = np.clip(data, .1, np.amax(data))
            data_negative = np.clip(-1* data, 0.1 , np.amax(data))
            
            img_data_positive = np.log10( data_positive)
            img_data_negative = -1 * np.log10( data_negative)
            '''img_data_positive[img_data_positive<.5] = 0
            img_data_negative[img_data_negative<.5] = 0'''
            img_data = img_data_positive + img_data_negative
            
            self.img.setImage(img_data.T)
            #self.mask_widget.img_widget.plot_image(img_data, auto_level=True)
        else:
            self.img.clear()
    

    def select_spectrum(self, index):
        self.set_cursor_pos(index, None)

    def select_value(self, val):
        self.set_cursor_pos(None, val)

    def set_image_scale(self, label, scale):
        
        current_label = self.current_scale['label']
        current_translate = self.current_scale['scale'][1]
        current_scale = self.current_scale['scale'][0]

        if label != current_label:
            inverse_translate = -1*current_translate
            inverse_scale =  1/current_scale
            self.img.scale(inverse_scale, 1)
            self.img.translate(inverse_translate, 0)
            
            self.img.translate(scale[1], 0)
            self.img.scale(scale[0], 1)
            
            self.current_scale['label'] = label
            self.current_scale['scale'] = scale
            
            self.p1.setLabel(axis='bottom', text=label)


    def set_image_row_scale(self, row_label, row_scale):
        
        current_row_label = self.current_scale['label']
        current_row_translate = self.current_row_scale['scale'][1]
        current_row_scale = self.current_row_scale['scale'][0]

        if row_label != current_row_label:
            inverse_row_translate = -1*current_row_translate
            inverse_row_scale =  1/current_row_scale
            
            self.img.scale(1, inverse_row_scale)
            self.img.translate(0, inverse_row_translate)
            
            self.img.translate(0, row_scale[1])
            self.img.scale(1, row_scale[0])

            self.current_row_scale['label'] = row_label
            self.current_row_scale['scale'] = row_scale
            
            self.p1.setLabel(axis='left', text=row_label)

    def make_img_plot(self):
        ## Create window with GraphicsView widget
        
        self.p1 = self.addPlot()
        self.p1.setLabel(axis='left', text='Spectrum index')
        self.p1.setLabel(axis='bottom', text='Channel')

       
        self.view = self.p1.getViewBox()
        self.view.setMouseMode(pg.ViewBox.RectMode)
        self.view.setAspectLocked(False)
        ## Create image item
        self.img = pg.ImageItem(border='w')



        #self.img.setScaledMode()
        self.view.addItem(self.img)

        #self.make_lr()

        # Contrast/color control
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.addItem(self.hist)
        

        self.vLine = pg.InfiniteLine(movable=False, pen=pg.mkPen(color=(0, 255, 0), width=2 , style=QtCore.Qt.DashLine))
        self.hLine = pg.InfiniteLine(movable=False, angle = 0, pen=pg.mkPen(color=(200, 200, 200), width=2 , style=QtCore.Qt.DashLine))
        self.hLineFast = pg.InfiniteLine(movable=False,angle = 0, pen=pg.mkPen({'color': '606060', 'width': 1, 'style':QtCore.Qt.DashLine}))
        self.proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=20, slot=self.fastCursorMove)

        #self.vLine.sigPositionChanged.connect(self.cursor_dragged)
        
        self.cursors = [self.hLine, self.hLineFast]
        # self.cursorPoints = [(cursor.index, cursor.channel),(fast.index, fast.channel)]
        self.cursorPoints = [[0,0],[0,0]]
        
        self.view.addItem(self.vLine, ignoreBounds=True)
        self.view.addItem(self.hLine, ignoreBounds=True)
        self.view.addItem(self.hLineFast, ignoreBounds=True)
        self.view.mouseClickEvent = self.customMouseClickEvent

    def fastCursorMove(self, evt):
        pos = evt[0]  ## using signal proxy turns original arguments into a tuple
        if self.view.sceneBoundingRect().contains(pos):
            mousePoint = self.view.mapSceneToView(pos)
            index = round(mousePoint.y(),5)
            if index >= 0:
                self.hLineFast.setPos(index)
                self.cursorPoints[1][0] = index
                self.plotMouseMoveSignal.emit(index)

    def customMouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            self.view.enableAutoRange(enable=1) 
        elif ev.button() == QtCore.Qt.LeftButton: 
            pos = ev.pos()  ## using signal proxy turns original arguments into a tuple
            mousePoint = self.view.mapToView(pos)
            index= mousePoint.y()
            y_scale = self.current_row_scale['scale']
            index_scaled = (index - y_scale[1])/ y_scale[0]

            scale_point = mousePoint.x()
            
            if index >=0 :
                self.set_cursor_pos(index_scaled, scale_point)
                self.plotMouseCursorSignal.emit([index_scaled, scale_point])  
        ev.accept()



    def set_cursor_pos(self, index = None, E=None):
        if E != None:
            self.vLine.blockSignals(True)
            
            self.vLine.setPos(E)
            self.cursorPoints[0] = (self.cursorPoints[0][0],E)
            self.vLine.blockSignals(False)
        if index != None:
            y_scale = self.current_row_scale['scale']
            index_scaled = (int(index)+0.5) * y_scale[0] + y_scale[1]
            self.hLine.blockSignals(True)
            self.hLine.setPos(index_scaled)
            self.cursorPoints[0] = (index_scaled,self.cursorPoints[0][1])
            self.hLine.blockSignals(False)

   