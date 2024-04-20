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


import pyqtgraph as pg
from pyqtgraph import QtCore, mkPen, mkColor, hsvColor
from PyQt5.QtCore import QObject, pyqtSignal, Qt
from PyQt5.QtGui import QColor, QPen, QPainter
from utilities.HelperModule import calculate_color
from hpm.widgets.ExLegendItem import LegendItem
from hpm.widgets.PhasePlot import PhasePlot
'''import pyqtgraph.exporters
import unicodedata'''
from numpy import argmax, nan
from PyQt5 import QtWidgets
'''import copy
import numpy as np'''

from .. import _platform

# used by aEDXD, will upgrade hpMCA to use this later as well
class plotWindow(QtWidgets.QWidget):
    widget_closed = QtCore.pyqtSignal()

    def __init__(self, title, left_label, bottom_label):
        super().__init__()

        self._layout = QtWidgets.QVBoxLayout()  
        self._layout.setContentsMargins(0,0,0,0)
        self.setWindowTitle(title)
        self.win = PltWidget()
        self.win.setLogMode(False,False)
        #self.win.vLineFast.setObjectName(title+'vLineFast')
        self.win.setWindowTitle(title)
        self.win.setBackground(background=(255,255,255))
        
        self.win.setLabel('left',left_label)
        self.win.setLabel('bottom', bottom_label)
        
        self._layout.addWidget(self.win)
        self.setLayout(self._layout)
        self.resize(600,400)
        self.plots = []
        self.fast_cursor = self.win.plotMouseMoveSignal
        self.cursor = self.win.viewBox.plotMouseCursorSignal
        self.win.create_graphics()
        self.win.legend.setParentItem(self.win.viewBox)
        self.win.legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, -10))
        self.legend_items = []


    def set_plot_label(self, text, plot_ind):
        self.win.legend.renameItem(plot_ind, text)
    
    def set_plot_label_color(self, color, plot_ind):
        self.win.legend.setItemColor(plot_ind, color)

    def set_fast_cursor(self, pos):
        self.win.set_cursorFast_pos(pos)

    def set_cursor(self, pos):
        self.win.set_cursor_pos(pos)
        

    def add_line_plot(self, x=[],y=[],color = (0,0,0),Width = 1):
        Pen=mkPen(color, width=Width)
        Plot = self.win.plot(x,y, 
                        pen= Pen, 
                        antialias=True)
        self.plots.append(Plot)

        self.win.legend.addItem(self.plots[-1], '') # can display name in upper right corner in same color 

    def add_fill_between_plot(self, x, curve_1, curve_2, brush=(150, 150, 150, 100)):
        
        c_1 = self.win.plot(x=x, y=curve_1, pen=(0,0,0,0))
        c_2 = self.win.plot(x=x, y=curve_2, pen=(0,0,0,0))
        self.plots.append(c_1)
        self.plots.append(c_2)
        self.win.legend.addItem(c_1, '')
        self.win.legend.addItem(c_2, '')
        self.fbi = pg.FillBetweenItem(c_1, c_2, brush)
        self.win.addItem(self.fbi)
        

    def add_scatter_plot(self, x=[],y=[],color=(200, 200, 200),opacity=100):
        sb = (color[0], color[1],color[2],opacity)
        Plot = self.win.plot(x,y, 
                                pen=None, symbol='o', \
                                symbolPen=None, symbolSize=7, \
                                symbolBrush=sb)
        self.plots.append(Plot)
         # can display name in upper right corner in same color 
        self.win.legend.addItem(self.plots[-1], '')
        self.set_plot_label_color(color, len(self.plots)-1)
        
        
    def clear(self):
        n = len(self.plots)
        for i in range(n):
            self.win.legend.removeItem(self.plots[-1])
            self.win.removeItem(self.plots[-1])
            self.plots.remove(self.plots[-1])
        try:
            self.win.removeItem(self.fbi)
            self.fbi = None
        except:
            pass

    def closeEvent(self, event):
        # Overrides close event to let controller know that widget was closed by user
        self.widget_closed.emit()

    def raise_widget(self):
        self.show()
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
        self.activateWindow()
        self.raise_()
        

class CustomViewBox(pg.ViewBox):  
    plotMouseCursorSignal = pyqtSignal(float)
    plotMouseCursor2Signal = pyqtSignal(float)  
    viewBoxScrollSingal = pyqtSignal(float)
    cursor_y_signal = pyqtSignal(float)
    def __init__(self, *args, **kwds):
        super().__init__()
        
        self.cursor_signals = [self.plotMouseCursorSignal, self.plotMouseCursor2Signal]
        self.vLine = myVLine(movable=False, pen=pg.mkPen(color=(0, 255, 0), width=2 , style=QtCore.Qt.DashLine))
        
        #self.vLine.sigPositionChanged.connect(self.cursor_dragged)
        self.vLineFast = myVLine(movable=False,pen=mkPen({'color': '#606060', 'width': 1, 'style':QtCore.Qt.DashLine}))
        self.cursors = [self.vLine, self.vLineFast]
        self.setMouseMode(self.RectMode)
        self.addItem(self.vLine, ignoreBounds=True)
        self.addItem(self.vLineFast, ignoreBounds=True)
        self.enableAutoRange(self.XYAxes, True)
        self.cursorPoint = 0
        self.cursorPoint_y = 0
        # Enable dragging and dropping onto the GUI 
        self.setAcceptDrops(True) 

    '''
    def cursor_dragged(self, cursor):
        ind = self.cursors.index(cursor)
        pos = cursor.getXPos()
        
        sig = self.cursor_signals[ind]
        sig.emit(pos)    
    '''

    ## reimplement right-click to zoom out
    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            #self.enableAutoRange(self.XYAxes, True)    
            
            self.enableAutoRange(enable=1) 
        elif ev.button() == QtCore.Qt.LeftButton: 
            pos = ev.pos()  ## using signal proxy turns original arguments into a tuple
            mousePoint = self.mapSceneToView(pos)

            mptx = mousePoint.x()
            mpty = mousePoint.y()
            self.cursorPoint=mptx
            self.cursorPoint_y = mpty

            self.plotMouseCursorSignal.emit(self.cursorPoint) 
        ev.accept()   

    def wheelEvent(self, ev, axis=None):

        '''
        don't scroll if control key is down
        '''
        if ev.modifiers() == Qt.ControlModifier and self.scrollModifierEnabled:
            delta = ev.delta() / 120
            self.viewBoxScrollSingal.emit(float(delta))
        else:
            
            return super().wheelEvent(ev, axis)

class myLegendItem(LegendItem):
    def __init__(self, size=None, offset=None, horSpacing=25, verSpacing=0, box=True, labelAlignment='center', showLines=True):
        super().__init__(size=size, offset=offset, horSpacing=horSpacing, verSpacing=verSpacing, box=box, labelAlignment=labelAlignment, showLines=showLines)

    def my_hoverEvent(self, ev):
        pass

    def mouseDragEvent(self, ev):
        pass

class myVLine(pg.InfiniteLine):
    def __init__(self, pos=None, angle=90, pen=None, movable=False, bounds=None,
                 hoverPen=None, label=None, labelOpts=None, span=(0, 1), markers=None, 
                 name=None):
        super().__init__(pos=pos, angle=angle, pen=pen, movable=movable, bounds=bounds,
                 hoverPen=hoverPen, label=label, labelOpts=labelOpts,  
                 name=name)
     

class PltWidget(pg.PlotWidget):
    """
    Subclass of PlotWidget
    """
    plotMouseMoveSignal = pyqtSignal(float)  
    range_changed = QtCore.pyqtSignal(list)
    auto_range_status_changed = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None, colors = None, toolbar_widgets=[], retina_display=False):
        """
        Constructor of the widget
        """
        #app = pg.mkQApp()
        vb = CustomViewBox()  
        self.parent_widget = parent
        self.toolbar_widgets =toolbar_widgets
        self.retina_display = retina_display
        super().__init__(parent, viewBox=vb)
        self.viewBox = self.getViewBox() # Get the ViewBox of the widget
        
        self.cursorPoints = [nan,nan]
        # defined default colors
        self.prefs = { 'plot_background_color': '#ffffff',
                        'data_color': '#2f2f2f',
                        'rois_color': '#00b4ff', 
                        'roi_cursor_color': '#ff0000', 
                        'xrf_lines_color': '#969600',
                        'mouse_cursor_color': '#00cc00', 
                        'mouse_fast_cursor_color': '#323232',
                        'plot_width': 1,
                        'plot_antialias': True}

        # override default colors here:
        if colors != None:
            for c in colors:
                if c in self.prefs:
                    self.prefs[c] = colors[c]
            
        plot_background_color = self.prefs['plot_background_color']
        self.setBackground(background=plot_background_color)
        
        #mouse_fast_cursor_color = self.prefs['mouse_fast_cursor_color']

        self.vLine = self.viewBox.vLine
        self.vLineFast  = self.viewBox.vLineFast
        #self.vLine.setPos(0)
        #self.vLineFast.setPos(0)

        self.selectionMode = False # Selection mode used to mark histo data
        
        #self.viewBox.setMouseMode(self.viewBox.RectMode) # Set mouse mode to rect for convenient zooming
        #self.setMenuEnabled(enableMenu=False)
        # cursor
        
        
        self.proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=20, slot=self.fastCursorMove)
        # self.getViewBox().addItem(self.hLine, ignoreBounds=True)
        self.create_graphics()
        self.pattern_plot = self.plotItem
        self.phases = []
        self.xrf = []
        self.roi_cursor = []
        #self.phases_vlines = []
        self.overlays = []
        self.overlay_names = []
        self.overlay_show = []
        #initialize data plots object names
        self._auto_range = True
        self.plotForeground = None
        self.plotRoi = None
        self.showGrid(x=True, y=True)
        
        self.set_log_mode(False,True)
        self.xAxis = None

    

    def set_cursorFast_pos(self, pos):
        self.vLineFast.blockSignals(True)
        self.vLineFast.setPos(pos)
        self.cursorPoints[1] = pos
        self.vLineFast.blockSignals(False)

    def set_cursor_pos(self, pos):
        self.vLine.blockSignals(True)
        self.vLine.setPos(pos)
        self.cursorPoints[0] = pos
        self.vLine.blockSignals(False)

    def get_cursor_pos(self):
        return self.cursorPoints[0]

    def get_cursorFast_pos(self):
        return self.cursorPoints[1]

    def set_log_mode(self, x, y):
        self.LM = (x,y)
        self.setLogMode(*self.LM)

    def get_colors(self):
        return self.prefs
        
    def set_colors(self, params):
        # originally used to set colors, thus the name, but now used for other parameters as well, width, aliasing, etc.
        for p in params:
            if p in self.prefs:
                color = params[p]
                self.prefs[p] = color
                if p == 'plot_background_color':
                    self.setBackground(color)
                    if self.parent_widget != None:
                        self.parent_widget.setStyleSheet( 'background-color: '+ color+';')
                    #toolbar_widgets are widgets containing or surrounding the plot
                    #this sets their backgorund to match the plot
                    for widget in self.toolbar_widgets:
                        widget.setStyleSheet( 'background-color: '+ color+';')
                        
                elif p == 'data_color':
                    if self.plotForeground is not None:
                        self.plotForeground.setPen(color)
                        self.legend.setItemColor(0, color)
                elif p == 'rois_color':
                    if self.plotRoi is not None:
                        self.plotRoi.setPen(color)
                        self.legend.setItemColor(1, color)
                elif p == 'roi_cursor_color':
                    if len(self.roi_cursor):
                        self.set_roi_cursor_color(0, params[p])
                elif p == 'xrf_lines_color':
                    pass
                elif p == 'mouse_cursor_color':
                    pass
                elif p == 'mouse_fast_cursor_color':
                    pass
                '''elif p == 'plot_width':
                    # this is currently not used because setting width more than 1 makes the plot slow
                    # maybe if they fix that issue in a future version of pyqtgraph
                    if self.plotForeground is not None:
                        pen = self.foreground_pen # need to rename this consistently at some point, see note above
                        pen.setWidth(color)
                        self.plotForeground.setPen(pen)
                    if self.plotRoi is not None:
                        pen = self.roi_pen # need to rename this consistently at some point, see note above
                        pen.setWidth(color)
                        self.plotRoi.setPen(pen)'''
                    
                

    def export_plot_png(self,filename):
        self.vLine.hide()
        self.vLineFast.hide()
        exporter = pg.exporters.ImageExporter(self.plotItem)
        exporter.params.param('width').setValue(1920)
        exporter.params.param('height').setValue(1080)
     
        exporter.export(filename)
        self.vLine.show()
        self.vLineFast.show()
    
    def export_plot_svg(self,filename):
        self.vLine.hide()
        self.vLineFast.hide()
        
        exporter = pg.exporters.SVGExporter(self.plotItem)
        
        
        exporter.export(filename)
        self.vLine.show()
        self.vLineFast.show()

    def emit_sig_range_changed(self):
        pass

    def create_plots(self, xAxis,data,roiHorz,roiData, xLabel):
        # initialize some plots
        #self.pattern_plot.buttonsHidden = True
        antialias = self.prefs['plot_antialias']
        plot_width = 1 #self.prefs['plot_width']
        self.setLabel('left', 'Counts')
        data_color = self.prefs['data_color']
        self.foreground_pen = pg.mkPen(color=data_color, width=plot_width)
      
        self.plotForeground = pg.PlotDataItem(xAxis, data, title="",
                 pen=self.foreground_pen, connect="finite" , antialias=antialias)
        self.addItem(self.plotForeground)

        # plot legend items 
        self.legend.addItem(self.plotForeground, '') # can display name in upper right corner in same color 
        self.legend.setParentItem(self.viewBox)
        
        self.legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, -10))
        self.phases_legend.setParentItem(self.viewBox)
        self.phases_legend.anchor(itemPos=(0, 0), parentPos=(0, 0), offset=(0, -10))

        self.xrf_legend.setParentItem(self.viewBox)
        self.xrf_legend.anchor(itemPos=(0.5, 0), parentPos=(0.5, 0), offset=(0, -10))

        #self.roi_cursor_legend.setParentItem(self.viewBox)
        #self.roi_cursor_legend.anchor(itemPos=(0, 0), parentPos=(0, 0), offset=(0, 30))

        
        # initialize roi plot 
        rois_color = self.prefs['rois_color']
        self.roi_pen = pg.mkPen(color=rois_color, width=plot_width)

        self.plotRoi = pg.PlotDataItem(roiHorz, roiData, 

             pen=self.roi_pen, connect="finite", antialias=antialias)
        self.addItem(self.plotRoi)  
        self.legend.addItem(self.plotRoi, '')
        self.setLabel('bottom', xLabel) 

    def plot_image(self, data):
        self.plotData(data[0], data[1])

    def plotData(self, xAxis,data,roiHorz=[],roiData=[], xLabel='', dataLabel=''):
        
        self.xAxis = xAxis
        if self.plotForeground == None:
            self.create_plots(xAxis,data,roiHorz,roiData, xLabel)
        else:
            self.plotForeground.setData(xAxis, data) 
            self.plotRoi.setData(roiHorz, roiData) 
       
        # if nonzero ROI data, show ROI legend on plot
        if len(roiHorz) > 0: roiLabel = 'ROIs'
        else:   roiLabel = ''
        
        self.legend.renameItem(0, dataLabel)
        self.legend.renameItem(1, roiLabel)
        self.setLabel('bottom', xLabel)     

    def lin_reg_mode(self, mode,**kwargs):
        if mode == 'Add':
            width = kwargs.get('width', None)
            if width is None:
                width = 0.6
            self.lr = pg.LinearRegionItem([self.cursorPoints[0]-width/2,self.cursorPoints[0]+width/2])
            self.lr.setZValue(-10)
            self.addItem(self.lr)
        if mode == 'Set':
            reg = self.lr.getRegion()
     
            self.removeItem(self.lr)
            return reg

    def setPlotMouseMode(self, mode):
        vb = self.getViewBox()
        if mode==0:
            mMode = pg.ViewBox.RectMode
        else:
            mMode = pg.ViewBox.PanMode
        vb.setMouseMode(mMode)
        
    def fastCursorMove(self, evt):
        pos = evt[0]  ## using signal proxy turns original arguments into a tuple
        if self.sceneBoundingRect().contains(pos):
            mousePoint = self.getViewBox().mapSceneToView(pos)
            index = mousePoint.x()
            self.plotMouseMoveSignal.emit(index)
            
           # self.hLine.setPos(mousePoint.y())

    

    ########### taken from DIOPTAS:

    def create_graphics(self):

        self.legend = myLegendItem(horSpacing=20, box=False, verSpacing=-3, labelAlignment='right', showLines=False)
        
        self.phases_legend = myLegendItem(horSpacing=20, box=False, verSpacing=-3, labelAlignment='left', showLines=False)
        self.xrf_legend = myLegendItem(horSpacing=20, box=False, verSpacing=-3, labelAlignment='left', showLines=False)
        #self.roi_cursor_legend = LegendItem(horSpacing=20, box=False, verSpacing=-3, labelAlignment='right', showLines=False)


    #### control phases #### 

    

    def add_phase(self, name, positions, intensities, baseline, color):
        self.phases.append(PhasePlot(self.pattern_plot, \
                        self.phases_legend, positions, intensities, \
                        name, baseline, line_width=2,color=color))
        

    def set_phase_color(self, ind, color):
        self.phases[ind].set_color(color)
        self.phases_legend.setItemColor(ind, color)

    def hide_phase(self, ind):
        self.phases[ind].hide()
        self.phases_legend.hideItem(ind)

    def show_phase(self, ind):
        self.phases[ind].show()
        self.phases_legend.showItem(ind)

    def rename_phase(self, ind, name):
        self.phases_legend.renameItem(ind, name)

    def update_phase_intensities(self, ind, positions, intensities, baseline=.5):
        if len(self.phases):
            self.phases[ind].update_intensities(positions, intensities, baseline)

    def update_phase_line_visibility(self, ind):
        x_range = self.plotForeground.dataBounds(0)
        self.phases[ind].update_visibilities(x_range)

    def update_phase_line_visibilities(self):
        x_range = self.plotForeground.dataBounds(0)
        for phase in self.phases:
            phase.update_visibilities(x_range)

    def del_phase(self, ind):
        self.phases[ind].remove()
        del self.phases[ind]

    #### END control phases ####

    #### control xrf #### 

    def add_xrf(self, name, positions, intensities, baseline):
        self.xrf.append(PhasePlot(self.pattern_plot, \
                    self.xrf_legend, positions, intensities, name, \
                    baseline,'dash',line_width=2))
        
        return self.xrf[-1].color

    def set_xrf_color(self, ind, color):
        self.xrf[ind].set_color(color)
        self.xrf_legend.setItemColor(ind+1, color)

    def hide_xrf(self, ind):
        self.xrf[ind].hide()
        self.xrf_legend.hideItem(ind+1)

    def show_xrf(self, ind):
        self.xrf[ind].show()
        self.xrf_legend.showItem(ind+1)

    def rename_xrf(self, ind, name):
        self.xrf_legend.renameItem(ind+1, name)

    def update_xrf_intensities(self, ind, positions, intensities, baseline=.5):
        if len(self.xrf):
            self.xrf[ind].update_intensities(positions, intensities, baseline)
            
    def update_xrf_line_visibility(self, ind):
        x_range = self.plotForeground.dataBounds(0)
        self.xrf[ind].update_visibilities(x_range)

    def update_xrf_line_visibilities(self):
        x_range = self.plotForeground.dataBounds(0)
        for xrf in self.xrf:
            xrf.update_visibilities(x_range)

    def del_xrf(self, ind):
        self.xrf[ind].remove()
        del self.xrf[ind]

    #### END control xrf ####

    #### control roi_cursor #### 

    def add_roi_cursor(self, name, positions, intensities, baseline):
        self.roi_cursor.append(PhasePlot(self.pattern_plot, \
                    self.xrf_legend, positions, intensities, \
                    name, baseline,'solid',line_width=5,color=(255,0,0)))
        return self.roi_cursor[-1].color

    def set_roi_cursor_color(self, ind, color):
        self.roi_cursor[ind].set_color(color)
        self.xrf_legend.setItemColor(ind, color)

    def hide_roi_cursor(self, ind):
        self.roi_cursor[ind].hide()
        self.xrf_legend.hideItem(ind)

    def show_roi_cursor(self, ind):
        self.roi_cursor[ind].show()
        self.xrf_legend.showItem(ind)

    def rename_roi_cursor(self, ind, name):
        self.xrf_legend.renameItem(ind, name)

    def update_roi_cursor_intensities(self, ind, positions, intensities, baseline):
        if len(self.roi_cursor):
            self.roi_cursor[ind].update_intensities(positions, \
                    intensities, baseline)

    def update_roi_cursor_line_visibility(self, ind):
        x_range = self.plotForeground.dataBounds(0)
        self.roi_cursor[ind].update_visibilities(x_range)

    def update_roi_cursor_line_visibilities(self):
        x_range = self.plotForeground.dataBounds(0)
        for roi_cursor in self.roi_cursor:
            roi_cursor.update_visibilities(x_range)

    def del_roi_cursor(self, ind):
        self.roi_cursor[ind].remove()
        del self.roi_cursor[ind]

    #### END control roi_cursor ####

 
    ###################  OVERLAY STUFF  ################### from DIOPTAS
    def update_graph_range(self):
        pass

    def add_overlay(self, pattern, show=True):
        [x,y]=pattern.get_pattern()
        
        color = calculate_color(len(self.overlays) + 1)
        self.overlays.append(pg.PlotDataItem(pen=pg.mkPen(color=color, width=1), \
                    antialias=True, conntect='finite'))
        #self.overlays.append(pg.PlotDataItem(x, y, pen=pg.mkPen(color=color, width=1), antialias=True, conntect='finite'))
        if x is not None:
            self.overlays[-1].setData(x, y)
        self.overlay_names.append(pattern.name)
        self.overlay_show.append(True)
        if show:
            self.pattern_plot.addItem(self.overlays[-1])
            self.legend.addItem(self.overlays[-1], pattern.name)
            self.update_graph_range()
        return color

    def remove_overlay(self, ind):
        self.pattern_plot.removeItem(self.overlays[ind])
        self.legend.removeItem(self.overlays[ind])
        self.overlays.remove(self.overlays[ind])
        self.overlay_names.remove(self.overlay_names[ind])
        self.overlay_show.remove(self.overlay_show[ind])
        self.update_graph_range()

    def hide_overlay(self, ind):
        self.pattern_plot.removeItem(self.overlays[ind])
        self.legend.hideItem(ind + 2)
        self.overlay_show[ind] = False
        self.update_graph_range()

    def show_overlay(self, ind):
        self.pattern_plot.addItem(self.overlays[ind])
        self.legend.showItem(ind + 2)
        self.overlay_show[ind] = True
        self.update_graph_range()

    def update_overlay(self, pattern, ind):
        [x, y] = pattern.get_pattern()
        if x is not None:
            self.overlays[ind].setData(x, y)
        else:
            self.overlays[ind].setData([], [])
        self.update_graph_range()

    def set_overlay_color(self, ind, color):
        self.overlays[ind].setPen(pg.mkPen(color=color, width=1))
        self.legend.setItemColor(ind + 2, color)

    def rename_overlay(self, ind, name):
        self.legend.renameItem(ind + 2, name)

    def move_overlay_up(self, ind):
        new_ind = ind - 1
        self.overlays.insert(new_ind, self.overlays.pop(ind))
        self.overlay_names.insert(new_ind, self.overlay_names.pop(ind))
        self.overlay_show.insert(new_ind, self.overlay_show.pop(ind))

        color = self.legend.legendItems[ind + 2][1].opts['color']
        label = self.legend.legendItems[ind + 2][1].text
        self.legend.legendItems[ind + 2][1].setAttr('color', self.legend.legendItems[new_ind + 2][1].opts['color'])
        self.legend.legendItems[ind + 2][1].setText(self.legend.legendItems[new_ind + 2][1].text)
        self.legend.legendItems[new_ind + 2][1].setAttr('color', color)
        self.legend.legendItems[new_ind + 2][1].setText(label)

    def move_overlay_down(self, cur_ind):
        self.overlays.insert(cur_ind + 2, self.overlays.pop(cur_ind))
        self.overlay_names.insert(cur_ind + 2, self.overlay_names.pop(cur_ind))
        self.overlay_show.insert(cur_ind + 2, self.overlay_show.pop(cur_ind))

        color = self.legend.legendItems[cur_ind + 2][1].opts['color']
        label = self.legend.legendItems[cur_ind + 2][1].text
        self.legend.legendItems[cur_ind + 2][1].setAttr('color', self.legend.legendItems[cur_ind + 3][1].opts['color'])
        self.legend.legendItems[cur_ind + 2][1].setText(self.legend.legendItems[cur_ind + 3][1].text)
        self.legend.legendItems[cur_ind + 3][1].setAttr('color', color)
        self.legend.legendItems[cur_ind + 3][1].setText(label)

    ##########  END OF OVERLAY STUFF  ##################

