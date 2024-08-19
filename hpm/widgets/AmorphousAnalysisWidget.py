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
from PyQt6.QtCore import QObject, pyqtSignal, Qt
from PyQt6 import QtWidgets, QtCore, QtGui
import copy, os
import numpy as np
import pyqtgraph as pg
from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight, TabWidget
from hpm.widgets.PltWidget import PltWidget
from hpm.widgets.MaskWidget import MaskWidget
from hpm.widgets.plot_widgets import ImgWidget2
from hpm.controllers.DisplayPrefsController import DisplayPreferences
from hpm.widgets.UtilityWidgets import save_file_dialog, open_file_dialog, open_files_dialog, CifConversionParametersDialog, open_folder_dialog
from hpm.controllers.MaskController import MaskController
from hpm.models.MaskModel import MaskModel


class AmorphousAnalysisWidget(QtWidgets.QWidget):

    color_btn_clicked = QtCore.pyqtSignal(int, QtWidgets.QWidget)
    #env_btn_clicked = QtCore.pyqtSignal(int)
    show_cb_state_changed = QtCore.pyqtSignal(int, int)
    pv_item_changed = QtCore.pyqtSignal(int, str)
    widget_closed = QtCore.pyqtSignal()
    key_signal = pyqtSignal(str)
    plotMouseMoveSignal = pyqtSignal(float)  
    plotMouseCursorSignal = pyqtSignal(list)

    def __init__(self,directories):
        super().__init__()

        self.directories = directories
        self._layout = QtWidgets.QVBoxLayout()  
        self.setWindowTitle('Amorphous analysis')
        self.resize(1000,700)
        self.button_widget = QtWidgets.QWidget(self)
        self.button_widget.setMaximumHeight(40)
        self.button_widget.setObjectName('multispectra_control_button_widget')
        self._button_layout = QtWidgets.QHBoxLayout()
        self._button_layout.setContentsMargins(0, 0, 0, 0)
        self._button_layout.setSpacing(6)


        self.step1_btn = FlatButton('Step 1')
        self.step1_btn.setMaximumWidth(90)
        self.step1_btn.setMinimumWidth(90)
        self.step2_btn = FlatButton('Step 2')
        self.step2_btn.setMaximumWidth(90)
        self.step2_btn.setMinimumWidth(90)
        self.step3_btn = FlatButton('Step 3')
        self.step3_btn.setMaximumWidth(90)
        self.step3_btn.setMinimumWidth(90)
        self.step4_btn = FlatButton('Step 4')
        self.step4_btn.setMaximumWidth(90)
        self.step4_btn.setMinimumWidth(90)
        self.step5_btn = FlatButton('Step 5')
        self.step5_btn.setMaximumWidth(90)
        self.step5_btn.setMinimumWidth(90)

        self.edit_btn = FlatButton('Edit')
        self.delete_btn = FlatButton('Delete')
        self.clear_btn = FlatButton('Clear')


        self._button_layout.addWidget(self.step1_btn)
        self._button_layout.addWidget(self.step2_btn)
        self._button_layout.addWidget(self.step3_btn)
        self._button_layout.addWidget(self.step4_btn)
        self._button_layout.addWidget(self.step5_btn)
        self._button_layout.addSpacerItem(HorizontalSpacerItem())
      
        
        self.button_widget.setLayout(self._button_layout)
        self._layout.addWidget(self.button_widget)
       
        self._body_layout = QtWidgets.QHBoxLayout()

        self.plot_tabs = TabWidget()
        #style_sheet = "QTabBar::tab { min-width: 20px; }"
        #self.plot_tabs.setStyleSheet(style_sheet)
        
        self.scratch_plots = {}
        self.scratch_plots_btns = {}
        self.mask_controllers = {}
        self.displayPrefs = {}

        self._body_layout.addWidget(self.plot_tabs)

        self._layout.addLayout(self._body_layout)
        
        self.setLayout(self._layout)
        self.style_widgets()
        self.env_show_cbs = []
        self.pv_items = []
        self.index_items = []
        


        self.current_scale = {'label': 'Channel', 'scale': [1,0]}
        self.current_row_scale = {'label': 'Index', 'scale': [1,0]}

        

    def add_scratch_plot(self, name, dims=2, mask = False):
        
        if dims ==2:
            if mask:
                mask_model = MaskModel()
                mask_widget = MaskWidget() 
                mask_controller = MaskController(mask_model, mask_widget, self.directories)
                self.mask_controllers[name] = mask_controller
                plot = mask_widget.img_widget
                self.add_tab(mask_widget, str(len(self.scratch_plots)), name)
            else:
                plot = ImgWidget2()
                self.add_tab(plot, str(len(self.scratch_plots)), name)
        elif dims == 1:
            plot = PltWidget()
            plot.set_log_mode(False,False)
            self.displayPrefs [ name] =  DisplayPreferences(plot)
            self.add_tab(plot, str(len(self.scratch_plots)), name)

        self.scratch_plots[name]=plot
        
        return plot
        
    def add_tab(self, widget, label, desc):
        btn = QtWidgets.QPushButton('Export')
        self.scratch_plots_btns[desc] = btn
        btn.clicked.connect(partial(self.handle_export_btn, desc))
        w = QtWidgets.QWidget()
        w_layout = QtWidgets.QVBoxLayout(w)
        w_layout.addWidget(QtWidgets.QLabel(desc))
        w_layout.addWidget(btn)
        w_layout.addWidget(widget)
        self.plot_tabs.addTab(w, label, desc)

    def handle_export_btn(self, desc):
        plot = self.scratch_plots[desc]
      
        filename = save_file_dialog(
            self, "Export plot.",'',('png (*.png)'))
        if filename != '':
            self.export(plot, filename)

    def export(self, plot, filename):
        
        if filename.endswith('.png'):
            plot.export_plot_png(filename)

    def set_scales_enabled_states(self, enabled=['Channel']):
        for btn in self.scales_btns:
            self.scales_btns[btn].setEnabled(btn in enabled)

    

    def plot_data(self, x=[],y=[]):
        self.line_plot_widget.plotData(x, y)

    def get_selected_row(self):
        selected  = self.file_list_view.selectionModel().selectedRows()
        if len(selected):
            row = selected[0].row()
        else:
            row = -1
        return row

    def reload_files(self, files):
        if len(files):
            row =  self.get_selected_row()
            self.file_list_view.blockSignals(True)
            self.file_list_view.clear()
            self.file_list_view.addItems(files)
            self.file_list_view.blockSignals(False)
            if row < 0 or row >= len(files):
                row = 0
            self.file_list_view.setCurrentRow(row)
        else:
            self.file_list_view.clear()

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
            self.mask_widget.img_widget.plot_image(img_data, auto_level=True)
        else:
            self.img.clear()

    def select_file(self,index):
        self.file_list_view.blockSignals(True)
        self.file_list_view.setCurrentRow(index)
        self.file_list_view.blockSignals(False)

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
        self.win = pg.GraphicsLayoutWidget(parent=self)
        self.p1 = self.win.addPlot()
        self.p1.setLabel(axis='left', text='Spectrum index')
        self.p1.setLabel(axis='bottom', text='Channel')

        #self.plot = pg.PlotItem(self.win)
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
        self.win.addItem(self.hist)
        

        self.vLine = pg.InfiniteLine(movable=False, pen=pg.mkPen(color=(0, 255, 0), width=2 , style=QtCore.Qt.PenStyle.DashLine))
        self.hLine = pg.InfiniteLine(movable=False, angle = 0, pen=pg.mkPen(color=(200, 200, 200), width=2 , style=QtCore.Qt.PenStyle.DashLine))
        self.hLineFast = pg.InfiniteLine(movable=False,angle = 0, pen=pg.mkPen({'color': '606060', 'width': 1, 'style':QtCore.Qt.PenStyle.DashLine}))
        self.proxy = pg.SignalProxy(self.win.scene().sigMouseMoved, rateLimit=20, slot=self.fastCursorMove)

        #self.vLine.sigPositionChanged.connect(self.cursor_dragged)
        
        self.cursors = [self.hLine, self.hLineFast]
        # self.cursorPoints = [(cursor.index, cursor.channel),(fast.index, fast.channel)]
        self.cursorPoints = [(0,0),(0,0)]
        
        self.view.addItem(self.vLine, ignoreBounds=True)
        self.view.addItem(self.hLine, ignoreBounds=True)
        self.view.addItem(self.hLineFast, ignoreBounds=True)
        self.view.mouseClickEvent = self.customMouseClickEvent


    def make_lr(self):
        self.lr1_p = pg.LinearRegionItem()
        #self.lr1_p.setZValue(-10)
        self.lr2_p = pg.LinearRegionItem()
        #self.lr2_p.setZValue(-10)

        self.echo_bounds_p = [self.lr1_p, self.lr2_p]
        self.view.addItem(self.lr1_p)
        self.view.addItem(self.lr2_p) 
        self.lr1_p.setRegion([0, 20])
        self.lr2_p.setRegion([40, 60])

        

    def fastCursorMove(self, evt):
        pos = evt[0]  ## using signal proxy turns original arguments into a tuple
        if self.view.sceneBoundingRect().contains(pos):
            mousePoint = self.view.mapSceneToView(pos)
            index = mousePoint.y()
            if index >= 0:
                self.hLineFast.setPos(index)
                self.plotMouseMoveSignal.emit(index)

    def customMouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            self.view.enableAutoRange(enable=1) 
        elif ev.button() == QtCore.Qt.MouseButton.LeftButton: 
            pos = ev.pos()  ## using signal proxy turns original arguments into a tuple
            mousePoint = self.view.mapSceneToView(pos)
            index= mousePoint.y()
            y_scale = self.current_row_scale['scale']
            index_scaled = (index - y_scale[1])/ y_scale[0]

            scale_point = mousePoint.x()
            
            if index >=0 :
                self.set_cursor_pos(index_scaled, scale_point)
                self.plotMouseCursorSignal.emit([index_scaled, scale_point])  
        ev.accept()

    def set_cursorFast_pos(self, index, E):
        self.hLine.blockSignals(True)
        
        self.hLine.setPos(int(index)+0.5)
        self.cursorPoints[1] = (index,E)
        self.hLineFast.blockSignals(False)


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
        
    def keyPressEvent(self, e):
        sig = None
        if e.key() == Qt.Key_Up:
            sig = 'up'
        if e.key() == Qt.Key_Down:
            sig = 'down'                
        if e.key() == Qt.Key_Delete:
            sig = 'delete'
        if e.key() == Qt.Key_Right:
            sig = 'right'   
        if e.key() == Qt.Key_Left:
            sig = 'left'   
        if e.key() == Qt.Key_Backspace:
            sig = 'delete'  
        if sig is not None:
            self.key_signal.emit(sig)
        else:
            super().keyPressEvent(e)

    def style_widgets(self):
        self.setStyleSheet("""
            #env_control_button_widget FlatButton {
                min-width: 90;
            }
        """)

    def closeEvent(self, event):
        # Overrides close event to let controller know that widget was closed by user
        self.widget_closed.emit()

    def raise_widget(self):
        self.show()
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowState.WindowMinimized | QtCore.Qt.WindowState.WindowActive)
        self.activateWindow()
        self.raise_()

  
