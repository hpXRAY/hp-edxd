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
import imp
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
from hpm.widgets.CustomImageWidget import CustomImageWidget

class MultiSpectraWidget(QtWidgets.QWidget):

    color_btn_clicked = QtCore.pyqtSignal(int, QtWidgets.QWidget)
    #env_btn_clicked = QtCore.pyqtSignal(int)
    show_cb_state_changed = QtCore.pyqtSignal(int, int)
    pv_item_changed = QtCore.pyqtSignal(int, str)
    widget_closed = QtCore.pyqtSignal()
    key_signal = pyqtSignal(str)
    
    linearRegionMovedSignal = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self._layout = QtWidgets.QVBoxLayout()  

        self.p1 : pg.PlotWidget
        self.setWindowTitle('Multiple spectra view')
        self.button_widget = QtWidgets.QWidget(self)
        self.button_widget.setMaximumHeight(40)
        self.button_widget.setObjectName('multispectra_control_button_widget')
        self._button_layout = QtWidgets.QHBoxLayout()
        self._button_layout.setContentsMargins(0, 0, 0, 0)
        self._button_layout.setSpacing(6)

       
        self.show_roi_btn = FlatButton('Show ROIs')
        self.show_roi_btn.setCheckable(True)
        self.show_roi_btn.setChecked(False)
        self.show_roi_btn.setMaximumWidth(90)
        self.show_roi_btn.setMinimumWidth(90)

        self.add_roi_btn = FlatButton('Add ROI')
        self.add_roi_btn.setMaximumWidth(90)
        self.add_roi_btn.setMinimumWidth(90)

        self.delete_roi_btn = FlatButton('Delete')
        self.delete_roi_btn.setMaximumWidth(90)
        self.delete_roi_btn.setMinimumWidth(90)
        self.clear_roi_btn = FlatButton('Clear')
        self.clear_roi_btn.setMaximumWidth(90)
        self.clear_roi_btn.setMinimumWidth(90)

        self.align_btn = FlatButton('Align')
        self.align_btn.setMaximumWidth(90)
        self.align_btn.setMinimumWidth(90)
        self.amorphous_btn = FlatButton('Amorphous')
        self.amorphous_btn.setMaximumWidth(90)
        self.amorphous_btn.setMinimumWidth(90)
       

        self.refresh_folder_btn = FlatButton('Refresh')
        self.refresh_folder_btn.setMaximumWidth(90)
        self.refresh_folder_btn.setMinimumWidth(90)
        self.transpose_btn = FlatButton(f'Transpose')
        self.transpose_btn.setMaximumWidth(90)
        self.transpose_btn.setMinimumWidth(90)
        self.copy_rois_btn = FlatButton('Copy ROIs')
        self.copy_rois_btn.setMaximumWidth(90)
        self.copy_rois_btn.setMinimumWidth(90)
        self.cal_btn = FlatButton('Calibrate')
        self.cal_btn.setMaximumWidth(90)
        self.cal_btn.setMinimumWidth(90)

        

        #self._button_layout.addWidget(self.refresh_folder_btn)
        self._button_layout.addSpacerItem(HorizontalSpacerItem())
        self._button_layout.addWidget(self.show_roi_btn)
        self._button_layout.addWidget(self.add_roi_btn)
        self._button_layout.addWidget(self.delete_roi_btn)
        self._button_layout.addWidget(self.clear_roi_btn)
    
        self._button_layout.addWidget(self.align_btn)
    
        
        self._button_layout.addWidget(self.copy_rois_btn)
        
        self._button_layout.addWidget(self.cal_btn)
        
        self.button_widget.setLayout(self._button_layout)
        self._layout.addWidget(self.button_widget)
       
        self._body_layout = QtWidgets.QHBoxLayout()
        self.file_view_tabs= QtWidgets.QTabWidget(self)
        
        self.file_view_tabs.setObjectName("file_view_tabs")
        #self.make_img_plot()
        self.plot_widget = QtWidgets.QWidget()
        self._plot_widget_layout = QtWidgets.QVBoxLayout(self.plot_widget)
        self._plot_widget_layout.setContentsMargins(0,0,0,0)

        self.win = CustomImageWidget()
        self._plot_widget_layout.addWidget( self.win)

        self.HorizontalScaleWidget = QtWidgets.QWidget()
        self.HorizontalScaleLayout = QtWidgets.QHBoxLayout(self.HorizontalScaleWidget)
        self.HorizontalScaleLayout.setSpacing(0)
        self.HorizontalScaleLayout.setContentsMargins(0,0,0,0)
        self.HorizontalScale_btn_group = QtWidgets.QButtonGroup()
        self.radioE = QtWidgets.QPushButton(self.HorizontalScaleWidget)
        self.radioE.setObjectName("radioE")
        self.HorizontalScaleLayout.addWidget(self.radioE)
        self.radioq = QtWidgets.QPushButton(self.HorizontalScaleWidget)
        self.radioq.setObjectName("radioq")
        self.HorizontalScaleLayout.addWidget(self.radioq)
        self.radiotth = QtWidgets.QPushButton(self.HorizontalScaleWidget)
        self.radiotth.setObjectName("radiotth")
        self.HorizontalScaleLayout.addWidget(self.radiotth)
        self.radioChannel = QtWidgets.QPushButton(self.HorizontalScaleWidget)
        self.radioChannel.setObjectName("radioChannel")
        self.HorizontalScaleLayout.addWidget(self.radioChannel)
        self.radioAligned = QtWidgets.QPushButton(self.HorizontalScaleWidget)
        self.radioAligned.setObjectName("radioAligned")
        self.HorizontalScaleLayout.addWidget(self.radioAligned)

        self.HorizontalScaleLayout.addSpacerItem(HorizontalSpacerItem())

        self.radioE.setCheckable(True)
        self.radioq.setCheckable(True)
        self.radiotth.setCheckable(True)
        self.radioChannel.setCheckable(True)
        self.radioAligned.setCheckable(True)

        self.radioE.setText("E")
        self.radioq.setText("q")
        self.radiotth.setText(f'2\N{GREEK SMALL LETTER THETA}')
        self.radioChannel.setText("Channel")
        self.radioAligned.setText("Aligned")

        self.HorizontalScale_btn_group.addButton(self.radioE)
        self.HorizontalScale_btn_group.addButton(self.radioq)
        self.HorizontalScale_btn_group.addButton(self.radiotth)
        self.HorizontalScale_btn_group.addButton(self.radioChannel)
        self.HorizontalScale_btn_group.addButton(self.radioAligned)

        self.radioChannel.setChecked(True)
        #self._plot_widget_layout.addWidget(self.HorizontalScaleWidget)

        self.navigation_buttons = QtWidgets.QWidget()
        self._nav_layout = QtWidgets.QHBoxLayout(self.navigation_buttons)
        self._nav_layout.setContentsMargins(0,0,0,0)
        self.prev_btn = QtWidgets.QPushButton('< Previous')
        self.next_btn = QtWidgets.QPushButton('Next >')
        self._nav_layout.addWidget(self.prev_btn)
        self._nav_layout.addWidget(self.next_btn)
        self._plot_widget_layout.addWidget(self.navigation_buttons)
        
        self.file_view_tabs.addTab(self.plot_widget, 'Spectra')

        self.file_list_view = QtWidgets.QListWidget()
        #self.mask_widget = MaskWidget()
        self.file_view_tabs.addTab(self.file_list_view, 'Files')



        self._body_layout.addWidget(self.file_view_tabs)
        

        self._layout.addLayout(self._body_layout)
        self._layout.addWidget(self.HorizontalScaleWidget)
        self.file_name = QtWidgets.QLabel('')
        self.file_name_fast = QtWidgets.QLabel('')
        self.file_name_fast.setObjectName("file_name_fast")
        self._layout.addWidget(self.file_name_fast)
        self.file_name_fast.setStyleSheet("""
                color: #909090;
        """)
        self._layout.addWidget(self.file_name)
        self.setLayout(self._layout)
        self.style_widgets()
        self.env_show_cbs = []
        self.pv_items = []
        self.index_items = []
        self.resize(500,633)

        self.HorizontalScaleWidget.setStyleSheet("""
            QPushButton{
                
                border-radius: 0px;
            }
            #radioE {

                border-top-left-radius:5px;
                border-bottom-left-radius:5px;
            }
            #radioAligned {

                border-top-right-radius:5px;
                border-bottom-right-radius:5px;
            }
       
	    """)

        self.current_scale = {'label': 'Channel', 'scale': [1,0]}
        self.current_row_scale = {'label': 'Index', 'scale': [1,0]}

        self.scales_btns = {'E':self.radioE,
                            'q':self.radioq,
                            'Aligned': self.radioAligned,
                            'Channel':self.radioChannel,
                            '2 theta':self.radiotth}

        self.alignment_rois = []

    def add_alignment_roi(self, roi, show):
        #self.alignment_rois. append(roi)
        self._add_alignment_roi(roi, show)


    def set_alignment_roi_visibility(self, visible):
        lr : pg.LinearRegionItem
        for lr in self.alignment_rois:
            if not visible:
                try:
                    self.p1.removeItem(lr)
                except:
                    pass
            else:
            
                try:
                    self.p1.addItem(lr)
                except:
                    pass

    def set_scales_enabled_states(self, enabled=['Channel']):
        for btn in self.scales_btns:
            self.scales_btns[btn].setEnabled(btn in enabled)

    def get_selected_unit(self):
        horzScale = 'Channel'
        if self.radioE.isChecked() == True:
            horzScale = 'E'
        elif self.radioq.isChecked() == True:
            horzScale = 'q'
        elif self.radiotth.isChecked() == True:
            horzScale = '2 theta'
        elif self.radioAligned.isChecked() == True:
            horzScale = 'Aligned'
        return horzScale

    def set_unit_btn(self, unit):
        if unit in self.scales_btns:
            btn = self.scales_btns[unit]
            btn.setChecked(True)



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
            self.select_file(row)
        else:
            self.file_list_view.clear()

 

    def set_linear_regions(self, rois, show:False):
        lr : pg.LinearRegionItem
        if len(self.alignment_rois):
            for lr in self.alignment_rois:
                self.p1.removeItem(lr)
                lr.sigRegionChangeFinished.disconnect(self.lr_moved)
            self.alignment_rois = []
        for roi in rois:
            self._add_alignment_roi(roi, show)
      

    def _add_alignment_roi(self, roi, show):
        lr = pg.LinearRegionItem()
            
        lr.setZValue(0)
        lr.setRegion(roi)
        lr.sigRegionChangeFinished.connect(self.lr_moved)
        self.alignment_rois.append(lr)

        if show:
            
            self.p1.addItem(lr)

    def lr_moved(self):
        lr : pg.LinearRegionItem
        rois = []
        for lr in self.alignment_rois:
            roi = lr.getRegion()
            rois.append([int(roi[0]),int(roi[1])])
        self.linearRegionMovedSignal.emit(rois)

    def select_file(self,index):
        self.file_list_view.blockSignals(True)
        self.file_list_view.setCurrentRow(index)
        self.file_list_view.blockSignals(False)

    def select_spectrum(self, index):
        self.win.set_cursor_pos(index, None)

    def select_value(self, val):
        self.win.set_cursor_pos(None, val)

    
        
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
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
        self.activateWindow()
        self.raise_()

  

