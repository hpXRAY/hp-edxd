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



from functools import partial
from numpy import argmax, nan
from PyQt6 import QtWidgets, QtCore
import copy
import pyqtgraph as pg
from pyqtgraph import QtCore, mkPen, mkColor, hsvColor
from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight, HorizontalLine
from hpm.widgets.PltWidget import CustomViewBox, PltWidget
from axd.widgets.aEDXD_custom_widgets import DoubleSlider

class aEDXDRoiWidget(QtWidgets.QWidget):

    color_btn_clicked = QtCore.pyqtSignal(int, QtWidgets.QWidget)
    
    show_cb_state_changed = QtCore.pyqtSignal(dict)
    name_item_changed = QtCore.pyqtSignal(int, str)
    widget_closed = QtCore.pyqtSignal()
    roi_selection_changed_signal = QtCore.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._layout = QtWidgets.QVBoxLayout()  
        self.setWindowTitle('Cut regions control')
        self.button_widget = QtWidgets.QWidget(self)
        self.button_widget.setObjectName('rois_control_button_widget')
        self._button_layout = QtWidgets.QHBoxLayout()
        self._button_layout.setContentsMargins(0, 0, 0, 0)
        self._button_layout.setSpacing(15)
        #self.add_btn = FlatButton('Add')
        #self.edit_btn = FlatButton('Edit')
        self.delete_btn = FlatButton('Delete')
        self.clear_btn = FlatButton('Clear')
        #self.show_fit_btn = FlatButton('Show')
        self.edit_btn = FlatButton('Edit')
        self.filter_btn = FlatButton(f'Filter 2\N{GREEK SMALL LETTER THETA}')
        self.filter_btn.setCheckable(True)
        self.filter_btn.setChecked(True)

        
        self._button_layout.addWidget(self.edit_btn)
        self._button_layout.addWidget(self.delete_btn)
        self._button_layout.addWidget(self.clear_btn)
        self._button_layout.addWidget(self.filter_btn)
        self._button_layout.addSpacerItem(HorizontalSpacerItem())

        self.button_widget.setLayout(self._button_layout)
        self._layout.addWidget(self.button_widget)
        self._body_layout = QtWidgets.QHBoxLayout()
        self.roi_tw = ListTableWidget(columns=3)
        self._body_layout.addWidget(self.roi_tw, 10)


####  Start peak parameters widget

        self.baseline_params = QtWidgets.QWidget()
        self.baseline_params_layout = QtWidgets.QGridLayout()

        self.baseline_params_V_layout = QtWidgets.QVBoxLayout()

        self.cut_peak_Wn = DoubleSpinBoxAlignRight()
        self.cut_peak_Wn.setMinimumWidth(90)
        self.cut_peak_Wn.setMinimum(0.001)
        self.cut_peak_Wn.setMaximum(1)
        self.cut_peak_Wn.setValue(0.2)
        self.cut_peak_Wn.setDecimals(3)
        #self.cut_peak_Wn.setSingleStep(0.1)

        
        self.cut_peak_iter = DoubleSpinBoxAlignRight()
        self.cut_peak_iter.setMinimumWidth(90)
        self.cut_peak_iter.setMinimum(1)
        self.cut_peak_iter.setMaximum(50)
        self.cut_peak_iter.setValue(50)
        self.cut_peak_iter.setDecimals(0)
        #self.cut_peak_iter.setSingleStep(5)

        self.cut_peak_Wn_step = DoubleMultiplySpinBoxAlignRight()
        self.cut_peak_Wn_step.setDecimals(3)
        self.cut_peak_Wn_step.setMinimumWidth(70)
        self.cut_peak_iter_step = DoubleMultiplySpinBoxAlignRight()
        self.cut_peak_iter_step.setDecimals(0)
        self.cut_peak_iter_step.setMinimumWidth(70)

        self.cut_peak_Wn_step.valueChanged.connect(partial(self.update_step, \
                                                    self.cut_peak_Wn,self.cut_peak_Wn_step))

        self.cut_peak_iter_step.valueChanged.connect(partial(self.update_step, \
                                                    self.cut_peak_iter,self.cut_peak_iter_step))

        self.cut_peak_Wn_step.setValue(0.1)
        self.cut_peak_iter_step.setValue(5)

        self.cut_peak_method_spline_choice = QtWidgets.QRadioButton(text="Spline")
        self.cut_peak_method_baseline_choice = QtWidgets.QRadioButton(text="Baseline")
        self.cut_peak_method_baseline_choice.setChecked(True)
        

        self.cut_peak_label = QtWidgets.QLabel("Peak cutting parameters")
        self.cut_peak_Wn_label = QtWidgets.QLabel("W<sub>n</sub>")
        self.cut_peak_iter_label = QtWidgets.QLabel("Iterations")
        self.cut_peak_method_label = QtWidgets.QLabel("Method")

        
        self.baseline_params_layout.addWidget(self.cut_peak_label,0,0,1,2)
        self.baseline_params_layout.addWidget(self.cut_peak_method_label,1,0)
        self.baseline_params_layout.addWidget(self.cut_peak_method_baseline_choice,1,1)
        self.baseline_params_layout.addWidget(self.cut_peak_method_spline_choice,1,2)

        self.baseline_params_layout.addWidget(self.cut_peak_Wn_label,2,0)
        self.baseline_params_layout.addWidget(self.cut_peak_iter_label,3,0)
        self.baseline_params_layout.addWidget(self.cut_peak_Wn,2,1)
        self.baseline_params_layout.addWidget(self.cut_peak_iter,3,1)
        self.baseline_params_layout.addWidget(self.cut_peak_Wn_step,2,2)
        self.baseline_params_layout.addWidget(self.cut_peak_iter_step,3,2)

        self.baseline_params_apply_all = QtWidgets.QCheckBox('Apply to all')
        self.baseline_params_apply_all.setChecked(False)
        #self.baseline_params_layout.addWidget(self.baseline_params_apply_all,3,0,1,2)
        

        self.baseline_params.setLayout(self.baseline_params_layout)
        self.baseline_params_V_layout.addWidget(self.baseline_params)
        self.baseline_params_V_layout.addSpacerItem(VerticalSpacerItem())

        # End peak parameters widget

        self._body_layout.addLayout(self.baseline_params_V_layout)



        self._layout.addLayout(self._body_layout)

        
        self.button_2_widget = QtWidgets.QWidget(self)
        self.button_2_widget.setObjectName('rois_control_button_2_widget')
        self._button_2_layout = QtWidgets.QHBoxLayout()
        self._button_2_layout.setContentsMargins(0, 0, 0, 0)
        self._button_2_layout.setSpacing(6)
        self.apply_btn = FlatButton('Apply')
        self._button_2_layout.addWidget(self.apply_btn,0)
        #self._button_2_layout.addWidget(VerticalLine())
        self._button_2_layout.addSpacerItem(HorizontalSpacerItem())
        #self._button_2_layout.addWidget(VerticalLine())
        self.button_2_widget.setLayout(self._button_2_layout)
        self._layout.addWidget(HorizontalLine())
        self._layout.addWidget(self.button_2_widget)
        

        


        self.setLayout(self._layout)

        self.style_widgets()
        self.roi_show_cbs = []
        self.name_items = []
        self.tth_items = []
       
        self.show_parameter_in_pattern = True
        header_view = QtWidgets.QHeaderView(QtCore.Qt.Orientation.Horizontal, self.roi_tw)
        self.roi_tw.setHorizontalHeader(header_view)
        header_view.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        
        self.default_header = [' Cut ', 'E range', f'2\N{GREEK SMALL LETTER THETA}']
        self.header = copy.deepcopy(self.default_header)
        self.roi_tw.setHorizontalHeaderLabels(self.header)
        #header_view.hide()
        self.roi_tw.setItemDelegate(NoRectDelegate())
        self.create_connections()


    def update_step(self, control, step_control):
        
        value = step_control.value()
        control.setSingleStep(value)


    def create_connections(self):
        self.roi_tw.currentCellChanged.connect(self.roi_selection_changed)

    

    def style_widgets(self):
        self.roi_tw.setSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        self.roi_tw.setMinimumWidth(370)
        self.roi_tw.setMinimumHeight(110)
        self.setStyleSheet("""
            #rois_control_button_widget FlatButton {
                max-width: 70;
                min-width: 70;
            }
            #rois_control_button_2_widget FlatButton {
                max-width: 70;
                min-width: 70;
            }
        """)

    def closeEvent(self, event):
        # Overrides close event to let controller know that widget was closed by user
        self.widget_closed.emit()

    


    ################################################################################################
    # Now comes all the roi tw stuff
    ################################################################################################

    def roi_selection_changed(self, row, **kwargs):
        tth = None
        name = None
        if len(self.tth_items):
            tth = float(self.tth_items[row].text())
            name = self.name_items[row].text()
            
        self.roi_selection_changed_signal.emit({'tth':tth,'name' :name})

    def set_tw_header_unit(self, unit, unit_=''):
        if unit_ !='':
            unit_=' ('+unit_+')'
        self.header[2] = self.default_header[2] + ', '+unit+unit_
        self.roi_tw.setHorizontalHeaderLabels(self.header)
    
    def filter_tth(self, fltr_tth):
        print(fltr_tth)

    def add_roi(self, use, name,tth):
        self.roi_tw.blockSignals(True)
        current_rows = self.roi_tw.rowCount()
        self.roi_tw.setRowCount(current_rows + 1)


        show_cb = QtWidgets.QCheckBox()
        show_cb.setChecked(use)
        show_cb.stateChanged.connect(partial(self.roi_show_cb_changed, show_cb))
        show_cb.setStyleSheet("background-color: transparent")
        self.roi_tw.setCellWidget(current_rows, 0, show_cb)
        self.roi_show_cbs.append(show_cb)
        

        name_item = QtWidgets.QTableWidgetItem(name)
        name_item.setText(name)
        #name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        name_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.roi_tw.setItem(current_rows, 1, name_item)
        self.name_items.append(name_item)

        tth_item = QtWidgets.QTableWidgetItem(tth)
        tth_item.setText(tth)
        #tth_item.setFlags(tth_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        tth_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.roi_tw.setItem(current_rows, 2, tth_item)
        self.tth_items.append(tth_item)

        self.roi_tw.setColumnWidth(0, 50)
        self.roi_tw.setRowHeight(current_rows, 25)
        self.roi_tw.blockSignals(False)

    
    
        
    def select_roi(self, ind):
        self.roi_tw.selectRow(ind)
        
    def get_selected_roi_row(self):
        tth = None
        name = None
        row = -1
        selected = self.roi_tw.selectionModel().selectedRows()
        if len(selected):
            row = selected[0].row()
            if len(self.tth_items):
                tth = float(self.tth_items[row].text())
                name = self.name_items[row].text()
                
        return {'tth':tth,'name' :name,'row':row}

    def get_selected_roi_name(self):
        selected = self.roi_tw.selectionModel().selectedRows()
        try:
            row = selected[0].row()
            name = self.name_items[row].text()
        except IndexError:
            row = -1
            name = ''
        return name

    def get_selected_roi_tth(self):
        selected = self.roi_tw.selectionModel().selectedRows()
        try:
            row = selected[0].row()
            tth_str = self.tth_items[row].text()
            tth = float(tth_str)
        except IndexError:
            row = -1
            tth = -1
        return tth

    

    def raise_widget(self):
        self.show()
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowState.WindowMinimized | QtCore.Qt.WindowState.WindowActive)
        self.activateWindow()
        self.raise_()

    def del_roi(self, ind):
        self.roi_tw.blockSignals(True)
        self.roi_tw.removeRow(ind)
        
        self.roi_tw.blockSignals(False)
        del self.roi_show_cbs[ind]
        del self.name_items[ind]
        del self.tth_items[ind]

    def rename_roi(self, ind, name):
        self.roi_tw.blockSignals(True)
        name_item = self.roi_tw.item(ind, 1)
        name_item.setText(name)
        self.roi_tw.blockSignals(False)

    def roi_show_cb_changed(self, checkbox):
        checked = checkbox.isChecked()
        ind = self.roi_show_cbs.index(checkbox)
        name=self.name_items[ind].text()
        tth=self.tth_items[ind].text()
        state = {'tth':tth, 'name':name, 'checked':checked}

        self.show_cb_state_changed.emit(state)

    
class plotFitWindow(QtWidgets.QWidget):
    widget_closed = QtCore.pyqtSignal()
    def __init__(self):
        super().__init__()

        self._layout = QtWidgets.QVBoxLayout()  
        self._layout.setContentsMargins(0,0,0,0)
        self.setWindowTitle('E cut view')
        self.fitPlots = PltWidget(self)
        self.fitPlots.setLogMode(False,False)
        self.fitPlots.setMenuEnabled(enableMenu=False)
        self.viewBox = self.fitPlots.getViewBox() # Get the ViewBox of the widget
        self.viewBox.setMouseMode(self.viewBox.RectMode)
        self.viewBox.enableAutoRange(0, False)
        
        self.fitPlot = self.fitPlots.plot([],[], 
                        pen=(155,155,155), name="Fit", 
                        antialias=True)
        self.fitPlot2 = self.fitPlots.plot([],[], 
                        pen=(100,100,255), name="Fit", 
                        antialias=True, width=2)
        self.dataPlot = self.fitPlots.plot([],[], 
                        pen = None, symbolBrush=(255,0,0), 
                        symbolPen='k', name="Data",antialias=True)
        self.vLineRight = pg.InfiniteLine(angle=90, movable=True,pen=mkPen({'color': mkColor(0,180,180,120), 'width': 2}))
        self.vLineLeft = pg.InfiniteLine(angle=90, movable=True,pen=mkPen({'color': mkColor(0,180,180,120), 'width': 2}))
        self.vLineRight.setPos(nan)
        self.vLineLeft.setPos(nan)
        self.viewBox.addItem(self.vLineRight, ignoreBounds=True)
        self.viewBox.addItem(self.vLineLeft, ignoreBounds=True)
        #self.fitPlots.setLabel('left','counts')
        #self.fitPlots.setLabel('bottom', 'channel')

        self._layout.addWidget(self.fitPlots)
        self.setLayout(self._layout)
        self.resize(800,500)

    def closeEvent(self, event):
        # Overrides close event to let controller know that widget was closed by user
        self.widget_closed.emit()

    def raise_widget(self):
        self.show()
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowState.WindowMinimized | QtCore.Qt.WindowState.WindowActive)
        self.activateWindow()
        self.raise_()

    def set_data(self,x_fit=[],y_fit=[],label='',x=[],y=[], x1=[],y1=[],unit='',unit_='', r = None, roi_range=None):
        self.fitPlot.setData(x_fit,y_fit) 
        self.fitPlots.setTitle(label)
        #self.dataPlot.setData(x,y)
        self.fitPlot2.setData(x1,y1)
        self.fitPlots.setLabel('bottom', unit+' ('+unit_+')')
        #self.fitPlots.getViewBox().enableAutoRange()
        if r != None:
            self.viewBox.enableAutoRange(0, False)
            self.viewBox.enableAutoRange(1, False)
            self.viewBox.setXRange(r[0][0], r[0][1], padding=0)
            self.viewBox.setYRange(r[1][0], r[1][1], padding=0)
        if roi_range != None:
            self.vLineRight.setPos(roi_range[0])
            self.vLineLeft.setPos(roi_range[1])
    

        
        