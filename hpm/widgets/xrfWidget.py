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

import hpm.models.Xrf as Xrf
from hpm.models.Xrf_model import atom
from functools import partial
import time
import copy
from PyQt6 import QtWidgets, QtCore
import numpy as np
from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight

class xrfWidget(QtWidgets.QWidget):

    
    xrf_selection_updated_signal = QtCore.pyqtSignal(str ) 
    color_btn_clicked = QtCore.pyqtSignal(int, QtWidgets.QWidget)
    xrf_btn_clicked = QtCore.pyqtSignal(int)
    show_cb_state_changed = QtCore.pyqtSignal(int, bool)
    name_item_changed = QtCore.pyqtSignal(int, str)
    widget_closed = QtCore.pyqtSignal()

    def __init__(self, plotWidget, plotContoller, roiController, mca):
        super().__init__()

        self.set_mca(mca) # in future change to get cal. from  plot model
        
        self.x_scale = 'E'
        
        
        self.pattern_widget = plotWidget
        self.roi_controller = roiController
        self.plotController = plotContoller
        self.log_scale = self.plotController.logMode
        self._layout = QtWidgets.QVBoxLayout()  
        self.setWindowTitle('XRF control')
        self.button_widget = QtWidgets.QWidget(self)
        self.button_widget.setObjectName('xrfs_control_button_widget')
        self._button_layout = QtWidgets.QHBoxLayout()
        self._button_layout.setContentsMargins(0, 0, 0, 0)
        self._button_layout.setSpacing(6)
        self.add_btn = QtWidgets.QPushButton('Add ROIs')
        self.add_btn.clicked.connect(self.rois_btn_click_callback)

        self.clear_btn = QtWidgets.QPushButton('Clear')
        self.clear_btn.clicked.connect(self.clear_xrf)
        self._button_layout.addWidget(QtWidgets.QLabel('Search:'),0)
        self.search_item = s =  QtWidgets.QLineEdit('')
        s.textChanged.connect(lambda:self.search_by_symbol(s.text()))
        self._button_layout.addWidget(self.search_item, 0)
        self._button_layout.addSpacerItem(HorizontalSpacerItem())
        self._button_layout.addWidget(self.add_btn,0)
        self._button_layout.addWidget(self.clear_btn,0)
        self.button_widget.setLayout(self._button_layout)
        self._layout.addWidget(self.button_widget)
        self._body_layout = QtWidgets.QHBoxLayout()
        self.header = ['Z', 'Symbol']
        self.cols = len(self.header)
        self.xrf_tw = ListTableWidget(columns=self.cols)
        self._body_layout.addWidget(self.xrf_tw)

        self.k_lines_tags = ['Ka','Ka1', 'Ka2','Kb', 'Kb1', 'Kb2']
        self.l_lines_tags = ['La1', 'Lb1', 'Lb2', 'Lg1', 'Lg2', 'Lg3', 'Lg4', 'Ll']
        self.g_lines_tags = ['G1','G2', 'G3']

        self.parameter_widget = QtWidgets.QWidget()
        self._parameter_layout = QtWidgets.QGridLayout()

        self.k_cbs = dict()
        for k in self.k_lines_tags:
            k_line = QtWidgets.QCheckBox(k)
            k_line.setEnabled(False)
            k_line.toggled.connect(partial(self.cbs_changed, k_line))
            self._parameter_layout.addWidget(k_line, 0, self.k_lines_tags.index(k)+2)
            self.k_cbs[k]=k_line
        
        self.l_cbs = dict()
        for l in self.l_lines_tags:
            l_line = QtWidgets.QCheckBox(l)
            l_line.setEnabled(False)
            l_line.toggled.connect(partial(self.cbs_changed, l_line))
            self._parameter_layout.addWidget(l_line,1, self.l_lines_tags.index(l)+2)
            self.l_cbs[l]=l_line

        self.g_cbs = dict()
        for g in self.g_lines_tags:
            g_line = QtWidgets.QCheckBox(g)
            g_line.setEnabled(False)
            g_line.toggled.connect(partial(self.cbs_changed, g_line))
            self._parameter_layout.addWidget(g_line,2, self.g_lines_tags.index(g)+2)
            self.g_cbs[g]=g_line
        
        self.parameter_widget.setLayout(self._parameter_layout)
        self._body_layout.addWidget(self.parameter_widget, 0)
        self._layout.addLayout(self._body_layout)
        self.setLayout(self._layout)
        self.style_widgets()
        self.xrf_show_cbs = []
        self.name_items = []
        self.index_items = []
        self.kl_cbs = dict()
        self.kl_cbs.update(self.k_cbs) 
        self.kl_cbs.update(self.l_cbs)
        self.kl_cbs.update(self.g_cbs)
        self.xrf = []

        self.atoms = dict()

        #self.show_parameter_in_pattern = True
        header_view = QtWidgets.QHeaderView(QtCore.Qt.Orientation.Horizontal, self.xrf_tw)
        self.xrf_tw.setHorizontalHeader(header_view)

        header_view.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)  
        
        self.xrf_tw.setHorizontalHeaderLabels(self.header)
        #header_view.hide()
        self.xrf_tw.setItemDelegate(NoRectDelegate())
        #self.xrf_tw.itemChanged.connect(self.xrf_name_item_changed)

        for atom in Xrf.atomic_symbols:
            self.add_xrf_row(Xrf.atomic_number(atom),atom)
        #self.xrf_tw.selectRow(Xrf.atomic_symbols.index(None))

        color = self.add_xrf_plot()
        self.xrf_tw.currentCellChanged.connect(self.selection_changed)
        #self.xrf_tw.setColumnWidth(0, 35)

        self.plotController.unitUpdated.connect(self.update_x_scale)
        self.plotController.dataPlotUpdated.connect(self.update_all_xrf_lines)

    def set_mca(self, mca, element=0):
        self.mca = mca
        
        calibrations = self.mca.get_calibration()
        if not element < len(calibrations):
            element = 0
        self.calibration = calibrations[element]
        
    def update_x_scale(self,unit):
        self.x_scale = unit
        self.update_xrf_lines (self.get_current_symbol())

    def set_log_mode(self, log_scale):
        self.log_scale = log_scale 

    def style_widgets(self):
        self.xrf_tw.setSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        self.xrf_tw.setMinimumWidth(200)
        self.xrf_tw.setMinimumHeight(110)

        self.setStyleSheet("""
            #xrfs_control_button_widget QPushButton {
                min-width: 95;
            }
        """)
        
        #self.xrf_tw.setShowGrid(True)
        #self.xrf_tw.setStyleSheet("gridline-color: #hhhhhh")



    ################################################################################################
    # Now comes all the xrf tw stuff
    ################################################################################################

    def navigate_btn_click_callback(self, action=''):
        if action == 'next':
            move = 1
        elif action == 'prev':
            move = -1
        else:
            move = 0
        curr_ind = self.get_selected_xrf_row()
        new_ind = curr_ind + move
        
        if new_ind >=0 :
            self.select_xrf(new_ind)



    def rois_btn_click_callback(self):
        cur_ind = self.get_selected_xrf_row()

        if cur_ind >=0:
            self.xrf_roi_btn_clicked(cur_ind)

    def xrf_roi_btn_clicked(self, index):
        symbol = Xrf.atomic_symbol( index+1)
        my_atom = self.get_stored_atom(symbol)
        xrf = dict()
        for cb in self.kl_cbs:
            line = self.kl_cbs[cb].text()
            e = my_atom.get_e(line)
            checked = my_atom.is_line_checked(line)
            if e != 0 and checked:
                xrf[symbol+' '+line] = e
        rois = []
        for line in xrf:
            lbl = line
            E = xrf[line]
            fwhm = int(round(0.07936 * E + 4.95677))

            rois.append([xrf[line],fwhm, lbl])
        self.roi_controller.addROISbyE(rois)

    def selection_changed(self, row, col, prev_row, prev_col):
        self.update_row(row)

    def update_row(self,row):
        symbol = Xrf.atomic_symbol( row+1)
        if symbol is not None:

            my_atom = self.get_stored_atom(symbol)

            for cb in self.kl_cbs:
                line = self.kl_cbs[cb].text()
                e = my_atom.get_e(line)
                if e != 0:
                    self.kl_cbs[cb].setEnabled(True)
                    checked = my_atom.is_line_checked(line)
                    self.kl_cbs[cb].setChecked(checked)
                    self.kl_cbs[cb].setStyleSheet("")
                else:
                    self.kl_cbs[cb].setEnabled(False)
                    self.kl_cbs[cb].setChecked(False)
                    self.kl_cbs[cb].setStyleSheet("color: rgb(100, 100, 100); ") 
            self.show_state_changed(0,True)
            self.update_xrf_lines(symbol)
            self.xrf_selection_updated_signal.emit(symbol)
            self.blockSignals(True)
            self.search_item.setText(symbol)
            self.blockSignals(False)
        

    def search_by_symbol(self, symbol):
        
        ind = Xrf.atomic_index(symbol)
        if ind is not None:
            self.show_state_changed(0,True)
            self.xrf_tw.selectRow(ind-1)
        else:
            self.clear_xrf()
        
    
    def get_stored_atom(self,name):
        my_atom = None
        for a in self.atoms:
            if name.upper() == a.upper(): my_atom = self.atoms[a]
        if my_atom == None:
            my_atom = atom(name)
            self.atoms.update({name:my_atom})
        return my_atom

    def add_xrf_row(self, Z, symbol):
        current_rows = self.xrf_tw.rowCount()
        self.xrf_tw.setRowCount(current_rows + 1)
        self.xrf_tw.blockSignals(True)
        
        z = QtWidgets.QTableWidgetItem(str(Z))
        z.setFlags(z.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        z.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.xrf_tw.setItem(current_rows, 0, z)
        
        symbol_item= QtWidgets.QTableWidgetItem(symbol)
        symbol_item.setFlags(symbol_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        symbol_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.xrf_tw.setItem(current_rows, 1, symbol_item)
        
        self.xrf_tw.blockSignals(False)

    def closeEvent(self, event):
        # Overrides close event to let controller know that widget was closed by user
        self.widget_closed.emit()
        

    def select_xrf(self, ind):
        self.xrf_tw.selectRow(ind)
        

    def get_selected_xrf_row(self):
        selected = self.xrf_tw.selectionModel().selectedRows()
        try:
            row = selected[0].row()
        except IndexError:
            row = -1
        return row

    def raise_widget(self):
        self.show()
        #self.setWindowState(self.windowState() & ~QtCore.Qt.WindowState.WindowMinimized | QtCore.Qt.WindowState.WindowActive)
        self.activateWindow()
        self.raise_()

    def update_cbs(self, atom):
        for m in self.kl_cbs:
            if atom.get_e(m.text()) != None:
                m.setChecked(atom.is_line_checked(m))
            
    def cbs_changed(self, cb):
        curr_row = self.xrf_tw.currentRow()
        symbol = Xrf.atomic_symbol(curr_row+1)
        cur_atom = self.get_stored_atom(symbol)
        line = cb.text()
        cb_item = self.kl_cbs[line]
        checked = cb.isChecked()
        cur_atom.set_line_checked(line,checked)
        self.update_row(curr_row)

    def xrf_show_cb_changed(self, checkbox):
        #self.show_cb_state_changed.emit(self.xrf_show_cbs.index(checkbox), checkbox.isChecked())
        symbol = Xrf.atomic_symbol(self.xrf_show_cbs.index(checkbox)+1)
        cur_atom = self.get_stored_atom(symbol)
        cur_atom.show = checkbox.isChecked()
        
        
    def get_current_symbol (self):
        curr_row = self.xrf_tw.currentRow()
        if curr_row >=0:
            symbol = Xrf.atomic_symbol(curr_row+1)
            return symbol
        return None

    def add_xrf_plot(self):
        """
        Adds a xrf to the Pattern view.
        :return:
        """
        axis_range = self.plotController.getRange()
        x_range = axis_range[0]
        y_range = axis_range[1]
        y_max = np.amax(y_range)
        symbol = ''
        positions = []
        intensities = []
        baseline = 1
        xrf_line_items = dict() # {tag: [position, intensity, index]}

        for tag in self.k_lines_tags+self.l_lines_tags+self.g_lines_tags:
            xrf_line_items.update({tag:[0,1,len(positions)]}) 
            positions.append(0)
            intensities.append(1)

        self.xrf.append(xrf_line_items)
        color = self.pattern_widget.add_xrf(symbol,
                                              positions,
                                              intensities,
                                              baseline)
    
        return color

    def show_state_changed(self, ind, state):
        if state:
            self.pattern_widget.show_xrf(ind)
        else:
            self.pattern_widget.hide_xrf(ind)
        pass

    def clear_xrf(self):
        self.show_state_changed(0,False)

    def update_all_xrf_lines(self):
        self.update_xrf_lines (self.get_current_symbol())
    

    def update_xrf_lines(self, symbol, ind=0, axis_range=None):
        """
        
        """
        if symbol is not None:
            if axis_range is None:
                axis_range = self.plotController.getRange()

            x_range = axis_range[0]
            y_range = axis_range[1]
            x_min = np.amin(x_range)
            x_max = np.amax(x_range)
            y_max = np.amax(y_range)
            cur_atom = self.get_stored_atom(symbol)
            positions = []
            intensities = []
            lines = dict()
            for tag in self.k_lines_tags+self.l_lines_tags + self.g_lines_tags:
                checked = cur_atom.is_line_checked(tag)
                e = cur_atom.get_e(tag)
                if checked:
                    intensity = y_max
                else: 
                    intensity = 0
                    e = 0
                e = self.scale_position(e)
                if not (e > x_min and e < x_max):
                    intensity = 0
                    e = 0
                lines[tag]=[e,intensity]
                positions.append(e)
                intensities.append(intensity)

            self.pattern_widget.update_xrf_intensities(
                ind, positions, intensities, .5)
            line_str = ''
            for tag in lines:
                if lines[tag][0]>0 and lines[tag][0] >0:
                    line_str = line_str + ' ' + tag + ','
            if len(line_str):

                line_str = symbol + ": " + line_str
                line_str = line_str[:-1]
            
            self.pattern_widget.rename_xrf(0,line_str)

    def scale_position(self, value):
        
        if self.x_scale == 'E':
            return value
        else:
            if self.x_scale == 'Channel':
                return np.nan
            channel = self.calibration.energy_to_channel(value)
            if self.x_scale == 'd':
                return self.calibration.channel_to_d(channel)
            elif self.x_scale == 'q':
                return self.calibration.channel_to_q(channel)
            else:
                return np.nan