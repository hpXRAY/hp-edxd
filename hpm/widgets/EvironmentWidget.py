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
from PyQt6 import QtWidgets, QtCore
import copy
import pyqtgraph as pg
from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight


class EnvironmentWidget(QtWidgets.QWidget):

    color_btn_clicked = QtCore.pyqtSignal(int, QtWidgets.QWidget)
    #env_btn_clicked = QtCore.pyqtSignal(int)
    show_cb_state_changed = QtCore.pyqtSignal(int, int)
    pv_item_changed = QtCore.pyqtSignal(int, str)
    widget_closed = QtCore.pyqtSignal()
    key_signal = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self._layout = QtWidgets.QVBoxLayout()  
        self.setWindowTitle('Environment view')
        '''self.button_widget = QtWidgets.QWidget(self)
        self.button_widget.setObjectName('environment_control_button_widget')
        self._button_layout = QtWidgets.QHBoxLayout()
        self._button_layout.setContentsMargins(0, 0, 0, 0)
        self._button_layout.setSpacing(6)
        #self.add_btn = FlatButton('Add')
        #self.edit_btn = FlatButton('Edit')
        self.delete_btn = FlatButton('Delete')
        self.clear_btn = FlatButton('Clear')
        self.show_fit_btn = FlatButton('Show fit')
        self.pressure_btn = FlatButton('Pressure')
        self.save_peaks_btn = FlatButton('Save List')

        #self._button_layout.addWidget(self.add_btn)
        #self._button_layout.addWidget(self.edit_btn)
        self._button_layout.addWidget(self.delete_btn)
        self._button_layout.addWidget(self.clear_btn)
        self._button_layout.addWidget(self.show_fit_btn)
        #self._button_layout.addWidget(self.pressure_btn)
        self._button_layout.addWidget(self.save_peaks_btn)

        self.button_widget.setLayout(self._button_layout)'''
        #self._layout.addWidget(self.button_widget)
        self._body_layout = QtWidgets.QHBoxLayout()
        self.env_tw = ListTableWidget(columns=2)
        self._body_layout.addWidget(self.env_tw, 10)
        self._layout.addLayout(self._body_layout)

        

        self.setLayout(self._layout)

        self.style_widgets()
        self.env_show_cbs = []
        self.pv_items = []
        self.index_items = []
        #self.env_color_btns = []
        self.show_parameter_in_pattern = True
        header_view = QtWidgets.QHeaderView(QtCore.Qt.Orientation.Horizontal, self.env_tw)
        self.env_tw.setHorizontalHeader(header_view)
        header_view.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        
        header_view.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.default_header = [ 'Variable', 'Value']
        self.header = copy.deepcopy(self.default_header)
        self.env_tw.setHorizontalHeaderLabels(self.header)
        #header_view.hide()
        self.env_tw.setItemDelegate(NoRectDelegate())
        self.env_tw.itemChanged.connect(self.env_pv_item_changed)

    def keyPressEvent(self, e):
        sig = None
        if e.key() == Qt.Key.Key_Up:
            sig = 'up'
        if e.key() == Qt.Key.Key_Down:
            sig = 'down'                
        if e.key() == Qt.Key.Key_Delete:
            sig = 'delete'
        if e.key() == Qt.Key.Key_Right:
            sig = 'right'   
        if e.key() == Qt.Key.Key_Left:
            sig = 'left'   
        if e.key() == Qt.Key.Key_Backspace:
            sig = 'delete'  
        if sig is not None:
            self.key_signal.emit(sig)
        else:
            super().keyPressEvent(e)

    

    def style_widgets(self):
        self.env_tw.setSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        self.env_tw.setMinimumWidth(280)
        self.env_tw.setMinimumHeight(110)
        self.setStyleSheet("""
            #env_control_button_widget FlatButton {
                min-width: 90;
            }
        """)


    ################################################################################################
    # Now comes all the env tw stuff
    ################################################################################################

    def add_env(self,  pv, value, description,  silent=False):
        self.env_tw.blockSignals(True)
        current_rows = self.env_tw.rowCount()
        self.env_tw.setRowCount(current_rows + 1)


        '''pv_item = QtWidgets.QTableWidgetItem(pv)
        pv_item.setText(pv)
        pv_item.setFlags(pv_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        pv_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.env_tw.setItem(current_rows, 0, pv_item)
        self.pv_items.append(pv_item)'''

        description_item = QtWidgets.QTableWidgetItem(description)
        description_item.setFlags(description_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        description_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.env_tw.setItem(current_rows, 0, description_item)

        value_item = QtWidgets.QTableWidgetItem(value)
        value_item.setFlags(value_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        value_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.env_tw.setItem(current_rows, 1, value_item)

   
        
        if not silent:
            self.select_env(current_rows)
            self.env_tw.blockSignals(False)

    def closeEvent(self, event):
        # Overrides close event to let controller know that widget was closed by user
        self.widget_closed.emit()
        
    def select_env(self, ind):
        self.env_tw.selectRow(ind)
        
    def get_selected_env_row(self):
        selected = self.env_tw.selectionModel().selectedRows()
        try:
            row = selected[0].row()
        except IndexError:
            row = -1
        return row

    def raise_widget(self):
        self.show()
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowState.WindowMinimized | QtCore.Qt.WindowState.WindowActive)
        self.activateWindow()
        self.raise_()

    def del_env(self, ind, silent=False):
        self.env_tw.blockSignals(True)
        self.env_tw.removeRow(ind)
        if not silent:
            self.env_tw.blockSignals(False)
        #del self.env_show_cbs[ind]
        
        #del self.env_color_btns[ind]
        if not silent:
            if self.env_tw.rowCount() > ind:
                self.select_env(ind)
            else:
                self.select_env(self.env_tw.rowCount() - 1)

    def rename_env(self, ind, name):
        self.env_tw.blockSignals(True)
        pv_item = self.env_tw.item(ind, 1)
        pv_item.setText(name)
        self.env_tw.blockSignals(False)

    def update_env(self, pv, value, description, ind):
        self.env_tw.blockSignals(True)
        #index_item = self.env_tw.item(ind, 0)
        #index_item.setText(str(ind))

        #show_cb = self.env_show_cbs[ind]
        #show_cb.setChecked(use)
        #pv_item = self.env_tw.item(ind, 1)
        #pv_item.setText(pv)
        description_item = self.env_tw.item(ind, 0)
        description_item.setText(description)
        value_item = self.env_tw.item(ind, 1)
        value_item.setText(str(value))
      
        self.env_tw.blockSignals(False)

    def env_show_cb_changed(self, checkbox):
        checked = checkbox.isChecked()
        if checked: state = 1
        else: state = 0
        self.show_cb_state_changed.emit(self.env_show_cbs.index(checkbox), state)

    def env_pv_item_changed(self, nameitem):
        
        selected = self.get_selected_env_row()
        self.pv_item_changed.emit(selected, self.pv_items[selected].text())
  


