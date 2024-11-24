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

# Based on code from Dioptas - GUI program for fast processing of 2D X-ray diffraction data

""" Modifications:
    October 9, 2018 Ross Hrubiak
        - modified for use with MCA
        - support for energy dispersive mode
        - added 2th control
        - removed references to integration, pattern, calibration and possibly other stuff)
"""

from functools import partial

from PyQt6 import QtWidgets, QtCore

from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight, HorizontalLine

class PhaseWidget(QtWidgets.QWidget):

    color_btn_clicked = QtCore.pyqtSignal(int, QtWidgets.QWidget)
    
    show_cb_state_changed = QtCore.pyqtSignal(int, bool)
    file_dragged_in = QtCore.pyqtSignal(list)

    pressure_sb_value_changed = QtCore.pyqtSignal(int, float)
    temperature_sb_value_changed = QtCore.pyqtSignal(int, float)

    def __init__(self):
        super(PhaseWidget, self).__init__()
        
        self._layout = QtWidgets.QVBoxLayout()  
        self.setWindowTitle('Phase control')
        self.button_widget = QtWidgets.QWidget(self)
        self.button_widget.setObjectName('phase_control_button_widget')
        self._button_layout = QtWidgets.QHBoxLayout()
        self._button_layout.setContentsMargins(0, 0, 0, 0)
        self._button_layout.setSpacing(6)

        self.add_btn = QtWidgets.QPushButton('Add')
        self.edit_btn = QtWidgets.QPushButton('Edit')
        self.delete_btn = QtWidgets.QPushButton('Delete')
        self.clear_btn = QtWidgets.QPushButton('Clear')
        self.rois_btn = QtWidgets.QPushButton('Set ROIs')
        self.save_list_btn = QtWidgets.QPushButton('Save List')
        self.load_list_btn = QtWidgets.QPushButton('Load List')

        self._button_layout.addWidget(self.add_btn,0)
        self._button_layout.addWidget(self.edit_btn,0)
        self._button_layout.addWidget(self.delete_btn,0)
        self._button_layout.addWidget(self.clear_btn,0)
        self._button_layout.addWidget(self.rois_btn,0)
       
        self._button_layout.addSpacerItem(HorizontalSpacerItem())
        self._button_layout.addWidget(VerticalLine())
        self._button_layout.addWidget(self.save_list_btn,0)
        self._button_layout.addWidget(self.load_list_btn,0)
        self.button_widget.setLayout(self._button_layout)
        self._layout.addWidget(self.button_widget)

        self.parameter_widget = QtWidgets.QWidget()

        self._parameter_layout = QtWidgets.QGridLayout()
        self.pressure_sb = DoubleSpinBoxAlignRight()
        self.temperature_sb = DoubleSpinBoxAlignRight()
        self.pressure_step_msb = DoubleMultiplySpinBoxAlignRight()
        self.temperature_step_msb = DoubleMultiplySpinBoxAlignRight()
        self.apply_to_all_cb = QtWidgets.QCheckBox('Apply to all phases')
        self.show_in_pattern_cb = QtWidgets.QCheckBox('Show in Pattern')

        self._tth_lbl = QtWidgets.QLabel(u'2θ:')
        self._tth_unit_lbl = QtWidgets.QLabel('deg')
        self.tth_lbl = DoubleSpinBoxAlignRight()
        self.tth_step = DoubleMultiplySpinBoxAlignRight()
        self.get_tth_btn = QtWidgets.QPushButton('Get')
        self.auto_2theta_btn = QtWidgets.QPushButton('Auto')

        self._wavelength_lbl = QtWidgets.QLabel(f'\N{GREEK SMALL LETTER LAMDA}:')
        self._wavelength_unit_lbl = QtWidgets.QLabel(f'\N{LATIN CAPITAL LETTER A WITH RING ABOVE}')
        self.wavelength_lbl = DoubleSpinBoxAlignRight()
        self.wavelength_step = DoubleMultiplySpinBoxAlignRight()
        self.get_wavelength_btn = QtWidgets.QPushButton('Get')


        self.edx_widgets= [self._tth_lbl, self._tth_unit_lbl, self.tth_lbl,self.tth_step,self.get_tth_btn, self.auto_2theta_btn]
        self.adx_widgets= [self._wavelength_lbl, self._wavelength_unit_lbl, self.wavelength_lbl,self.wavelength_step,self.get_wavelength_btn]

        self._parameter_layout.addWidget(QtWidgets.QLabel('Parameter'), 0, 1)
        self._parameter_layout.addWidget(QtWidgets.QLabel('Step'), 0, 3)
        self._parameter_layout.addWidget(QtWidgets.QLabel('P:'), 1, 0)
        self._parameter_layout.addWidget(QtWidgets.QLabel('T:'), 2, 0)
        self._parameter_layout.addWidget(QtWidgets.QLabel('GPa'), 1, 2)
        self._parameter_layout.addWidget(QtWidgets.QLabel('K'), 2, 2)

        self._parameter_layout.addWidget(self.pressure_sb, 1, 1)
        self._parameter_layout.addWidget(self.pressure_step_msb, 1, 3)
        self._parameter_layout.addWidget(self.temperature_sb, 2, 1)
        self._parameter_layout.addWidget(self.temperature_step_msb, 2, 3)

        self._parameter_layout.addWidget(self.apply_to_all_cb, 3, 0, 1, 5)
        
        self._parameter_layout.addItem(VerticalSpacerItem(), 6, 0)
        self._parameter_layout.addWidget(HorizontalLine(),7,0,1,6)

        self._parameter_layout.addWidget(self._tth_lbl, 8, 0)
        self._parameter_layout.addWidget(self.tth_lbl, 8, 1)
        self._parameter_layout.addWidget(self._tth_unit_lbl, 8, 2)
        self._parameter_layout.addWidget(self.tth_step, 8, 3)
        self._parameter_layout.addWidget(self.get_tth_btn, 8, 4)

        self._parameter_layout.addWidget(self._wavelength_lbl, 9, 0)
        self._parameter_layout.addWidget(self.wavelength_lbl, 9, 1)
        self._parameter_layout.addWidget(self._wavelength_unit_lbl, 9, 2)
        self._parameter_layout.addWidget(self.wavelength_step, 9, 3)
        self._parameter_layout.addWidget(self.get_wavelength_btn, 9, 4)

        
        self.auto_2theta_btn.setCheckable(True)

        self._parameter_layout.addWidget(self.auto_2theta_btn, 8, 5)
        
        self.parameter_widget.setLayout(self._parameter_layout)

        self._body_layout = QtWidgets.QHBoxLayout()
        self.phase_tw = ListTableWidget(columns=5)
        self._body_layout.addWidget(self.phase_tw )
        self._body_layout.addWidget(self.parameter_widget, 0)

        self._layout.addLayout(self._body_layout)

        self.setLayout(self._layout)
        
        self.style_widgets()

        self.phase_show_cbs = []
        self.phase_color_btns = []
        #self.phase_roi_btns = [] #add ROIs (RH)
        self.show_parameter_in_pattern = True
        header_view = QtWidgets.QHeaderView(QtCore.Qt.Orientation.Horizontal, self.phase_tw)
        self.phase_tw.setHorizontalHeader(header_view)
        
        #header_view.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header_view.hide()
        self.phase_tw.setItemDelegate(NoRectDelegate())

        self.pressure_sb.valueChanged.connect(self.pressure_sb_changed)
        self.temperature_sb.valueChanged.connect(self.temperature_sb_changed)

     
        self.setAcceptDrops(True) 
    
    def set_edx(self):
        on = self.edx_widgets
        off = self.adx_widgets
        
        self.hide_widgets(off)
        self.show_widgets(on)

    def set_adx(self):
        off = self.edx_widgets
        on = self.adx_widgets
        
        self.hide_widgets(off)
        self.show_widgets(on)

    def show_widgets(self, widgets):
        for w in widgets:
            w.show()

    def hide_widgets(self, widgets):
        for w in widgets:
            w.hide()

    def pressure_sb_changed(self):
        cur_ind = self.get_selected_phase_row()
        pressure = self.pressure_sb.value()
        self.pressure_sb_value_changed.emit(cur_ind, pressure)

    def temperature_sb_changed(self):
        cur_ind = self.get_selected_phase_row()
        temperature = self.temperature_sb.value()
        self.temperature_sb_value_changed.emit(cur_ind, temperature)

    def style_widgets(self):
        self.phase_tw.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        self.parameter_widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        self.phase_tw.setMinimumHeight(120)

        self.temperature_step_msb.setMaximumWidth(75)
        self.pressure_step_msb.setMaximumWidth(75)
        self.tth_step.setMaximumWidth(75)
        self.get_tth_btn.setMaximumWidth(75)
        self.wavelength_step.setMaximumWidth(75)
        self.get_wavelength_btn.setMaximumWidth(75)
        
        self.pressure_sb.setMinimumWidth(80)

        self.pressure_sb.setMaximum(9999999)
        self.pressure_sb.setMinimum(-9999999)
        self.pressure_sb.setValue(0)

        self.pressure_step_msb.setMaximum(1000.0)
        self.pressure_step_msb.setMinimum(0.01)
        self.pressure_step_msb.setValue(0.2)

        self.temperature_sb.setMaximum(99999999)
        self.temperature_sb.setMinimum(0)
        self.temperature_sb.setValue(298)

        self.temperature_step_msb.setMaximum(1000.0)
        self.temperature_step_msb.setMinimum(1.0)
        self.temperature_step_msb.setValue(50.0)

        self.tth_lbl.setMaximum(179.0)
        self.tth_lbl.setMinimum(1)
        self.tth_lbl.setDecimals(5)
        self.tth_step.setMaximum(180)
        self.tth_step.setMinimum(0.001)
        self.tth_step.setValue(1)
        self.tth_step.setDecimals(3)

        self.wavelength_lbl.setMaximum(10.0)
        self.wavelength_lbl.setMinimum(0.1)
        self.wavelength_lbl.setValue(0.4)
        self.wavelength_lbl.setDecimals(5)
        self.wavelength_lbl.setSingleStep(0.01)
        self.wavelength_step.setMaximum(180)
        self.wavelength_step.setMinimum(0.0001)
        self.wavelength_step.setValue(.01)
        self.wavelength_step.setDecimals(5)

        self.setStyleSheet("""
            #phase_control_button_widget QPushButton {
                min-width: 70;
            }
        """)

        self.apply_to_all_cb.setChecked(True)
        self.show_in_pattern_cb.setChecked(True)

    # ###############################################################################################
    # Now comes all the phase tw stuff
    ################################################################################################

    def add_phase(self, name, color):
        self.phase_tw.blockSignals(True)
        current_rows = self.phase_tw.rowCount()
        self.phase_tw.setRowCount(current_rows + 1)
        

        show_cb = QtWidgets.QCheckBox()
        show_cb.setChecked(True)
        show_cb.stateChanged.connect(partial(self.phase_show_cb_changed, show_cb))
        show_cb.setStyleSheet("background-color: transparent")
        self.phase_tw.setCellWidget(current_rows, 0, show_cb)
        self.phase_show_cbs.append(show_cb)

        

        color_button = FlatButton()
        color_button.setStyleSheet("background-color: " + color)
        color_button.clicked.connect(partial(self.phase_color_btn_click, color_button))
        self.phase_tw.setCellWidget(current_rows, 1, color_button)
        self.phase_color_btns.append(color_button)

        name_item = QtWidgets.QTableWidgetItem(name)
        name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        name_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.phase_tw.setItem(current_rows, 2, name_item)

        pressure_item = QtWidgets.QTableWidgetItem('0 GPa')
        pressure_item.setFlags(pressure_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        pressure_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.phase_tw.setItem(current_rows, 3, pressure_item)

        temperature_item = QtWidgets.QTableWidgetItem('298 K')
        temperature_item.setFlags(temperature_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        temperature_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.phase_tw.setItem(current_rows, 4, temperature_item)

        self.phase_tw.setColumnWidth(0, 35)
        self.phase_tw.setColumnWidth(1, 25)
        self.phase_tw.setRowHeight(current_rows, 25)
        self.select_phase(current_rows)
        self.phase_tw.blockSignals(False)



    def select_phase(self, ind):
        self.phase_tw.selectRow(ind)

    def get_selected_phase_row(self):
        selected = self.phase_tw.selectionModel().selectedRows()
        try:
            row = selected[0].row()
        except IndexError:
            row = -1
        return row

    def get_phase(self):
        pass

    def del_phase(self, ind):
        self.phase_tw.blockSignals(True)
        self.phase_tw.removeRow(ind)
        self.phase_tw.blockSignals(False)
        del self.phase_show_cbs[ind]
        del self.phase_color_btns[ind]

        if self.phase_tw.rowCount() > ind:
            self.select_phase(ind)
        else:
            self.select_phase(self.phase_tw.rowCount() - 1)
    
    def rename_phase(self, ind, name):
        name_item = self.phase_tw.item(ind, 2)
        name_item.setText(name)

    def set_phase_temperature(self, ind, T):
        temperature_item = self.phase_tw.item(ind, 4)
        try:
            temperature_item.setText("{0:.2f} K".format(T))
        except ValueError:
            temperature_item.setText("{0} K".format(T))

    def get_phase_temperature(self, ind):
        temperature_item = self.phase_tw.item(ind, 4)
        try:
            temperature = float(str(temperature_item.text()).split()[0])
        except:
            temperature = None
        return temperature

    def set_phase_pressure(self, ind, P):
        pressure_item = self.phase_tw.item(ind, 3)
        try:
            pressure_item.setText("{0:.2f} GPa".format(P))
        except ValueError:
            pressure_item.setText("{0} GPa".format(P))

    def get_phase_pressure(self, ind):
        pressure_item = self.phase_tw.item(ind, 3)
        pressure = float(str(pressure_item.text()).split()[0])
        return pressure

    def phase_color_btn_click(self, button):
        self.color_btn_clicked.emit(self.phase_color_btns.index(button), button)
        
    def phase_show_cb_changed(self, checkbox):
        self.show_cb_state_changed.emit(self.phase_show_cbs.index(checkbox), checkbox.isChecked())

    def phase_show_cb_set_checked(self, ind, state):
        checkbox = self.phase_show_cbs[ind]
        checkbox.setChecked(state)

    def phase_show_cb_is_checked(self, ind):
        checkbox = self.phase_show_cbs[ind]
        return checkbox.isChecked()

    def raise_widget(self):
        self.show()
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowState.WindowMinimized | QtCore.Qt.WindowState.WindowActive)
        self.activateWindow()
        self.raise_()


        ########################################################################################

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        """
        Drop files directly onto the widget

        File locations are stored in fname
        :param e:
        :return:
        """
        if e.mimeData().hasUrls:
            e.setDropAction(QtCore.Qt.CopyAction)
            e.accept()
            fnames = list()
            for url in e.mimeData().urls():
                fname = str(url.toLocalFile())  
                fnames.append(fname)
            self.file_dragged_in.emit(fnames)
        else:
            e.ignore() 


    def show_error_msg(self, msg):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowFlags(QtCore.Qt.WindowType.Tool)
        msg_box.setText(msg)
        msg_box.setIcon(QtWidgets.QMessageBox.Critical)
        msg_box.setWindowTitle('Error')
        msg_box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Ok)
        msg_box.exec()