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
        - made the module standalone, can be used outside of dioptas
            (removed references to integration, pattern, calibration and possibly other stuff)

"""

from PyQt6 import QtWidgets, QtCore, QtGui



from hpm.widgets.CustomWidgets import NumberTextField, LabelAlignRight, DoubleSpinBoxAlignRight, HorizontalSpacerItem, \
    VerticalSpacerItem, FlatButton, CleanLooksComboBox

from utilities.HelperModule import convert_d_to_two_theta
from hpm.models.eos_definitions import equations_of_state
from functools import partial
import numpy as np

class EosGroupbox(QtWidgets.QWidget):
    param_edited_signal = QtCore.pyqtSignal(dict)
    eos_type_edited_signal = QtCore.pyqtSignal(dict)
    def __init__(self, equation_of_state='jcpds4'):
        super().__init__()
        self.equation_of_state = equation_of_state
        self._layout = QtWidgets.QVBoxLayout()
        
        self.eos_gb = QtWidgets.QGroupBox("Equation of State")
        self._eos_gb_layout = QtWidgets.QVBoxLayout()
        #self._eos_gb_layout.setContentsMargins(10,25,10,10)

        self.EOS_params = equations_of_state
        self.EOS_widgets = {}
        


        self.txt_fields = {}
        self.scales = {}
        
        self.eos_type_cb = CleanLooksComboBox()
        self.eos_info = ''
        


        self.eos_types = list(self.EOS_params.keys())
        self.eos_type_cb.addItems(self.eos_types)
        self._eos_type_cb_layout = QtWidgets.QHBoxLayout()
        self._eos_type_cb_layout.addWidget(QtWidgets.QLabel('Form:'))
        self._eos_type_cb_layout.addWidget(self.eos_type_cb)
        self._eos_type_info_btn = QtWidgets.QPushButton('i')
        self._eos_type_cb_layout.addWidget(self._eos_type_info_btn)
        self._eos_type_info_btn.clicked.connect(self.show_eos_info)

        self._eos_gb_layout.addLayout(self._eos_type_cb_layout)

        
        for key in self.EOS_params:
            eos = self.EOS_params[key]['params']
            eos_widget = QtWidgets.QWidget()
            _eos_layout = QtWidgets.QGridLayout()
            
            row = 0
            txt_fields = {}
            for param_key in eos:
                if param_key == 'V_0':
                    continue
                param = eos[param_key]
                symbol = param['symbol']
                
                desc = param['desc']
                unit = param['unit']
                if unit == "Pa":
                    unit = "GPa"
                    self.scales[param_key]=1e9

                text_field =  NumberTextField()
                text_field.setText('0')
                txt_fields[param_key]=text_field
                self.add_field(_eos_layout, text_field, symbol+':', unit, row, 0)
                text_field.returnPressed.connect(partial(self.text_field_edited_callback, {'eos':key,'param_key':param_key}))
                
                row = row + 1
            self.txt_fields[key]=txt_fields
            eos_widget.setLayout(_eos_layout)    
            self.EOS_widgets[key] = eos_widget
            eos_widget.setStyleSheet("""
                    NumberTextField {
                        min-width: 60;
                    }
                """)
        
        self.eos_widget = self.EOS_widgets[self.equation_of_state]
        self._eos_gb_layout.addWidget(self.eos_widget)

        
            
        self.eos_type_cb.setCurrentText(self.equation_of_state)
        self.eos_gb.setLayout(self._eos_gb_layout)
        self._layout.addWidget(self.eos_gb)
        self.setLayout(self._layout)
        
        self.eos_type_cb.currentTextChanged.connect(self.eos_type_edited)

    
    def show_eos_info(self):
        QtWidgets.QMessageBox.about(None, "EOS formulation", self.get_current_eos_info() )

    def get_current_eos_info(self):
        selected_eos = self.eos_type_cb.currentText()
        eos_name = 'EOS: ' +self.EOS_params[selected_eos]['name']
        ref = self.EOS_params[selected_eos]['reference']
        eos = self.EOS_params[selected_eos]['params']
        desc = eos_name + '<br><br>'
        for key in eos:
            param = eos[key]
            s = param['symbol'] + ': ' + param['desc'] + '<br>'
            desc = desc + s
        if ref != '':
            desc = desc +'<br>Reference:<br>' +ref 
        return desc

    def eos_type_edited(self, key):
        
        if key in self.EOS_widgets:
            self._change_eos_layout(key)
            if key in self.txt_fields:
                '''_txt_fields = self.txt_fields[key]
                d = self.dictionarize_fields(_txt_fields)
                '''
                d = {}
                d['equation_of_state']= key
                self.eos_type_edited_signal.emit(d)

    def _change_eos_layout(self, key):
        _widget = self.EOS_widgets[key]
        self._eos_gb_layout.removeWidget(self.eos_widget)
        self.eos_widget.setParent(None)
        self.eos_widget = _widget
        self._eos_gb_layout.addWidget(self.eos_widget)
            
    def dictionarize_fields(self, fields):
        d = {}
        for key in fields:
            field = fields[key]
            val = float(str(field.text()))
            if key in self.scales:
                val = float(str(val)) * self.scales[key]     
            d[key]= val
        return d

    def text_field_edited_callback(self, eos_param_key):
        key = eos_param_key['param_key']
        eos = eos_param_key['eos']
        self.equation_of_state = eos
        val = self.txt_fields[self.equation_of_state][key].text()
        if key in self.scales:
            val = float(str(val)) * self.scales[key]
        self.param_edited_signal.emit({key:val})


    def add_field(self, layout, widget, label_str, unit, x, y):
        layout.addWidget(LabelAlignRight(label_str), x, y)
        layout.addWidget(widget, x, y + 1)
        if unit:
            layout.addWidget(QtWidgets.QLabel(unit), x, y + 2)

    def setEOSparams(self, params):
        self.blockSignals(True)
        if 'equation_of_state' in params:
            eos = params['equation_of_state']
            if self.equation_of_state != eos:
                if eos in self.EOS_widgets:
                    self.eos_type_cb.setCurrentText(eos)
                    self._change_eos_layout(eos)
                self.equation_of_state = eos

            fields = self.txt_fields[self.equation_of_state]
            for key in params:
                param = params[key]
                if key in fields:
                    param = params[key]
                    if key in self.scales:
                        param = param / self.scales[key]
                    fields[key].setText(str(param))
            self.blockSignals(False)


class JcpdsEditorWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(JcpdsEditorWidget, self).__init__(parent)

        self.setWindowTitle('JCPDS Editor')

        self._layout = QtWidgets.QVBoxLayout()

        self._file_layout = QtWidgets.QGridLayout()
        self._file_layout.addWidget(LabelAlignRight('Filename:'), 0, 0)
        self._file_layout.addWidget(LabelAlignRight('Comment:'), 1, 0)

        self.filename_txt = QtWidgets.QLineEdit('')
        self.comments_txt = QtWidgets.QLineEdit('')
        self._file_layout.addWidget(self.filename_txt, 0, 1)
        self._file_layout.addWidget(self.comments_txt, 1, 1)
        self._layout.addLayout((self._file_layout))

        self.lattice_parameters_gb = QtWidgets.QGroupBox('Lattice Parameters')
        self._lattice_parameters_layout = QtWidgets.QVBoxLayout()
       

        self._symmetry_layout = QtWidgets.QHBoxLayout()
        self._symmetry_layout.addWidget(LabelAlignRight('Symmetry'))
        self.symmetry_cb = CleanLooksComboBox()
        self.symmetries = ['cubic', 'tetragonal', 'hexagonal', 'trigonal', 'rhombohedral',
                           'orthorhombic', 'monoclinic', 'triclinic']
        self.symmetry_cb.addItems(self.symmetries)
        self._symmetry_layout.addWidget(self.symmetry_cb)
        self._symmetry_layout.addSpacerItem(HorizontalSpacerItem())
        self._lattice_parameters_layout.addLayout(self._symmetry_layout)

        self._parameters_layout = QtWidgets.QGridLayout()

        self.lattice_a_sb = DoubleSpinBoxAlignRight()
        self.lattice_a_sb.setSingleStep(0.01)
        self.lattice_a_sb.setMinimum(0)
        self.lattice_a_sb.setMaximum(99999)
        self.lattice_a_sb.setDecimals(4)
        self.lattice_b_sb = DoubleSpinBoxAlignRight()
        self.lattice_b_sb.setMinimum(0)
        self.lattice_b_sb.setMaximum(99999)
        self.lattice_b_sb.setDecimals(4)
        self.lattice_b_sb.setSingleStep(0.01)
        self.lattice_c_sb = DoubleSpinBoxAlignRight()
        self.lattice_c_sb.setMinimum(0)
        self.lattice_c_sb.setMaximum(99999)
        self.lattice_c_sb.setDecimals(4)
        self.lattice_c_sb.setSingleStep(0.01)
        self.lattice_length_step_txt = NumberTextField('0.01')

        self.add_field(self._parameters_layout, self.lattice_a_sb, 'a<sub>0</sub>:', u"Å", 0, 0)
        self.add_field(self._parameters_layout, self.lattice_b_sb, 'b<sub>0</sub>:', u"Å", 0, 3)
        self.add_field(self._parameters_layout, self.lattice_c_sb, 'c<sub>0</sub>:', u"Å", 0, 6)
        self.add_field(self._parameters_layout, self.lattice_length_step_txt, 'st:', u"Å", 0, 9)

        self.lattice_eos_a_txt = NumberTextField()
        self.lattice_eos_b_txt = NumberTextField()
        self.lattice_eos_c_txt = NumberTextField()

        self.add_field(self._parameters_layout, self.lattice_eos_a_txt, 'a:', u"Å", 1, 0)
        self.add_field(self._parameters_layout, self.lattice_eos_b_txt, 'b:', u"Å", 1, 3)
        self.add_field(self._parameters_layout, self.lattice_eos_c_txt, 'c:', u"Å", 1, 6)

        self.lattice_alpha_sb = DoubleSpinBoxAlignRight()
        self.lattice_alpha_sb.setMaximum(180)
        self.lattice_beta_sb = DoubleSpinBoxAlignRight()
        self.lattice_beta_sb.setMaximum(180)
        self.lattice_gamma_sb = DoubleSpinBoxAlignRight()
        self.lattice_gamma_sb.setMaximum(180)
        self.lattice_angle_step_txt = NumberTextField('1')

        self.add_field(self._parameters_layout, self.lattice_alpha_sb, u'α:', u"°", 2, 0)
        self.add_field(self._parameters_layout, self.lattice_beta_sb, u'β:', u"°", 2, 3)
        self.add_field(self._parameters_layout, self.lattice_gamma_sb, u'γ:', u"°", 2, 6)
        self.add_field(self._parameters_layout, self.lattice_angle_step_txt, u'st:', u"°", 2, 9)

        self.lattice_ab_sb = DoubleSpinBoxAlignRight()
        self.lattice_ab_sb.setDecimals(4)
        self.lattice_ca_sb = DoubleSpinBoxAlignRight()
        self.lattice_ca_sb.setDecimals(4)
        self.lattice_cb_sb = DoubleSpinBoxAlignRight()
        self.lattice_cb_sb.setDecimals(4)
        self.lattice_ratio_step_txt = NumberTextField('0.01')

        self.add_field(self._parameters_layout, self.lattice_ab_sb, 'a/b:', None, 3, 0)
        self.add_field(self._parameters_layout, self.lattice_ca_sb, 'c/a:', None, 3, 3)
        self.add_field(self._parameters_layout, self.lattice_cb_sb, 'c/b:', None, 3, 6)
        self.add_field(self._parameters_layout, self.lattice_ratio_step_txt, 'st:', None, 3, 9)

        self.lattice_volume_txt = NumberTextField()
        self.lattice_eos_volume_txt = NumberTextField()
        self.lattice_eos_molar_volume_txt = NumberTextField()
        self.lattice_eos_z_txt = NumberTextField()

        self.add_field(self._parameters_layout, self.lattice_volume_txt, 'V<sub>0</sub>:', u'Å³', 4, 0)
        self.add_field(self._parameters_layout, self.lattice_eos_volume_txt, 'V:', u'Å³', 4, 3)

        self.add_field(self._parameters_layout, self.lattice_eos_molar_volume_txt, 'V<sub>m</sub>:', u'm³/mol', 5, 3)
        self.add_field(self._parameters_layout, self.lattice_eos_z_txt, 'Z:', u'', 5, 0)
        

        self._lattice_parameters_layout.addLayout(self._parameters_layout)
        self.lattice_parameters_gb.setLayout(self._lattice_parameters_layout)

        self.eos_widget = EosGroupbox()

        self.reflections_gb = QtWidgets.QGroupBox('Reflections')
        self._reflection_layout = QtWidgets.QGridLayout()
        self.reflection_table_view = QtWidgets.QTableView()
        self.reflection_table_model = ReflectionTableModel()
        self.reflection_table_view.setModel(self.reflection_table_model)
        #self.reflection_table.setColumnCount(8)
        self.reflections_add_btn = FlatButton('Add')
        self.reflections_delete_btn = FlatButton('Delete')
        self.reflections_clear_btn = FlatButton('Clear')

        self._reflection_layout.addWidget(self.reflection_table_view, 0, 0, 1, 3)
        self._reflection_layout.addWidget(self.reflections_add_btn, 1, 0)
        self._reflection_layout.addWidget(self.reflections_delete_btn, 1, 1)
        self._reflection_layout.addWidget(self.reflections_clear_btn, 1, 2)

        self.reflections_gb.setLayout(self._reflection_layout)

        self._body_layout = QtWidgets.QGridLayout()
        self._body_layout.addWidget(self.eos_widget, 0, 0)
        self._body_layout.addItem(VerticalSpacerItem(), 1, 0)
        self._body_layout.addWidget(self.reflections_gb, 0, 1, 2, 1)

        self._button_layout = QtWidgets.QHBoxLayout()
        self.save_as_btn = FlatButton('Save As')
        self.reload_file_btn = FlatButton('Reload File')
        

        self._button_layout.addWidget(self.save_as_btn)
        self._button_layout.addWidget(self.reload_file_btn)
        self._button_layout.addSpacerItem(HorizontalSpacerItem())


        self._layout.addWidget(self.lattice_parameters_gb)
        self._layout.addLayout(self._body_layout)
        self._layout.addLayout(self._button_layout)
        self.setLayout(self._layout)

        self.style_widgets()
        

    def style_widgets(self):
        self.lattice_angle_step_txt.setMaximumWidth(60)
        self.lattice_length_step_txt.setMaximumWidth(60)
        self.lattice_ratio_step_txt.setMaximumWidth(60)
        self.lattice_eos_z_txt.setMaximumWidth(40)

        
        self.reflection_table_view.setShowGrid(False)
        self.reflection_table_view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.reflection_table_view.setItemDelegate(TextDoubleDelegate())
        self.reflection_table_view.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)

        self.eos_widget.setMaximumWidth(250)
        self.eos_widget.setStyleSheet("""
            QLineEdit {
                max-width: 80;
            }
        """)

        self.reflection_table_view.verticalHeader().setDefaultSectionSize(20)
        self.reflection_table_view.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Fixed)

        self.setWindowFlags(QtCore.Qt.WindowType.Tool)
        #self.setAttribute(QtCore.Qt.WA_MacAlwaysShowToolWindow)
        

    def close(self, *args, **kwargs):

        super().close()

    def raise_widget(self):
        self.show()
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowState.WindowMinimized | QtCore.Qt.WindowState.WindowActive)
        self.activateWindow()
        self.raise_()

    def add_field(self, layout, widget, label_str, unit, x, y):
        layout.addWidget(LabelAlignRight(label_str), x, y)
        layout.addWidget(widget, x, y + 1)
        if unit:
            layout.addWidget(QtWidgets.QLabel(unit), x, y + 2)

    def show_jcpds(self, jcpds_phase, wavelength=None):
        self.update_name(jcpds_phase)
        self.update_lattice_parameters(jcpds_phase)
        self.update_eos_parameters(jcpds_phase)
        self.reflection_table_model.update_reflection_data(jcpds_phase.reflections,
                                                           wavelength)


    # jcpds V5
    def update_eos_parameters(self, jcpds_phase):
        if 'eos' in jcpds_phase.params and 'z' in jcpds_phase.params:
            self.eos_widget.setEOSparams(jcpds_phase.params['eos'])
            self.lattice_eos_z_txt.setText(str(jcpds_phase.params['z']))

    def update_name(self, jcpds_phase):
        self.filename_txt.setText(jcpds_phase.filename)
        self.comments_txt.setText("/n".join(jcpds_phase.params['comments']))

    def update_lattice_parameters(self, jcpds_phase):
        self.blockAllSignals(True)
        self.symmetry_cb.setCurrentIndex(self.symmetries.index(jcpds_phase.params['symmetry'].lower()))
        self.update_spinbox_enable(jcpds_phase.params['symmetry'])

        if not self.lattice_a_sb.hasFocus():
            self.lattice_a_sb.setValue(jcpds_phase.params['a0'])
        if not self.lattice_b_sb.hasFocus():
            self.lattice_b_sb.setValue(jcpds_phase.params['b0'])
        if not self.lattice_c_sb.hasFocus():
            self.lattice_c_sb.setValue(jcpds_phase.params['c0'])

        self.lattice_eos_a_txt.setText('{0:.4f}'.format(jcpds_phase.params['a']))
        self.lattice_eos_b_txt.setText('{0:.4f}'.format(jcpds_phase.params['b']))
        self.lattice_eos_c_txt.setText('{0:.4f}'.format(jcpds_phase.params['c']))

        self.lattice_eos_volume_txt.setText('{0:.4f}'.format(jcpds_phase.params['v']))
        
        # jcpds V5
        if 'vm' in jcpds_phase.params.keys():
            vm = jcpds_phase.params['vm']
            vm_txt = '{0:.4e}'.format(vm)
        else:
            vm_txt = ''
        self.lattice_eos_molar_volume_txt.setText(vm_txt)
        
        try:
            if not self.lattice_ab_sb.hasFocus():
                self.lattice_ab_sb.setValue(jcpds_phase.params['a0'] / float(jcpds_phase.params['b0']))
        except ZeroDivisionError:
            self.lattice_ab_sb.setSpecialValueText('Inf')

        try:
            if not self.lattice_ca_sb.hasFocus():
                self.lattice_ca_sb.setValue(jcpds_phase.params['c0'] / float(jcpds_phase.params['a0']))
        except ZeroDivisionError:
            self.lattice_ca_sb.setSpecialValueText('Inf')

        try:
            if not self.lattice_cb_sb.hasFocus():
                self.lattice_cb_sb.setValue(jcpds_phase.params['c0'] / float(jcpds_phase.params['b0']))
        except ZeroDivisionError:
            self.lattice_cb_sb.setSpecialValueText('Inf')

        self.lattice_volume_txt.setText(str('{0:g}'.format(jcpds_phase.params['v0'])))

        if not self.lattice_alpha_sb.hasFocus():
            self.lattice_alpha_sb.setValue(jcpds_phase.params['alpha0'])
        if not self.lattice_beta_sb.hasFocus():
            self.lattice_beta_sb.setValue(jcpds_phase.params['beta0'])
        if not self.lattice_gamma_sb.hasFocus():
            self.lattice_gamma_sb.setValue(jcpds_phase.params['gamma0'])

        self.blockAllSignals(False)

    def blockAllSignals(self, bool=True):
        self.lattice_a_sb.blockSignals(bool)
        self.lattice_b_sb.blockSignals(bool)
        self.lattice_c_sb.blockSignals(bool)

        self.lattice_alpha_sb.blockSignals(bool)
        self.lattice_beta_sb.blockSignals(bool)
        self.lattice_gamma_sb.blockSignals(bool)

        self.lattice_ab_sb.blockSignals(bool)
        self.lattice_ca_sb.blockSignals(bool)
        self.lattice_cb_sb.blockSignals(bool)

        self.symmetry_cb.blockSignals(bool)

    def update_spinbox_enable(self, symmetry):
        if symmetry == 'CUBIC':
            self.lattice_a_sb.setEnabled(True)
            self.lattice_b_sb.setEnabled(False)
            self.lattice_c_sb.setEnabled(False)

            self.lattice_alpha_sb.setEnabled(False)
            self.lattice_beta_sb.setEnabled(False)
            self.lattice_gamma_sb.setEnabled(False)

            self.lattice_ab_sb.setEnabled(False)
            self.lattice_ca_sb.setEnabled(False)
            self.lattice_cb_sb.setEnabled(False)

        elif symmetry == 'TETRAGONAL':
            self.lattice_a_sb.setEnabled(True)
            self.lattice_b_sb.setEnabled(False)
            self.lattice_c_sb.setEnabled(True)

            self.lattice_alpha_sb.setEnabled(False)
            self.lattice_beta_sb.setEnabled(False)
            self.lattice_gamma_sb.setEnabled(False)

            self.lattice_ab_sb.setEnabled(False)
            self.lattice_ca_sb.setEnabled(True)
            self.lattice_cb_sb.setEnabled(False)

        elif symmetry == 'ORTHORHOMBIC':
            self.lattice_a_sb.setEnabled(True)
            self.lattice_b_sb.setEnabled(True)
            self.lattice_c_sb.setEnabled(True)

            self.lattice_alpha_sb.setEnabled(False)
            self.lattice_beta_sb.setEnabled(False)
            self.lattice_gamma_sb.setEnabled(False)

            self.lattice_ab_sb.setEnabled(True)
            self.lattice_ca_sb.setEnabled(True)
            self.lattice_cb_sb.setEnabled(True)

        elif symmetry == 'HEXAGONAL' or symmetry == 'TRIGONAL':
            self.lattice_a_sb.setEnabled(True)
            self.lattice_b_sb.setEnabled(False)
            self.lattice_c_sb.setEnabled(True)

            self.lattice_alpha_sb.setEnabled(False)
            self.lattice_beta_sb.setEnabled(False)
            self.lattice_gamma_sb.setEnabled(False)

            self.lattice_ab_sb.setEnabled(False)
            self.lattice_ca_sb.setEnabled(True)
            self.lattice_cb_sb.setEnabled(False)

        elif symmetry == 'RHOMBOHEDRAL':
            self.lattice_a_sb.setEnabled(True)
            self.lattice_b_sb.setEnabled(False)
            self.lattice_c_sb.setEnabled(False)

            self.lattice_alpha_sb.setEnabled(True)
            self.lattice_beta_sb.setEnabled(False)
            self.lattice_gamma_sb.setEnabled(False)

            self.lattice_ab_sb.setEnabled(False)
            self.lattice_ca_sb.setEnabled(False)
            self.lattice_cb_sb.setEnabled(False)

        elif symmetry == 'MONOCLINIC':
            self.lattice_a_sb.setEnabled(True)
            self.lattice_b_sb.setEnabled(True)
            self.lattice_c_sb.setEnabled(True)

            self.lattice_alpha_sb.setEnabled(False)
            self.lattice_beta_sb.setEnabled(True)
            self.lattice_gamma_sb.setEnabled(False)

            self.lattice_ab_sb.setEnabled(True)
            self.lattice_ca_sb.setEnabled(True)
            self.lattice_cb_sb.setEnabled(True)

        elif symmetry == 'TRICLINIC':
            self.lattice_a_sb.setEnabled(True)
            self.lattice_b_sb.setEnabled(True)
            self.lattice_c_sb.setEnabled(True)

            self.lattice_alpha_sb.setEnabled(True)
            self.lattice_beta_sb.setEnabled(True)
            self.lattice_gamma_sb.setEnabled(True)

            self.lattice_ab_sb.setEnabled(True)
            self.lattice_ca_sb.setEnabled(True)
            self.lattice_cb_sb.setEnabled(True)

        else:
            print('Unknown symmetry: {0}.'.format(symmetry))

    def get_selected_reflections(self):
        selected = self.reflection_table_view.selectionModel().selectedRows()
        try:
            row = []
            for element in selected:
                row.append(int(element.row()))
        except IndexError:
            row = None
        return row

    

    def remove_reflection_from_table(self, ind):
        self.reflection_table.blockSignals(True)
        self.reflection_table.removeRow(ind)
        self.reflection_table.blockSignals(False)


        # def get_jcpds(self):
        # self.jcpds_phase.filename = str(self.filename_txt.text())
        #     self.jcpds_phase.comments_txt = [str(self.comments_text())]
        #
        #     self.jcpds_phase.symmetry = str(self.symmetry_cb.text()).upper()
        #
        #     self.jcpds_phase.a0 = float(str(self.lattice_a_txt.text()))
        #     self.jcpds_phase.b0 = float(str(self.lattice_b_txt.text()))
        #     self.jcpds_phase.c0 = float(str(self.lattice_c_txt.text()))
        #
        #     self.jcpds_phase.alpha0 = float(str(self.lattice_alpha_txt.text()))
        #     self.jcpds_phase.beta0 = float(str(self.lattice_beta_txt.text()))
        #     self.jcpds_phase.gamma0 = float(str(self.lattice_gamma_txt.text()))
        #
        #     self.jcpds_phase.k0 = float(str(self.eos_K_txt.text()))
        #     self.jcpds_phase.k0p = float(str(self.eos_Kp_txt.text()))
        #     self.jcpds_phase.alpha_t0 = float(str(self.eos_alphaT_txt.text()))
        #     self.jcpds_phase.d_alpha_dt = float(str(self.eos_dalphadT_txt.text()))
        #     self.jcpds_phase.dk0dt = float(str(self.eos_dKdT_txt.text()))
        #     self.jcpds_phase.dk0pdt = float(str(self.eos_dKpdT_txt.text()))

class NoRectDelegate(QtWidgets.QItemDelegate):
    def drawFocus(self, painter, option, rect):
        option.state &= ~QtWidgets.QStyle.StateFlag.State_HasFocus
        QtWidgets.QItemDelegate.drawFocus(self, painter, option, rect)


class TextDoubleDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent):
        super(TextDoubleDelegate, self).__init__(parent)

    def createEditor(self, parent, _, model):
        self.editor = QtWidgets.QLineEdit(parent)
        self.editor.setFrame(False)
        self.editor.setValidator(QtGui.QDoubleValidator())
        self.editor.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        return self.editor

    def setEditorData(self, parent, index):
        value = index.model().data(index, QtCore.Qt.EditRole)
        if str(value) != '':
            self.editor.setText("{0:g}".format(float(str(value))))

    def setModelData(self, parent, model, index):
        value = self.editor.text()
        model.setData(index, value, QtCore.Qt.EditRole)

    def updateEditorGeometry(self, editor, option, _):
        editor.setGeometry(option.rect)


class CenteredQTableWidgetItem(QtWidgets.QTableWidgetItem):
    def __init__(self, value):
        super(CenteredQTableWidgetItem, self).__init__(value)
        self.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)

    def __gt__(self, other):
        return float(str(self.text()) > float(str(other.text())))

    def __lt__(self, other):
        return float(str(self.text())) < float(str(other.text()))

    def __ge__(self, other):
        return float(str(self.text()) >= float(str(other.text())))

    def __le__(self, other):
        return float(str(self.text())) <= float(str(other.text()))

    def __eq__(self, other):
        return float(str(self.text())) == float(str(other.text()))

    def __ne__(self, other):
        return float(str(self.text())) != float(str(other.text()))


class CenteredNonEditableQTableWidgetItem(CenteredQTableWidgetItem):
    def __init__(self, value):
        super(CenteredNonEditableQTableWidgetItem, self).__init__(value)
        self.setFlags(self.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    from jcpds import jcpds
    import os

    test_phase = jcpds()
    path = os.path.join(os.path.dirname(__file__), '../../')
    path = os.path.join(path, 'tests', 'data', 'jcpds', 'ag.jcpds')
    print(os.path.abspath(path))
    test_phase.load_file(path)
    widget = JcpdsEditorWidget(None)
    widget.show_jcpds(test_phase, 0.31)
    widget.show()
    widget.raise_()
    app.exec_()


class TextDoubleDelegate(NoRectDelegate):
    def createEditor(self, parent, _, model):
        self.editor = QtWidgets.QLineEdit(parent)
        self.editor.setFrame(False)
        self.editor.setValidator(QtGui.QDoubleValidator())
        self.editor.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        return self.editor

class ReflectionTableModel(QtCore.QAbstractTableModel):
    reflection_edited = QtCore.pyqtSignal(int, int, str)  # row, column, value

    def __init__(self, reflections=None, wavelength=None):
        super(ReflectionTableModel, self).__init__()
        self.wavelength = wavelength
        if reflections is not None:
            self.reflections = reflections
            self.update_reflection_data(reflections)
        else:
            self.reflections = []
        self.header_labels = ['h', 'k', 'l', 'Intensity', 'd0', 'd', u"2θ_0", u"2θ", 'Q0', 'Q']

    def rowCount(self, *_):
        return len(self.reflections)

    def columnCount(self, *_):
        return 10

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        col = index.column()
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            return QtCore.Qt.AlignmentFlag.AlignCenter
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if col < 4:
                format_str = '{0:g}'
            else:
                format_str = '{0:.4f}'
            return format_str.format(self.reflection_data[index.row(), index.column()])
        else:
            return QtCore.QVariant()

    def setData(self, index, value, role):
        self.reflection_edited.emit(index.row(), index.column(), value)
        return True

    def update_reflection_data(self, reflections, wavelength=None):
        if wavelength is None:
            wavelength = self.wavelength
        else:
            self.wavelength = wavelength

        cur_row_num = self.rowCount()
        row_diff = len(reflections) - cur_row_num
        if row_diff < 0:
            self.beginRemoveRows(QtCore.QModelIndex(), cur_row_num + row_diff, cur_row_num - 1)
        elif row_diff > 0:
            self.beginInsertRows(QtCore.QModelIndex(), cur_row_num, cur_row_num + row_diff - 1)

        self.reflections = reflections
        self.reflection_data = np.zeros((len(reflections), self.columnCount()))
        for i, refl in enumerate(reflections):
            self.reflection_data[i, 0] = refl.h
            self.reflection_data[i, 1] = refl.k
            self.reflection_data[i, 2] = refl.l
            self.reflection_data[i, 3] = refl.intensity
            self.reflection_data[i, 4] = refl.d0
            self.reflection_data[i, 5] = refl.d

        if wavelength is not None:
            self.reflection_data[:, 6] = convert_d_to_two_theta(self.reflection_data[:, 4], wavelength)  # two_theta0
            self.reflection_data[:, 7] = convert_d_to_two_theta(self.reflection_data[:, 5], wavelength)  # two_theta
            valid_ind = np.where(self.reflection_data[:, 4] > 0)
            self.reflection_data[valid_ind, 8] = 2.0 * np.pi / self.reflection_data[valid_ind, 4]  # q0
            self.reflection_data[valid_ind, 9] = 2.0 * np.pi / self.reflection_data[valid_ind, 5]  # q

        if row_diff < 0:
            self.endRemoveRows()
        elif row_diff > 0:
            self.endInsertRows()

        self.modelReset.emit()

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = ...):
        if orientation == QtCore.Qt.Orientation.Horizontal and role == QtCore.Qt.ItemDataRole.DisplayRole:
            return self.header_labels[section]
        if orientation == QtCore.Qt.Orientation.Vertical and role == QtCore.Qt.ItemDataRole.DisplayRole:
            return section + 1

    def flags(self, index):
        col = index.column()
        ans = QtCore.QAbstractTableModel.flags(self, index)
        if col <= 3:
            return QtCore.Qt.ItemFlag.ItemIsEditable | ans
        else:
            return ans
