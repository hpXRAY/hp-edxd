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
import numpy as Numeric
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

import copy
import functools
import math
import pyqtgraph as pg
from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight

class LatticeRefinementWidget(QtWidgets.QWidget):
    widget_closed = QtCore.pyqtSignal()
    
    

    def __init__(self,):
        super().__init__()
        self.setWindowTitle("Lattice refinement")
        self.setMinimumWidth(600)
        self.roi = []
        
        self._layout = QtWidgets.QVBoxLayout()  
        self.button_widget = QtWidgets.QWidget(self)
        self.button_widget.setObjectName('reflections_control_button_widget')
        self._button_layout = QtWidgets.QHBoxLayout(self.button_widget)
        self._button_layout.setContentsMargins(0, 0, 0, 0)
        self._button_layout.setSpacing(6)
        self.do_fit = t = QtWidgets.QPushButton(self.button_widget, default=False, autoDefault=False)
        t.setText("Refine Lattice")
        t.setFixedWidth(100)
        self._button_layout.addWidget(t)
        self.auto_fit = t = QtWidgets.QPushButton(self.button_widget, default=False, autoDefault=False)
        t.setText("Auto process")
        t.setCheckable(True)
        t.setFixedWidth(100)
        self._button_layout.addWidget(t)
        self.plot_cal = t = QtWidgets.QPushButton(self.button_widget, default=False, autoDefault=False)
        t.setText(f'Plot \N{GREEK CAPITAL LETTER DELTA} d')
        t.setFixedWidth(100)
        self._button_layout.addWidget(t)



        self.lbltwo_theta = t = QtWidgets.QLabel(self.button_widget)
        t.setText(f'2\N{GREEK SMALL LETTER THETA}:')
        t.setFixedWidth(30)
        t.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self._button_layout.addWidget(t)
        self.two_theta = t = QtWidgets.QLabel(self.button_widget)
        self._button_layout.addWidget(t)

        self._button_layout.addSpacerItem(HorizontalSpacerItem())

        self.update_pressure_btn = t = QtWidgets.QPushButton('Send Pressure')
        t.setFixedWidth(100)
        self._button_layout.addWidget(t)

        self.auto_pressure_btn = t = QtWidgets.QPushButton('Auto Send')
        t.setFixedWidth(100)
        t.setCheckable(True)
        self._button_layout.addWidget(t)
        t = 0
        
        
        self._layout.addWidget(self.button_widget)


        self._phase_selection_layout = QtWidgets.QHBoxLayout()
        self.phase_file_label = QtWidgets.QLabel("Phase")
        self._phase_selection_layout.addWidget(self.phase_file_label)
        self.phases_cbx = QtWidgets.QComboBox()
        self.phases_cbx.setMaximumWidth(200)
        self.phases_cbx.setMinimumWidth(200)
        self._phase_selection_layout.addWidget(self.phases_cbx)
        self.model_lbl = QtWidgets.QLabel('')
        self._phase_selection_layout.addWidget(self.model_lbl)
        self._phase_selection_layout.addSpacerItem(HorizontalSpacerItem())
        self._layout.addLayout(self._phase_selection_layout)


        
        self._body_layout = QtWidgets.QHBoxLayout()
        self._layout.addLayout(self._body_layout)


        self._horizontal_layout = QtWidgets.QHBoxLayout()
        
        self.roi_tw = reflectionsTableWidget()
        self._horizontal_layout.addWidget(self.roi_tw)
        
        #self.param_tw = latticeOutputTableWidget()
        #self._horizontal_layout.addWidget(self.phases_lbl)
        self.parameter_widget = parametersTableWidget()
        self.parameter_widget.setMinimumHeight(207)
        self.right_widget = QtWidgets.QWidget()
        self._right_widget_layout = QtWidgets.QVBoxLayout(self.right_widget)
        
        self._right_widget_layout.addWidget(self.model_lbl)
        self._right_widget_layout.addWidget(self.parameter_widget)
        self._right_widget_layout.setContentsMargins(10,0,10,0)
        self._right_widget_layout.addSpacerItem(VerticalSpacerItem())

        self._horizontal_layout.addWidget(self.right_widget)

        
        self._body_layout.addLayout(self._horizontal_layout)
        self.setLayout(self._layout)
        self.style_widgets()

    
        

    def style_widgets(self):
        self.setStyleSheet("""
            #roi_control_button_widget QPushButton {
                min-width: 75;
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

class paramValue(QtWidgets.QLabel):
    def __init__(self, text=''):
        super().__init__(text)
        self.setMinimumWidth(60)
        self.setMaximumWidth(60)
        

class paramOutWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self._parameter_layout = QtWidgets.QGridLayout(self)
        self.setStyleSheet('background: #101010')
        self._parameter_layout.setContentsMargins(0,0,0,0)
        self._parameter_layout.setSpacing(1)
        
        self.rows =[
                    ['','value', 'esd'],
                    [f' a (\N{LATIN CAPITAL LETTER A WITH RING ABOVE})','a_out','a_delta_out'],
                    [f' b (\N{LATIN CAPITAL LETTER A WITH RING ABOVE})','b_out','b_delta_out'],
                    [f' c (\N{LATIN CAPITAL LETTER A WITH RING ABOVE})','c_out','c_delta_out'],
                    [f' \N{GREEK SMALL LETTER ALPHA} (\N{DEGREE SIGN})','alpha_out','alpha_delta_out'] ,
                    [f' \N{GREEK SMALL LETTER BETA} (\N{DEGREE SIGN})','beta_out','beta_delta_out'] ,
                    [f' \N{GREEK SMALL LETTER GAMMA} (\N{DEGREE SIGN})','gamma_out','gamma_delta_out'],
                    [f' V (\N{LATIN CAPITAL LETTER A WITH RING ABOVE}\N{SUPERSCRIPT THREE})','v_out','v_delta_out'] ,
                    [f' V/V\N{SUBSCRIPT ZERO}','v_v0_out','v_v0_delta_out'],
                    [' P (GPa)','p_out','p_delta_out'],
                    [' T (K)','t_out','t_delta_out']
                   ]
        
        #### display parameters 
        for i in range(len(self.rows)):
            label = paramValue(self.rows[i][0])
            
            setattr(self, self.rows[i][1], paramValue())
            setattr(self, self.rows[i][2], paramValue())
            self._parameter_layout.addWidget(label, i, 0)
            self._parameter_layout.addWidget(getattr(self,self.rows[i][1]), i, 1)
            self._parameter_layout.addWidget(getattr(self,self.rows[i][2]), i, 2)
        getattr(self,self.rows[0][1]).setText(' Value')
        getattr(self,self.rows[0][2]).setText(' esd')

    def update_output(self,  lattice,lattice_esd, P,V,T, v_over_v0, P_esd, V_esd, v_over_v0_esd):
        
        
        
        if len(lattice):
            
            row = 1
            for i, line in enumerate( lattice):
                text = ' %.4f'%(round(lattice[line],4)) 
                text_esd = ' %.4f'%(round(lattice_esd[line],4))
                getattr(self, self.rows[i+1][1]).setText(text)
                getattr(self, self.rows[i+1][2]).setText(text_esd)
                row = row +1
                
            pvt =  [' %.3f'%(V),' %.4f'%(v_over_v0),' %.2f'%(round(P,2)),' %.2f'%(T) ]
            pvt_esd = [' %.3f'%(V_esd),' %.4f'%(v_over_v0_esd),' %.2f'%(round(P_esd,2)),' ']
            for i, line in enumerate( pvt):
                text = line
                text_esd = pvt_esd[i]
                getattr(self, self.rows[i+row][1]).setText(text)
                getattr(self, self.rows[i+row][2]).setText(text_esd)

    def clear(self):
        for i, row in enumerate( self.rows):
            getattr(self, self.rows[i][1]).setText('')
            getattr(self, self.rows[i][2]).setText('')
 
class reflectionsTableWidget(ListTableWidget):
    show_cb_state_changed = QtCore.pyqtSignal(int, int)
    name_item_changed = QtCore.pyqtSignal(int, str)
    def __init__(self):
        super().__init__(columns=6)
        self.roi_show_cbs = []
        self.name_items = []
        self.index_items = []
        self.nreflections = 0
        #### display column headings    
        

        self.setMinimumWidth(400)
        header_view = QtWidgets.QHeaderView(QtCore.Qt.Horizontal, self)
        self.setHorizontalHeader(header_view)
        header_view.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        header_view.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self.default_header = ['Use','ROI','HKL',
            'd obs','d calc',f'\N{GREEK CAPITAL LETTER DELTA} d']
        self.header = copy.deepcopy(self.default_header)
        self.setHorizontalHeaderLabels(self.header)
        self.setItemDelegate(NoRectDelegate())

    def populate_reflections(self):
        nreflections = self.nreflections
        #### display reflections parameters
        for i in range(nreflections):
            row=i+1
            use = self.roi[i].use==1
            label = self.roi[i].label.split(' ')[-1]
            dobs = '%.4f' % self.roi[i].d_spacing
            dcalc = '%.4f' % 0.0
            ddiff = '%.4f' % 0.0
            self.add_roi(row, use, label, dobs, dcalc, ddiff)

    def set_reflections(self, reflections):
        self.roi = reflections
        self.nreflections = len(self.roi)
        
        self.populate_reflections()
        
    def roi_removed(self, ind):
        self.del_roi(ind)

    def update_roi(self, ind, dcalc, ddiff):
        self.blockSignals(True)
        counts_item = self.item(ind, 4)
        counts_item.setText('%.4f' % dcalc)
        fwhm_item = self.item(ind, 5)
        fwhm_item.setText('%.4f' % ddiff)
        self.blockSignals(False)

    def clear_reflections(self, *args, **kwargs):
        """
        Deletes all reflections from the GUI
        """
        
        while self.rowCount() > 0:
            self.roi_removed(self.rowCount()-1)
        
        self.roi_show_cbs = []
        self.name_items = []
        self.index_items = []

    def add_roi(self, row, use, label, dobs ,dcalc, ddiff, silent=False):

        self.blockSignals(True)
        current_rows = self.rowCount()
        self.setRowCount(current_rows + 1)

        index_item = QtWidgets.QTableWidgetItem(str(row))
        #index_item.setText(str(ind))
        index_item.setFlags(index_item.flags() & ~QtCore.Qt.ItemIsEditable)
        index_item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.setItem(current_rows, 1, index_item)
        self.index_items.append(index_item)
        
        show_cb = QtWidgets.QCheckBox()
        show_cb.setFixedWidth(40)
        show_cb.setChecked(use)
        show_cb.stateChanged.connect(partial(self.roi_show_cb_changed, show_cb))
        show_cb.setStyleSheet("background-color: transparent")
        self.setCellWidget(current_rows, 0, show_cb)
        self.roi_show_cbs.append(show_cb)

        name_item = QtWidgets.QTableWidgetItem(label)
        name_item.setText(label)
        name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemIsEditable)
        name_item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.setItem(current_rows, 2, name_item)
        self.name_items.append(name_item)

        centroid_item = QtWidgets.QTableWidgetItem(dobs)
        centroid_item.setFlags(centroid_item.flags() & ~QtCore.Qt.ItemIsEditable)
        centroid_item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.setItem(current_rows, 3, centroid_item)

        counts_item = QtWidgets.QTableWidgetItem(dcalc)
        counts_item.setFlags(counts_item.flags() & ~QtCore.Qt.ItemIsEditable)
        counts_item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.setItem(current_rows, 4, counts_item)

        fwhm_item = QtWidgets.QTableWidgetItem(ddiff)
        fwhm_item.setFlags(fwhm_item.flags() & ~QtCore.Qt.ItemIsEditable)
        fwhm_item.setTextAlignment(QtCore.Qt.AlignHCenter| QtCore.Qt.AlignVCenter)
        self.setItem(current_rows, 5, fwhm_item)

        self.setColumnWidth(0, 25)
        self.setRowHeight(current_rows, 25)
        
        if not silent:
            self.select_roi(current_rows)
            self.blockSignals(False)

    

    def del_roi(self, ind, silent=False):
        self.blockSignals(True)
        self.removeRow(ind)
        if not silent:
            self.blockSignals(False)
        #del self.roi_show_cbs[ind]
        del self.name_items[ind]
        del self.index_items[ind]
        #del self.roi_color_btns[ind]
        if not silent:
            if self.rowCount() > ind:
                self.select_roi(ind)
            else:
                self.select_roi(self.rowCount() - 1)

    def select_roi(self, ind):
        self.selectRow(ind)

    def roi_show_cb_changed(self, checkbox):
        checked = checkbox.isChecked()
        index = self.roi_show_cbs.index(checkbox)
        self.roi[index].use = checked 
        if checked: state = 1
        else: state = 0
        self.show_cb_state_changed.emit(index, state) 

    def get_selected_roi_row(self):
        selected = self.selectionModel().selectedRows()
        try:
            row = selected[0].row()
        except IndexError:
            row = -1
        return row

    def get_use(self):
        use = []
        for cb in self.roi_show_cbs:
            use.append(cb.isChecked())
        return use


class parametersTableWidget(QtWidgets.QTableWidget):
    show_cb_state_changed = QtCore.pyqtSignal(int, int)
    name_item_changed = QtCore.pyqtSignal(int, str)
    def __init__(self):
        super().__init__()
        self.setColumnCount(3)
        self.name_items = []
        self.value_items = []
        self.esd_items = []
        self.nparameters = 0
        #### display column headings    
        self.verticalHeader().setVisible(False)

        self.setMinimumWidth(170)
        header_view = QtWidgets.QHeaderView(QtCore.Qt.Horizontal, self)
        self.setHorizontalHeader(header_view)
        header_view.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        
        header_view.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header_view.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.default_header = ['Parameter','Value','esd']
        self.header = copy.deepcopy(self.default_header)
        self.setHorizontalHeaderLabels(self.header)
        #self.setItemDelegate(NoRectDelegate())
        self.populate_parameters()

    def keyPressEvent(self, event):
        
        super().keyPressEvent(event)
        if event.key() == Qt.Key_C and (event.modifiers() & Qt.ControlModifier):
            copied_cells = sorted(self.selectedIndexes())

            copy_text = ''
            max_column = copied_cells[-1].column()
            for c in copied_cells:
                copy_text += self.item(c.row(), c.column()).text()
                if c.column() == max_column:
                    copy_text += '\n'
                else:
                    copy_text += '\t'
                    
            QApplication.clipboard().setText(copy_text)
        


    def populate_parameters(self):
        self.rows =[
                    [f' a (\N{LATIN CAPITAL LETTER A WITH RING ABOVE})','a_out','a_delta_out'],
                    [f' b (\N{LATIN CAPITAL LETTER A WITH RING ABOVE})','b_out','b_delta_out'],
                    [f' c (\N{LATIN CAPITAL LETTER A WITH RING ABOVE})','c_out','c_delta_out'],
                    [f' \N{GREEK SMALL LETTER ALPHA} (\N{DEGREE SIGN})','alpha_out','alpha_delta_out'] ,
                    [f' \N{GREEK SMALL LETTER BETA} (\N{DEGREE SIGN})','beta_out','beta_delta_out'] ,
                    [f' \N{GREEK SMALL LETTER GAMMA} (\N{DEGREE SIGN})','gamma_out','gamma_delta_out'],
                    [f' V (\N{LATIN CAPITAL LETTER A WITH RING ABOVE}\N{SUPERSCRIPT THREE})','v_out','v_delta_out'] ,
                    [f' V/V\N{SUBSCRIPT ZERO}','v_v0_out','v_v0_delta_out'],
                    [' P (GPa)','p_out','p_delta_out'],
                    [' T (K)','t_out','t_delta_out']
                   ]
        #### display parameters parameters
        for row in self.rows:
            
            
            label = row[0]
            val = ''
            esd = ''
            
            self.add_param(label, val, esd)

    def clear(self):
        for i in range (len (self.rows)):
            self.update_param(i, '', '')

    def update_output(self,  lattice,lattice_esd, P,V,T, v_over_v0, P_esd, V_esd, v_over_v0_esd):
        
        if len(lattice):
            
            row = 0
            for i, line in enumerate( lattice):
                text = ' %.4f'%(round(lattice[line],4)) 
                text_esd = ' %.4f'%(round(lattice_esd[line],4))
                self.update_param(i, text, text_esd)
                
                row = row +1
                
            pvt =  [' %.3f'%(V),' %.4f'%(v_over_v0),' %.2f'%(round(P,2)),' %.2f'%(T) ]
            pvt_esd = [' %.3f'%(V_esd),' %.4f'%(v_over_v0_esd),' %.2f'%(round(P_esd,2)),' ']
            for i, line in enumerate( pvt):
                text = line
                text_esd = pvt_esd[i]
                self.update_param(i+row, text, text_esd)
            


    def update_param(self, ind, val, esd):
        self.blockSignals(True)
        val_item = self.item(ind, 1)
        val_item.setText(val)
        esd_item = self.item(ind, 2)
        esd_item.setText(esd)
        self.blockSignals(False)

    

    def add_param(self, label, val, esd, silent=False):

        self.blockSignals(True)
        current_rows = self.rowCount()
        self.setRowCount(current_rows + 1)

        

        name_item = QtWidgets.QTableWidgetItem(label)
        name_item.setText(label)
        name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemIsEditable)
        name_item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.setItem(current_rows, 0, name_item)
        self.name_items.append(name_item)

        val_item = QtWidgets.QTableWidgetItem(val)
        val_item.setFlags(val_item.flags() & ~QtCore.Qt.ItemIsEditable)
        val_item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.setItem(current_rows, 1, val_item)

        esd_item = QtWidgets.QTableWidgetItem(esd)
        esd_item.setFlags(esd_item.flags() & ~QtCore.Qt.ItemIsEditable)
        esd_item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.setItem(current_rows, 2, esd_item)

        
        self.setRowHeight(current_rows, 19)
        
        if not silent:
            self.select_param(current_rows)
            self.blockSignals(False)

    


    def select_param(self, ind):
        self.selectRow(ind)


