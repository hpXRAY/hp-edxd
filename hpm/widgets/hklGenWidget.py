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
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QItemDelegate, QTreeView


from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight, HorizontalLine

class hklGenWidget(QtWidgets.QWidget):

    color_btn_clicked = QtCore.pyqtSignal(int, QtWidgets.QWidget)
    
    show_cb_state_changed = QtCore.pyqtSignal(int, bool)
    file_dragged_in = QtCore.pyqtSignal(list)
    symmetry_edited_signal = QtCore.pyqtSignal(str)
    lattice_parameter_edited_signal = QtCore.pyqtSignal(dict)

    cell_selection_edited_signal = QtCore.pyqtSignal(int)

    def __init__(self, model):
        super(hklGenWidget, self).__init__()
        self.model = model
        self._layout = QtWidgets.QVBoxLayout()  
        self.setWindowTitle('Space Group control')
        self.button_widget = QtWidgets.QWidget(self)
        self.button_widget.setObjectName('cell_control_button_widget')
        self._button_layout = QtWidgets.QHBoxLayout()
        self._button_layout.setContentsMargins(0, 0, 0, 0)
        self._button_layout.setSpacing(6)

        self.add_btn = QtWidgets.QPushButton('New')
        self.edit_btn = QtWidgets.QPushButton('Edit')
        self.delete_btn = QtWidgets.QPushButton('Delete')
        self.clear_btn = QtWidgets.QPushButton('Clear')
        self.rois_btn = QtWidgets.QPushButton('Add ROIs')
        self.save_list_btn = QtWidgets.QPushButton('Save List')
        self.load_list_btn = QtWidgets.QPushButton('Load List')

        # 2th
        self.tth_lbl = DoubleSpinBoxAlignRight()
        self.tth_step = DoubleMultiplySpinBoxAlignRight()
        self.get_tth_btn = QtWidgets.QPushButton('Get')

        self._button_layout.addWidget(self.add_btn,0)
        #self._button_layout.addWidget(self.edit_btn,0)
        self._button_layout.addWidget(self.delete_btn,0)
        self._button_layout.addWidget(self.clear_btn,0)
        self._button_layout.addWidget(self.rois_btn,0)
        #self._button_layout.addWidget(VerticalLine())
        self._button_layout.addSpacerItem(HorizontalSpacerItem())
        #self._button_layout.addWidget(VerticalLine())
        #self._button_layout.addWidget(self.save_list_btn,0)
        #self._button_layout.addWidget(self.load_list_btn,0)
        self.button_widget.setLayout(self._button_layout)
        self._layout.addWidget(self.button_widget)


        self.symmetry_widget = QtWidgets.QWidget()
        self._symmetry_layout = QtWidgets.QVBoxLayout()
        self._symmetry_layout.setContentsMargins(0, 0, 0, 0)
        self.symmetry_opts = {}
        self.symmetry_parameters = ('CUBIC',
                                'TETRAGONAL',
                                'ORTHORHOMBIC',
                                'HEXAGONAL',
                                "TRIGONAL",
                                'RHOMBOHEDRAL',
                                'MONOCLINIC',
                                'TRICLINIC')
        for s in self.symmetry_parameters:
            o = s[0]+s[1:].lower()
            opt = QtWidgets.QRadioButton(o)
            opt.clicked.connect(partial(self.symmetry_edited_callback,o))
            self.symmetry_opts[s] = opt
            self._symmetry_layout.addWidget(opt)
        self.symmetry_widget.setLayout(self._symmetry_layout)

        self.lattice_widget = QtWidgets.QWidget()
        self._lattice_layout = QtWidgets.QVBoxLayout()
        self.spgrp_widget = QtWidgets.QWidget()
        self._spgrp_layout = QtWidgets.QHBoxLayout()
        self._spgrp_layout.setContentsMargins(0, 0, 0, 0)
        self.spgrp_lbl = QtWidgets.QLabel('Space Group:')
        self.spgrp_tb = QtWidgets.QLineEdit('P 2 3')
        self.spgrp_tb.setMaximumWidth(80)
        self.spgrp_tb.setMinimumWidth(80)
        self._spgrp_layout.addWidget(self.spgrp_lbl)
        self._spgrp_layout.addWidget(self.spgrp_tb)
        self._spgrp_layout.addSpacerItem(HorizontalSpacerItem())
        self.spgrp_widget.setLayout(self._spgrp_layout)
        self._lattice_layout.addWidget(self.spgrp_widget)
        self.lattice_opts = {}
        self.lattice_opts_steps = {}
        self.lattice_parameters = ('a','b','c',
                                'alpha',"beta",'gamma')
        default_step = 0.1
        default_decimals = 4
        for s in self.lattice_parameters:
            lattice_opt = QtWidgets.QWidget()
            _lattice_opt_layout = QtWidgets.QHBoxLayout()
            _lattice_opt_layout.setContentsMargins(0, 0, 0, 0)
            opt_lbl = QtWidgets.QLabel(s)
            opt = DoubleSpinBoxAlignRight()
            opt.setDecimals(default_decimals)
            opt.setValue(0.)
            opt.setSingleStep(default_step)
            opt.setMinimumWidth(90)
            opt.setMaximum(1000)
            opt.valueChanged.connect(partial(self.lattice_parameter_edited_callback, s))
            opt_step = DoubleMultiplySpinBoxAlignRight()
            opt_step.setDecimals(default_decimals)
            opt_step.setMinimumWidth(60)
            opt_step.setValue(default_step)
            opt_step.editingFinished.connect(partial(self.update_lattice_param_step, s))
            _lattice_opt_layout.addWidget(opt_lbl)
            _lattice_opt_layout.addWidget(opt)
            _lattice_opt_layout.addWidget(opt_step)
            lattice_opt.setLayout(_lattice_opt_layout)
            self.lattice_opts[s] = opt
            self.lattice_opts_steps[s] = opt_step
            self._lattice_layout.addWidget(lattice_opt)


        self.tth_widget = QtWidgets.QWidget()
        self._tth_layout = QtWidgets.QGridLayout()
        self._tth_layout.addWidget(QtWidgets.QLabel(u'2θ:'), 0, 0)
        self._tth_layout.addWidget(self.tth_lbl, 0, 1)
        self._tth_layout.addWidget(QtWidgets.QLabel('deg'), 0, 2)
        self._tth_layout.addWidget(self.tth_step, 0, 3)
        self._tth_layout.addWidget(self.get_tth_btn, 0, 4)
        self.tth_widget.setLayout(self._tth_layout)
        self._lattice_layout.addWidget(self.tth_widget)

        self.lattice_widget.setLayout(self._lattice_layout)

        



        self._body_layout = QtWidgets.QHBoxLayout()


        self.cell_tw = TreeView()
        #self._tm=self.model

        #self.cell_tw.setSortingEnabled(True)
        self.cell_tw.setModel(self.model)

        #self.setSelectionMode(QAbstractItemView.ExtendedSelection)  # <- optional
        selection_model = self.cell_tw.selectionModel()
        selection_model.selectionChanged.connect(self.cell_selection_changed)

        self.cell_tw.setColumnWidth(0, 50)
        self.cell_tw.setColumnWidth(1, 25)
        self.cell_tw.setColumnWidth(2, 50)
        self.cell_tw.header().hide()
        
        self.cell_tw.setMaximumWidth(175)

        self._body_layout.addWidget(self.cell_tw )


        self._body_layout.addWidget(self.symmetry_widget)
        self._body_layout.addWidget(self.lattice_widget)

        self.hkl_widget = QtWidgets.QTextEdit()
        self.hkl_widget.setMaximumWidth(175)
        self._body_layout.addWidget(self.hkl_widget)

        self._layout.addLayout(self._body_layout)

        self.setLayout(self._layout)
        
        self.style_widgets()

        self.cell_show_cbs = []
        self.cell_color_btns = []
        self.cell_roi_btns = [] #add ROIs 
        self.show_parameter_in_pattern = True
        #header_view = QtWidgets.QHeaderView(QtCore.Qt.Orientation.Horizontal, self.cell_tw)
        #self.cell_tw.setHorizontalHeader(header_view)
        
        
        #header_view.hide()
        s#elf.cell_tw.setItemDelegate(NoRectDelegate())

        
     
        self.setAcceptDrops(True) 

    def style_widgets(self):

      
        self.tth_step.setMaximumWidth(75)
        self.get_tth_btn.setMaximumWidth(75)

        self.tth_lbl.setMaximum(179.0)
        self.tth_lbl.setMinimum(1)
        self.tth_lbl.setDecimals(5)
        self.tth_step.setMaximum(180)
        self.tth_step.setMinimum(0.001)
        self.tth_step.setValue(1)
        self.tth_step.setDecimals(3)

        
        
        self.setStyleSheet("""
            #cell_control_button_widget QPushButton {
                min-width: 70;
            }
        """)

    def update_lattice_param_step(self, lattice_parameter):
        opt_step = self.lattice_opts_steps[lattice_parameter]
        value = round(opt_step.value(), 4)
        self.lattice_opts[lattice_parameter].setSingleStep(value)
        
    # ###############################################################################################
    # Now comes all the cell tw stuff
    ################################################################################################

    def cell_selection_changed(self):
        selected = self.get_selected_row()
        self.cell_selection_edited_signal.emit(selected)
    
    def get_selected_row(self):
        selected = self.cell_tw.selectionModel().selectedRows()
        try:
            row = selected[0].row()
        except IndexError:
            row = -1
        return row

    def lattice_parameter_edited_callback(self, key):
        opt = self.lattice_opts[key]
        new_val = round(opt.value(), 6)
        param = {key:new_val}
        self.lattice_parameter_edited_signal.emit(param)

    def symmetry_edited_callback(self, symmetry):
        opt = self.symmetry_opts[symmetry.upper()]
        checked = opt.isChecked()
        if checked:
            self.symmetry_edited_signal.emit(symmetry)

    def set_cell(self, params, params_enabled):
        for key in params:
            if key in self.lattice_opts:
                param = params[key]
                opt = self.lattice_opts[key]
                opt.blockSignals(True)
                opt.setValue(float(param))
                opt.blockSignals(False)
        spgrp = params['spacegroup']
        symm = params['symmetry']

        self.spgrp_tb.blockSignals(True)
        self.spgrp_tb.setText(spgrp)
        self.spgrp_tb.blockSignals(False)

        sym_opt = self.symmetry_opts[symm.upper()]
        sym_opt.blockSignals(True)
        sym_opt.setChecked(True)
        sym_opt.blockSignals(False)

        self.set_params_enabled(params_enabled)

        # this makes the button show, QTreeView quirk when using a QItemDelegate
        for row in range(0, self.model.rowCount()):
            self.cell_tw.openPersistentEditor(self.model.index(row, 1))

    def set_params_enabled(self, params):
        for key in params:
            if key in self.lattice_opts:
                param_enabled = params[key]
                self.lattice_opts[key].setEnabled(param_enabled)

    def select_cell(self, ind):
        self.cell_tw.selectRow(ind)

    def del_cell(self, ind):
        self.cell_tw.blockSignals(True)
        self.cell_tw.removeRow(ind)
        self.cell_tw.blockSignals(False)
        del self.cell_show_cbs[ind]
        del self.cell_color_btns[ind]

        if self.cell_tw.rowCount() > ind:
            self.select_cell(ind)
        else:
            self.select_cell(self.cell_tw.rowCount() - 1)
    
    def rename_cell(self, ind, name):
        name_item = self.cell_tw.item(ind, 2)
        name_item.setText(name)


    def raise_widget(self):
        self.show()
        #self.setWindowState(self.windowState() & ~QtCore.Qt.WindowState.WindowMinimized | QtCore.Qt.WindowState.WindowActive)
        self.activateWindow()
        self.raise_()


        ########################################################################################


class TreeView(QTreeView):
    """
    A simple table to demonstrate the QComboBox delegate.
    """
    def __init__(self, *args, **kwargs):
        QTreeView.__init__(self, *args, **kwargs)

        # Set the delegate for column 0 of our table
        # self.setItemDelegateForColumn(0, ButtonDelegate(self))
        #self.setItemDelegateForColumn(0, ComboDelegate(self))
        self.setItemDelegateForColumn(1, ButtonDelegate(self))

        

    
class ButtonDelegate(QItemDelegate):
    """
    A delegate that places a fully functioning QPushButton in every
    cell of the column to which it's applied
    """
    def __init__(self, parent):
        QItemDelegate.__init__(self, parent)
        self.parent = parent
        
    def createEditor(self, parent, option, index):
        btn = QtWidgets.QPushButton(str(index.data()), parent)
        btn.clicked.connect(self.currentIndexChanged)
        return btn
        
    def setEditorData(self, editor, index):
        editor.blockSignals(True)
        color = str(index.model().data(index))
        editor.setStyleSheet("background-color: "+color )
        editor.setText('')
        editor.blockSignals(False)
        
    def setModelData(self, editor, model, index):
        previous_color = editor.palette().color(QPalette.ColorRole.Button)
        new_color = QtWidgets.QColorDialog.getColor(previous_color, self.parent)
        if new_color.isValid():
            color = str(new_color.name())
        else:
            color = str(previous_color.name())
        editor.setStyleSheet('background-color:' + color)
        model.setData(index, str(color))
        
    def currentIndexChanged(self):
        self.commitData.emit(self.sender())

    

class ComboDelegate(QItemDelegate):
    """
    A delegate that places a fully functioning QComboBox in every
    cell of the column to which it's applied
    """
    def __init__(self, parent):

        QItemDelegate.__init__(self, parent)
        
    def createEditor(self, parent, option, index):
        combo = QtWidgets.QComboBox(parent)
        li = []
        li.append("Zero")
        li.append("One")
        li.append("Two")
        li.append("Three")
        li.append("Four")
        li.append("Five")
        combo.addItems(li)
        #self.connect(combo, QtCore.pyqtSignal("currentIndexChanged(int)"), self, QtCore.SLOT("currentIndexChanged()"))
        return combo
        
    def setEditorData(self, editor, index):
        editor.blockSignals(True)
        editor.setCurrentIndex(int(index.model().data(index)))
        editor.blockSignals(False)
        
    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentIndex())
        
    def currentIndexChanged(self):
        self.commitData.emit(self.sender())