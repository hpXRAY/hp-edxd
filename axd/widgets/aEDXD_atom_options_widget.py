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



""" 
    Ross Hrubiak
"""

from functools import partial
from PyQt6 import QtWidgets, QtCore, QtGui
import copy

from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight, HorizontalLine
from axd.models.aEDXD_atomic_parameters import aEDXDAtomicParameters


class aEDXDAtomWidget(QtWidgets.QWidget):

    widget_closed = QtCore.pyqtSignal()
    fract_item_changed_signal = QtCore.pyqtSignal(int, float)
  
    def __init__(self):
        super().__init__()
        self._layout = QtWidgets.QVBoxLayout()  
        self.setWindowTitle('Atoms control')
        self.button_widget = QtWidgets.QWidget(self)
        self.button_widget.setObjectName('atom_control_button_widget')
        self._button_layout = QtWidgets.QHBoxLayout()
        self._button_layout.setContentsMargins(0, 0, 0, 0)
        self._button_layout.setSpacing(15)

        self.add_btn = FlatButton('Add')
        self.delete_btn = FlatButton('Delete')
        self.clear_btn = FlatButton('Clear')

        self._button_layout.addWidget(self.add_btn)
        self._button_layout.addWidget(self.delete_btn)
        self._button_layout.addWidget(self.clear_btn)
        self._button_layout.addSpacerItem(HorizontalSpacerItem())
        '''
        self._button_layout.addWidget(VerticalLine())
        
        self._button_layout.addWidget(VerticalLine())
        self._button_layout.addWidget(self.save_btn)
        '''
        
        self.button_widget.setLayout(self._button_layout)
        self._layout.addWidget(self.button_widget)
        self._body_layout = QtWidgets.QHBoxLayout()
        self.atom_tw = ListTableWidget(columns=4)
        self._body_layout.addWidget(self.atom_tw, 10)
        self._layout.addLayout(self._body_layout)

        self.button_2_widget = QtWidgets.QWidget(self)
        self.button_2_widget.setObjectName('options_control_button_2_widget')
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
        
        self.name_items = []
        self.index_items = []
        self.fract_items = []
        self.show_parameter_in_pattern = True
        header_view = QtWidgets.QHeaderView(QtCore.Qt.Horizontal, self.atom_tw)
        self.atom_tw.setHorizontalHeader(header_view)
        header_view.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        
        header_view.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header_view.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        header_view.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self.default_header = ['#', 'Atom','Note', 'Fraction', 'ind']
        self.header = copy.deepcopy(self.default_header)
        self.atom_tw.setHorizontalHeaderLabels(self.header)
        
        self.atom_tw.setItemDelegate(NoRectDelegate())
        
        self.ap = aEDXDAtomicParameters()
        self.sq_pars = []
        self.make_connections()

    def make_connections(self):
        self.atom_tw.itemChanged.connect(self.fract_item_changed)
        

    def style_widgets(self):
        self.atom_tw.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        self.atom_tw.setMinimumWidth(400)
        self.atom_tw.setMinimumHeight(110)
        self.setStyleSheet("""
            #atom_control_button_widget FlatButton {
                min-width: 70;
                max-width: 70;
            }
            #options_control_button_2_widget FlatButton {
                min-width: 70;
                max-width: 70;
            }
        """)


    ################################################################################################
    # Now comes all the atom tw stuff
    ################################################################################################

    def add_atom(self, use, name, note,fract, ind, silent=False):
        self.atom_tw.blockSignals(True)
        current_rows = self.atom_tw.rowCount()
        self.atom_tw.setRowCount(current_rows + 1)

        index_item = QtWidgets.QTableWidgetItem(str(ind))
        #index_item.setText(str(ind))
        index_item.setFlags(index_item.flags() & ~QtCore.Qt.ItemIsEditable)
        index_item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.atom_tw.setItem(current_rows, 0, index_item)
        self.index_items.append(index_item)

        name_item = QtWidgets.QTableWidgetItem(name)
        name_item.setText(name)
        name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemIsEditable)
        name_item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.atom_tw.setItem(current_rows, 1, name_item)
        self.name_items.append(name_item)

        note_item = QtWidgets.QTableWidgetItem(note)
        note_item.setFlags(note_item.flags() & ~QtCore.Qt.ItemIsEditable)
        note_item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.atom_tw.setItem(current_rows, 2, note_item)

        fract_item = QtWidgets.QTableWidgetItem(str(fract))
        self.fract_items.append(fract_item)
        #fract_item.setFlags(fract_item.flags() & ~QtCore.Qt.ItemIsEditable)
        fract_item.setTextAlignment(QtCore.Qt.AlignHCenter| QtCore.Qt.AlignVCenter)
        self.atom_tw.setItem(current_rows, 3, fract_item)

        self.atom_tw.setColumnWidth(0, 25)
        #self.atom_tw.setColumnWidth(1, 20)
        self.atom_tw.setRowHeight(current_rows, 25)
        
        if not silent:
            self.select_atom(current_rows)
            self.atom_tw.blockSignals(False)

    def closeEvent(self, event):
        # Overrides close event to let controller know that widget was closed by user
        self.widget_closed.emit()
        
    def select_atom(self, ind):
        self.atom_tw.selectRow(ind)
        
    def get_selected_atom_row(self):
        selected = self.atom_tw.selectionModel().selectedRows()
        try:
            row = selected[0].row()
        except IndexError:
            row = -1
        return row

    def fract_item_changed(self, nameitem):
        
        selected = self.get_selected_atom_row()
        self.fract_item_changed_signal.emit(selected, float(self.fract_items[selected].text()))

    def raise_widget(self):
        self.show()
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
        self.activateWindow()
        self.raise_()

    def del_atom(self, ind):
        if ind > -1:
            self.atom_tw.blockSignals(True)
            self.atom_tw.removeRow(ind)
            
            self.atom_tw.blockSignals(False)
            
            del self.name_items[ind]
            del self.index_items[ind]
            del self.fract_items[ind]
        

    
    


class plotFitWindow(QtWidgets.QWidget):
    widget_closed = QtCore.pyqtSignal()
    def __init__(self):
        super().__init__()

        self._layout = QtWidgets.QVBoxLayout()  
        self._layout.setContentsMargins(0,0,0,0)
        self.setWindowTitle('Fit plot')
        self.win = pg.GraphicsWindow(title="Fit plot")
        self.win.resize(400,400)
        self.win.setWindowTitle('Peak fit plot')
        self.fitPlots = self.win.addPlot(title="Peak fit")
        self.win.setBackground(background=(255,255,255))
        self.fitPlot = self.fitPlots.plot([],[], 
                        pen=(0,0,0), name="Gaussian peak fit", 
                        antialias=True)
        self.dataPlot =  self.fitPlots.plot([],[], 
                        pen = None, symbolBrush=(255,0,0), 
                        symbolPen='k', name="Data",antialias=True)
        self.fitPlots.setLabel('left','counts')
        self.fitPlots.setLabel('bottom', 'channel')

        self._layout.addWidget(self.win)
        self.setLayout(self._layout)
        self.resize(400,400)

    def closeEvent(self, event):
        # Overrides close event to let controller know that widget was closed by user
        self.widget_closed.emit()

    def raise_widget(self):
        self.show()
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
        self.activateWindow()
        self.raise_()

    def set_data(self,x_fit=[],fit=[],label='',x=[],counts=[],unit='',unit_=''):
        self.fitPlot.setData(x_fit,fit)
        self.fitPlots.setTitle(label)
        self.dataPlot.setData(x,counts)
        self.fitPlots.setLabel('bottom', unit+' ('+unit_+')')
        self.fitPlots.getViewBox().enableAutoRange()
        