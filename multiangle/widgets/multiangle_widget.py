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
# Copyright (C) 2018-2024 ANL, Lemont, USA



from functools import partial
from PyQt6 import QtWidgets, QtCore, QtGui
import copy
from multiangle.utilities.utilities import optimize_tth, tth_e_to_d, d_to_q
import time
from multiangle.models.multiangle_model import multiangleModel
from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight, HorizontalLine

from axd.widgets.aEDXD_widget import customWidget

from utilities.HelperModule import calculate_color
import numpy as np

class parameterWidget(QtWidgets.QWidget):
    sequence_changed_signal = QtCore.pyqtSignal(dict)
    widget_closed = QtCore.pyqtSignal()
    def __init__(self):
        super().__init__()
        self.setWindowTitle('New multiangle setup')
        self._layout = QtWidgets.QVBoxLayout()
        self._parameter_layout = QtWidgets.QGridLayout()
        self.q_low_sb = DoubleSpinBoxAlignRight()
        self.E_low_sb = DoubleSpinBoxAlignRight()
        self.q_high_sb = DoubleSpinBoxAlignRight()
        self.E_high_sb = DoubleSpinBoxAlignRight()
        self.overlap_sb = DoubleSpinBoxAlignRight()
        self.q_low_sb.setMinimumWidth(100)
        self.E_low_sb.setMinimumWidth(100)
        self.q_high_sb.setMinimumWidth(100)
        self.E_high_sb.setMinimumWidth(100)
        self.overlap_sb.setMinimumWidth(100)

        self.q_low_sb.setMaximum(50)
        self.q_low_sb.setMinimum(0.001)
        self.q_low_sb.setValue(1)
        self.E_low_sb.setMaximum(1000)
        self.E_low_sb.setMinimum(1)
        self.E_low_sb.setValue(34)
        self.q_high_sb.setMaximum(50)
        self.q_high_sb.setMinimum(0.001)
        self.q_high_sb.setValue(15)
        self.E_low_sb.setMaximum(1000)
        self.E_low_sb.setMinimum(1)
        self.E_high_sb.setValue(68)
        self.overlap_sb.setMaximum(90)
        self.overlap_sb.setMinimum(10)
        self.overlap_sb.setValue(75)

        self._parameter_layout.addWidget(QtWidgets.QLabel('Low'), 0, 1)
        self._parameter_layout.addWidget(QtWidgets.QLabel('High'), 0, 2)
        self._parameter_layout.addWidget(QtWidgets.QLabel('q:'), 1, 0)
        self._parameter_layout.addWidget(QtWidgets.QLabel('E:'), 2, 0)
        self._parameter_layout.addWidget(QtWidgets.QLabel('Overlap %:'), 3, 0)
        self._parameter_layout.addWidget(QtWidgets.QLabel(f'\N{LATIN CAPITAL LETTER A WITH RING ABOVE}\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}'), 1, 3)
        self._parameter_layout.addWidget(QtWidgets.QLabel('KeV'), 2, 3)

        self._parameter_layout.addWidget(self.q_low_sb, 1, 1)
        self._parameter_layout.addWidget(self.q_high_sb, 1, 2)
        self._parameter_layout.addWidget(self.E_low_sb, 2, 1)
        self._parameter_layout.addWidget(self.E_high_sb, 2, 2)
        self._parameter_layout.addWidget(self.overlap_sb, 3, 1)

        self.apply_btn = FlatButton('Apply')
        self.apply_btn.setMinimumWidth(75)
        self.overlap_lbl = QtWidgets.QLabel('')

        self._parameter_layout.addWidget(self.apply_btn, 4, 1, 1, 1)
        self._parameter_layout.addWidget(self.overlap_lbl, 3, 2, 1, 1)
        
        self._layout.addLayout(self._parameter_layout)
        self._layout.addSpacerItem(VerticalSpacerItem())
        
        self.setLayout(self._layout)
        self.apply_btn.clicked.connect(self.compute_2th)

    def compute_2th(self):
        q_low  =self.q_low_sb.value()
        e_low  =self.E_low_sb.value()
        q_high =self.q_high_sb.value()
        e_high =self.E_high_sb.value()
        overlap_fract = self.overlap_sb.value() * 0.01

        start_time=time.time()
        n, overlap_fract, seq = optimize_tth(q_low,q_high,e_low,e_high,overlap_fract)
        timer = time.time() - start_time
        t = []
        for tth in seq:
            t.append(tth['tth'])
        d = {}
        d['n']=n
        d['overlap_fract'] = overlap_fract
        d['seq'] = seq
        d['tth'] = t
        low_q = seq[0]['q_low']
        high_q = seq[-1]['q_high']
        d['q_low']= low_q
        d['q_high']= high_q

     
        self.overlap_lbl.setText('n: ' + str(int(n)) + '\nOverlap %: ' +str(round(overlap_fract*100,3)))
        self.sequence_changed_signal.emit(d)

    def raise_widget(self):
        self.show()
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowState.WindowMinimized | QtCore.Qt.WindowState.WindowActive)
        self.activateWindow()
        self.raise_()
        

class multiangleWidget(QtWidgets.QWidget):

    widget_closed = QtCore.pyqtSignal()
    fract_item_changed_signal = QtCore.pyqtSignal(int, float)
  
    def __init__(self,multiangle_model:multiangleModel):
        super().__init__()
        self.multiangle_model: multiangleModel
        self.tth_tv: QtWidgets.QTreeView
        self.multiangle_model = multiangle_model
        self.parameter_widget = parameterWidget()
        self._layout = QtWidgets.QVBoxLayout()  
        self.setWindowTitle('Multiangle control')
        self.button_widget = QtWidgets.QWidget(self)
        self.button_widget.setObjectName('control_button_widget')
        self._button_layout = QtWidgets.QHBoxLayout()
        self._button_layout.setContentsMargins(0, 0, 0, 0)
        self._button_layout.setSpacing(15)
        self.new_btn = FlatButton('Setup')
        self.load_btn = FlatButton('Load')
        self.save_btn = FlatButton('Save')
        self.delete_btn = FlatButton('Delete')
        self.clear_btn = FlatButton('Clear')
        self.add_btn = FlatButton('Add')
        self.plot_btn = FlatButton('Plot')
        self._button_layout.addWidget(self.new_btn)
        self._button_layout.addWidget(self.load_btn)
        self._button_layout.addWidget(self.save_btn)
        self._button_layout.addWidget(self.add_btn)
        self._button_layout.addWidget(self.delete_btn)
        self._button_layout.addWidget(self.clear_btn)
        self._button_layout.addWidget(self.plot_btn)
        
        self._button_layout.addSpacerItem(HorizontalSpacerItem())
        
        self.button_widget.setLayout(self._button_layout)
        self._layout.addWidget(self.button_widget)
        self._body_layout = QtWidgets.QHBoxLayout()
        self.tth_tv = QtWidgets.QTreeView()
        self.tth_tv.sortByColumn(0, QtCore.Qt.SortOrder.AscendingOrder)
        self.tth_tv.setSortingEnabled(True)
        self.tth_tv.setModel(self.multiangle_model)

        self._body_layout.addWidget(self.tth_tv, 10)

        self.make_beam_parameter_widget()

        self._body_layout.addWidget(self.beam_parameter_widget)

        self._layout.addLayout(self._body_layout)
        self.button_2_widget = QtWidgets.QWidget(self)
        self.button_2_widget.setObjectName('options_control_button_2_widget')
        self._button_2_layout = QtWidgets.QHBoxLayout()
        self._button_2_layout.setContentsMargins(0, 0, 0, 0)
        self._button_2_layout.setSpacing(15)
        self.iterations_lbl = QtWidgets.QLabel('Iterations:')
        self._button_2_layout.addWidget(self.iterations_lbl)
        self.iterations_control = QtWidgets.QLineEdit()
        self.iterations_control.setText('1')
        self._button_2_layout.addWidget(self.iterations_control)
        self.run_btn = FlatButton('Run')
        self._button_2_layout.addWidget(self.run_btn,0)
        self.stop_btn = FlatButton('Stop')
        self._button_2_layout.addWidget(self.stop_btn,0)
        self.test_btn = FlatButton('Test')
        self._button_2_layout.addWidget(self.test_btn,0)
        self.setup_btn = FlatButton('Setup')
        self._button_2_layout.addWidget(self.setup_btn,0)
        
        self._button_2_layout.addSpacerItem(HorizontalSpacerItem())
     
        self.button_2_widget.setLayout(self._button_2_layout)
        self._layout.addWidget(HorizontalLine())
        self._layout.addWidget(self.button_2_widget)

        self.setLayout(self._layout)
        self.style_widgets()
        
        self.show_parameter_in_pattern = True
        columns = self.multiangle_model.columnCount()
        self.tth_tv.resizeColumnToContents(0)
        header = self.tth_tv.header()
        for ind in range(columns):
            header.setSectionResizeMode(ind, QtWidgets.QHeaderView.ResizeMode.Stretch)

        self.make_connections()


    def make_beam_parameter_widget(self):
        
        self.beam_parameter_widget = QtWidgets.QWidget()
        self._beam_layout = QtWidgets.QVBoxLayout()
        self._beam_parameter_layout = QtWidgets.QGridLayout()

        self.tip_size_sb = DoubleSpinBoxAlignRight()
        self.tip_size_sb.setMinimumWidth(100)
        self.tip_size_sb.setMaximum(1000)
        self.tip_size_sb.setMinimum(5e-3)
        self.tip_size_sb.setValue(0.05)
        self.tip_size_sb.setDecimals(3)
        self.tip_size_sb.setSingleStep(0.005)

        self.tip_distance_sb = DoubleSpinBoxAlignRight()
        self.tip_distance_sb.setMinimumWidth(100)
        self.tip_distance_sb.setMaximum(10000)
        self.tip_distance_sb.setMinimum(1e-3)
        self.tip_distance_sb.setValue(60)

        self.det_distance_sb = DoubleSpinBoxAlignRight()
        self.det_distance_sb.setMinimumWidth(100)
        self.det_distance_sb.setMaximum(10000)
        self.det_distance_sb.setMinimum(1e-3)
        self.det_distance_sb.setValue(600)
        
        self._beam_parameter_layout.addWidget(QtWidgets.QLabel('Beam alignment parameters'), 0, 0 ,1,2)
        self._beam_parameter_layout.addWidget(QtWidgets.QLabel('Tip size'), 1, 0)
        self._beam_parameter_layout.addWidget(QtWidgets.QLabel('Tip distance'), 2, 0)
        self._beam_parameter_layout.addWidget(QtWidgets.QLabel('Detector distance'), 3, 0)
 
        self._beam_parameter_layout.addWidget(self.tip_size_sb, 1, 1)
        self._beam_parameter_layout.addWidget(self.tip_distance_sb, 2, 1)
        self._beam_parameter_layout.addWidget(self.det_distance_sb, 3, 1)
        self._beam_parameter_layout.addWidget(QtWidgets.QLabel('D0: Collimation depth at the sample center (mm)'), 5, 0 ,1,2)
        self._beam_parameter_layout.addWidget(QtWidgets.QLabel('D1: Maximum collimation depth (mm)'), 6, 0 ,1,2)
        self._beam_layout.addLayout(self._beam_parameter_layout)
        self._beam_layout.addSpacerItem(VerticalSpacerItem())
        self.beam_parameter_widget.setLayout(self._beam_layout)
        

    def make_connections(self):
      
        self.new_btn.clicked.connect(self.new_callback)

    def new_callback(self):
        self.parameter_widget.raise_widget()

    def style_widgets(self):
        self.tth_tv.setSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        self.tth_tv.setMinimumWidth(600)
        self.tth_tv.setMinimumHeight(110)
        self.setStyleSheet("""
            #control_button_widget FlatButton {
                min-width: 70;
                max-width: 70;
            }
            #options_control_button_2_widget FlatButton {
                min-width: 70;
                max-width: 70;
            }
            
        """)


    ################################################################################################
    # Now comes all the tth tw stuff
    ################################################################################################


    def closeEvent(self, QCloseEvent, *event):
        # Overrides close event to let controller know that widget was closed by user
        self.widget_closed.emit()
        
    def select_tth(self, ind): 
        
        self.tth_tv.selectRow(ind)
        
    def get_selected_tth_row(self):
        selected = self.tth_tv.selectionModel().selectedRows()
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

class QCoverageWidget(customWidget):
    def __init__(self, model):
        plot_widget_parameters = 'q space coverage',f'2\N{GREEK SMALL LETTER THETA}', 'q' 
        super().__init__(plot_widget_parameters)
        self.model = model
        self.setWindowTitle('q space coverage')
        self.make_E_widgets()
        self.add_button_widget_item(self.E_widget)
        self.add_button_widget_spacer()
        self.model.itemChanged.connect(self.update)

    def update(self, *karg, **kwarg):
        if not self.model.block_update:
            tth = self.model.get_tth()
            self.plot_tth(tth)
        
    def plot_tth(self,tth):
        self.fig.clear()
        if len(tth):
            for i, t in enumerate(tth):
                color  = calculate_color(i) 
                x, y = self.make_plot(t)
                self.fig.add_line_plot(x,y,color,2)
                self.fig.set_plot_label_color(color,i)

    def make_plot(self,tth):
        E = []
        E.append(self.E_low_sb.value())
        E.append(self.E_high_sb.value())
        d = [tth_e_to_d(tth,E[0]), tth_e_to_d(tth,E[1])]
        q = [d_to_q(d[0]),d_to_q(d[1])]
        x = np.asarray(q)
        y = np.asarray([tth,tth])
        return x, y

    def make_E_widgets(self):
        self.E_widget = QtWidgets.QWidget()
        self._layout = QtWidgets.QVBoxLayout()
        self._parameter_layout = QtWidgets.QGridLayout()
        self.E_low_sb = DoubleSpinBoxAlignRight()
        self.E_high_sb = DoubleSpinBoxAlignRight()
        self.E_low_sb.setMinimumWidth(100)
        self.E_high_sb.setMinimumWidth(100)
        self.E_low_sb.setMaximum(1000)
        self.E_low_sb.setMinimum(1)
        self.E_low_sb.setValue(34)
        self.E_low_sb.setMaximum(1000)
        self.E_low_sb.setMinimum(1)
        self.E_high_sb.setValue(68)
        self._parameter_layout.addWidget(QtWidgets.QLabel('E'), 1, 0)
        self._parameter_layout.addWidget(QtWidgets.QLabel('Low'), 0, 1)
        self._parameter_layout.addWidget(QtWidgets.QLabel('High'), 0, 2)
        self._parameter_layout.addWidget(QtWidgets.QLabel('KeV'), 1, 3)
        self._parameter_layout.addWidget(self.E_low_sb, 1, 1)
        self._parameter_layout.addWidget(self.E_high_sb, 1, 2)
        self._layout.addLayout(self._parameter_layout)
        self._layout.addSpacerItem(VerticalSpacerItem())
        self.E_widget.setLayout(self._layout)
        self.E_low_sb.valueChanged.connect(self.update)
        self.E_high_sb.valueChanged.connect(self.update)