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
from PyQt6 import QtWidgets, QtCore, QtGui
import copy
import pyqtgraph as pg
from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight


class SaveFileWidget(QtWidgets.QWidget):
    '''
        Widget for saving files, made to mimic the functionality of the NDFileTIFF.adl screen used with Area Detector
        "https://cars9.uchicago.edu/software/epics/NDFileTIFF.html"

        Readback value "RBV" widgets:
            self.path_rbv               QLabel
            self.path_exists_rbv        QLabel
            self.name_rbv               QLabel
            self.next_n_rbv             QLabel
            self.autoincrement_rbv      QLabel
            self.format_rbv             QLabel
            self.last_filename_rbv      QLabel
            self.save_rbv               QLabel
            self.autosave_rbv           QLabel
            self.write_message_rbv      QLabel

        Setpoint value "VAL" widgets:
            self.path_val               QLineEdit
            self.name_val               QLineEdit
            self.next_n_val             QLineEdit
            self.autoincrement_val      QComboBox
            self.format_val             QLineEdit
            self.save_val               QPushButton
            self.autosave_val           QComboBox
    '''
    widget_closed = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self._layout = QtWidgets.QVBoxLayout()  
        self.setWindowTitle('Save File')
        self.resize(550,300)
        
        self.grid_layout = QtWidgets.QGridLayout()
        self.grid_layout.setContentsMargins(0,0,0,0)

        # Field Names

        lbl = QtWidgets.QLabel("File path", self)
        lbl.setAlignment(Qt.AlignRight)
        self.grid_layout.addWidget(lbl,1,0)

        lbl = QtWidgets.QLabel("File name", self)
        lbl.setAlignment(Qt.AlignRight)
        self.grid_layout.addWidget(lbl,3,0)

        lbl = QtWidgets.QLabel("Next file #", self)
        lbl.setAlignment(Qt.AlignRight)
        self.grid_layout.addWidget(lbl,4,0)

        lbl = QtWidgets.QLabel('Auto Increment', self)
        lbl.setAlignment(Qt.AlignRight)
        self.grid_layout.addWidget(lbl,5,0)

        lbl = QtWidgets.QLabel('Filename format', self)
        lbl.setAlignment(Qt.AlignRight)
        self.grid_layout.addWidget(lbl,7,0)

        lbl = QtWidgets.QLabel('Last filename', self)
        lbl.setAlignment(Qt.AlignRight)
        self.grid_layout.addWidget(lbl,8,0)

        lbl = QtWidgets.QLabel('Save file', self)
        lbl.setAlignment(Qt.AlignRight)
        self.grid_layout.addWidget(lbl,10,0)

        lbl = QtWidgets.QLabel('Write status', self)
        lbl.setAlignment(Qt.AlignRight)
        self.grid_layout.addWidget(lbl,11,0)
    

        # Field Values
        self.path_rbv_widget = QtWidgets.QWidget()
        self.path_rbv_widget_layout = QtWidgets.QHBoxLayout()
        self.path_rbv_widget_layout.setContentsMargins(0,0,0,0)

        self.path_rbv = QtWidgets.QLabel('/home/epics/scratch/', self)
        self.path_rbv.setStyleSheet("QLabel {color: #73DFFF; }")
        self.path_rbv_widget_layout.addWidget(self.path_rbv)
        self.path_rbv_widget_layout.addSpacerItem(HorizontalSpacerItem())
        self.path_rbv_widget_layout.addWidget(QtWidgets.QLabel('Exists: '))
        
        self.path_exists_rbv = QtWidgets.QLabel('Yes', self)
        self.path_exists_rbv.setStyleSheet("QLabel {color: #00CD00; }")
        self.path_rbv_widget_layout.addWidget(self.path_exists_rbv)
        self.path_rbv_widget.setLayout(self.path_rbv_widget_layout)
        self.grid_layout.addWidget(self.path_rbv_widget,0,1)

        
        self.path_val = QtWidgets.QLineEdit('/home/epics/scratch')
        self.grid_layout.addWidget(self.path_val,1,1)

        self.name_rbv = QtWidgets.QLabel('test', self)
        self.name_rbv.setStyleSheet("QLabel {color: #73DFFF; }")
        self.grid_layout.addWidget(self.name_rbv,2,1)
        self.name_val = QtWidgets.QLineEdit('test')
        self.grid_layout.addWidget(self.name_val,3,1)
        
        
        self.next_file_widget = QtWidgets.QWidget()
        self.next_file_widget_layout = QtWidgets.QHBoxLayout()
        self.next_file_widget_layout.setContentsMargins(0,0,0,0)
        self.next_n_val = QtWidgets.QLineEdit('510')
        self.next_file_widget_layout.addWidget(self.next_n_val)
        self.next_n_rbv = QtWidgets.QLabel('510', self)
        self.next_n_rbv.setStyleSheet("QLabel {color: #73DFFF; }")
        self.next_file_widget_layout.addWidget(self.next_n_rbv )
        self.next_file_widget_layout.addSpacerItem(HorizontalSpacerItem())
        self.next_file_widget.setLayout(self.next_file_widget_layout)
        self.grid_layout.addWidget(self.next_file_widget,4,1)

        self.autoincrement_widget = QtWidgets.QWidget()
        self.autoincrement_layout = QtWidgets.QHBoxLayout()
        self.autoincrement_layout.setContentsMargins(0,0,0,0)
        self.autoincrement_val = QtWidgets.QComboBox()
        self.autoincrement_val.addItems(['Yes', 'No'])
        self.autoincrement_layout.addWidget(self.autoincrement_val)
        self.autoincrement_rbv = QtWidgets.QLabel('Yes', self)
        self.autoincrement_rbv.setStyleSheet("QLabel {color: #73DFFF; }")
        self.autoincrement_layout.addWidget(self.autoincrement_rbv)
        self.autoincrement_widget.setLayout(self.autoincrement_layout)

        self.grid_layout.addWidget(self.autoincrement_widget,5,1)

        self.format_rbv = QtWidgets.QLabel('%s%s_%3.3d.tif')
        self.format_rbv.setStyleSheet("QLabel {color: #73DFFF; }")
        self.grid_layout.addWidget(self.format_rbv,6,1)

        self.format_widget = QtWidgets.QWidget()
        self.format_layout = QtWidgets.QHBoxLayout()
        self.format_layout.setContentsMargins(0,0,0,0)
        self.format_val = QtWidgets.QLineEdit('%s%s_%3.3d.tif')
        self.format_layout.addWidget(self.format_val )
        self.format_layout.addWidget(QtWidgets.QLabel('Example: %s%s_%3.3d.tif'))
        self.format_layout.addSpacerItem(HorizontalSpacerItem())
        self.format_widget.setLayout(self.format_layout)
        self.grid_layout.addWidget(self.format_widget ,7,1)

        self.last_filename_rbv = QtWidgets.QLabel('/home/epics/scratch/test_509.tif')
        self.last_filename_rbv.setStyleSheet("QLabel {color: #73DFFF; }")
        self.grid_layout.addWidget(self.last_filename_rbv,8,1)

        self.save_rbv = QtWidgets.QLabel('Writing')
        self.save_rbv.setStyleSheet("QLabel {color: #FFFF00; }")
        self.grid_layout.addWidget(self.save_rbv,9,1)


        self.save_widget = QtWidgets.QWidget()
        self.save_layout = QtWidgets.QHBoxLayout()
        self.save_layout.setContentsMargins(0,0,0,0)
        self.save_val = QtWidgets.QPushButton('Save')
        self.save_val.setMinimumWidth(90)
        self.save_layout.addWidget(self.save_val )
        self.save_layout.addSpacerItem(HorizontalSpacerItem())
        self.save_layout.addWidget(QtWidgets.QLabel('Autosave'))
        self.autosave_val = QtWidgets.QComboBox()
        self.autosave_val.addItems(['Yes', 'No'])
        self.save_layout.addWidget(self.autosave_val )
        self.autosave_rbv = QtWidgets.QLabel('Yes')
        self.autosave_rbv.setStyleSheet("QLabel {color: #73DFFF; }")
        self.save_layout.addWidget(self.autosave_rbv )
        self.save_widget.setLayout(self.save_layout)

        self.grid_layout.addWidget(self.save_widget,10,1)

        self.write_message_rbv = QtWidgets.QLabel('Write OK')
        self.write_message_rbv.setStyleSheet("QLabel {color: #00CD00; }")
        self.grid_layout.addWidget(self.write_message_rbv,11,1)


        self._layout.addLayout(self.grid_layout)
        self._layout.addSpacerItem(VerticalSpacerItem())

        self.setLayout(self._layout)

        self.style_widgets()
       
    

    def style_widgets(self):
       
        self.setStyleSheet("""
            #roi_control_button_widget FlatButton {
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



