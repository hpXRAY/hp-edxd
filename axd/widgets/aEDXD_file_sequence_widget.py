
import json
from PyQt6.QtCore import pyqtSignal, Qt

import os
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QWidget

from functools import partial
import copy

class FileSequenceWidget(QtWidgets.QWidget):
    sequence_accepted_signal = pyqtSignal(str)
    def __init__(self, options=[]):
        super( ).__init__()
        self._layout = QtWidgets.QVBoxLayout()
        



############################################################
class FileSequenceDialog(QtWidgets.QDialog):
    def __init__(self, options=[]):
        super( ).__init__()
        self.options = options
        self._layout = QtWidgets.QVBoxLayout()  
        self.table_widget = FileSequenceWidget(self)
        self._layout.addWidget(self.table_widget)
        self.setLayout(self._layout)
        self.setWindowTitle('Choose an atom')
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint )
        self.ok = ''
        self.table_widget.sequence_accepted_signal.connect(self.button_press)
        
    @classmethod
    def showDialog(cls,options = []):
        dialog = cls( options)
        dialog.exec_()
        ok = copy.deepcopy(dialog.ok)
        dialog.deleteLater()
        return ok

 

   ############################################################
    def button_press(self, element):
        self.ok = element

        self.close()