import json
from PyQt6.QtCore import pyqtSignal, Qt
import os
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QWidget
from functools import partial
import copy
from axd import data_path

class spacerWidget(QtWidgets.QWidget):
    def __init__(self, width, height):
        super( ).__init__()
        spacer_widget_layout = QtWidgets.QVBoxLayout()
        spacer_widget_layout.setContentsMargins(0,0,0,0)
        spacer_widget_layout.addSpacerItem(QtWidgets.QSpacerItem(width, height, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))
        self.setLayout(spacer_widget_layout)

class PeriodicTable(QWidget):
    element_clicked_signal = pyqtSignal(str)
    def __init__(self, options):
        super().__init__()
        elements_file = os.path.join(data_path,"elements.json")
        with open(elements_file) as f:
            elements_data = json.load(f)
            f.close()
        self.elements_data = {}
        for e in elements_data:
            symbol = e['symbol']
            self.elements_data[symbol] = e
        self.setUI(options)

    def setUI(self, options):
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(2)
        symbols = [
        "H","","","","","","","","","","","","","","","","","He",
        "Li","Be","","","","","","","","","","","B","C","N","O","F","Ne",
        "Na","Mg","","","","","","","","","","","Al","Si","P","S","Cl","Ar"
        ,"K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge"
        ,"As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd",
        "Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Hf","Ta","W",
        "Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra",
        "Ac","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg"]
        lan_act_symbols = ["Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho",
        "Er","Tm","Yb","Lu", "Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es",
        "Fm", "Md","No","Lr"]
        positions = [(i,j) for i in range(9) for j in range(18)]
        for position, name in zip(positions, symbols):
            if name == '':
                continue
            disabled = len(options) and not name in options
            button = self.make_atom_button(name, disabled=disabled)
            grid.addWidget(button, *position)     
        grid.addWidget(spacerWidget(35,15),7,0)
        lan_act_pos = [(i,j) for i in range(2) for j in range(14)]
        for position, name in zip(lan_act_pos, lan_act_symbols):
            if name == '':
                continue
            disabled = len(options) and not name in options
            button = self.make_atom_button(name, disabled=disabled)
            grid.addWidget(button, position[0]+8, position[1])     
        self.setLayout(grid)

    def make_atom_button(self, name, max_width=35,max_height=35, disabled = False):
        button = QtWidgets.QPushButton(name,self)
        button.setToolTip(self.elements_data[name]['name'])
        button.setMaximumWidth(35)
        button.setMaximumHeight(35)
        if disabled:
            button.setEnabled(False)
        else:
            button.clicked.connect(partial( self.button_clicked_callback, button))
        return button
       
    def button_clicked_callback(self,button):
        e = button.text()
        self.element_clicked_signal.emit(e)
    
############################################################
class PeriodicTableDialog(QtWidgets.QDialog):
    def __init__(self, options):
        super( ).__init__()
        self.options = options
        self._layout = QtWidgets.QVBoxLayout()  
        self.table_widget = PeriodicTable(options)
        self._layout.addWidget(self.table_widget)
        self.setLayout(self._layout)
        self.setWindowTitle('Choose an atom')
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint )
        self.ok = ''
        self.table_widget.element_clicked_signal.connect(self.button_press)
        
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