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

from PyQt6 import QtCore, QtGui, QtWidgets

from  PyQt6.QtWidgets import QMainWindow, QApplication, QInputDialog, QWidget, QLabel
from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight, HorizontalLine, NumberTextField
from hpm.widgets.PltWidget import plotWindow
from functools import partial


class DisplayPreferencesWidget(QWidget):

    apply_clicked_signal = QtCore.pyqtSignal(dict)

    def __init__(self, fields, title='Options control'):
        super().__init__()
        self.title = title
        self.opts_fields = fields
        self.setupUi()
   
    def raise_widget(self):
        self.show()
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowState.WindowMinimized | QtCore.Qt.WindowState.WindowActive)
        self.activateWindow()
        self.raise_()    

    def setupUi(self):
        self._layout = QtWidgets.QVBoxLayout()
        self.setWindowTitle(self.title)
        self.button_widget = QtWidgets.QWidget(self)
        self.button_widget.setObjectName('options_control_button_widget')
        self._button_layout = QtWidgets.QHBoxLayout()
        self._button_layout.setContentsMargins(0, 0, 0, 0)
        self._button_layout.setSpacing(15)
        self.apply_btn = FlatButton('Apply')
        self.apply_btn.clicked.connect(self.apply)
        self._button_layout.addWidget(self.apply_btn,0)
        #self._button_layout.addWidget(VerticalLine())
        self._button_layout.addSpacerItem(HorizontalSpacerItem())
        #self._button_layout.addWidget(VerticalLine())
        self.button_widget.setLayout(self._button_layout)
        
        self.parameter_widget = QtWidgets.QWidget()
        self._parameter_layout = QtWidgets.QGridLayout()
        self._parameter_layout.addWidget(QtWidgets.QLabel('Color'), 0, 1)
        
        self.opt_controls = {}
        
        i = 1
        for opt in self.opts_fields:
            desc = self.opts_fields[opt]['desc']
            self._parameter_layout.addWidget(QtWidgets.QLabel(self.opts_fields[opt]['label']), i, 0)
            if desc == 'color':
                
                cb = FlatButton()
                cb.setObjectName(opt+"_control")
                color = self.opts_fields[opt]['val']
                set_btn_color(cb,color)
                self._parameter_layout.addWidget(cb)
                self.opt_controls[opt]= {'control':cb}
                cb.clicked.connect(partial(self.color_btn_clicked, cb))
                self._parameter_layout.addItem(HorizontalSpacerItem(), i, 4)
            
            if desc == 'num':
             
                cb = NumberTextField()
                cb.setMaximumWidth(50)
                min = self.opts_fields[opt]['min']
                max = self.opts_fields[opt]['max']
                cb.setMaximum(max)
                cb.setMinimum(min)
                cb.setObjectName(opt+"_control")
                num = self.opts_fields[opt]['val']
                cb.setValue(num)
                self._parameter_layout.addWidget(cb)
                self.opt_controls[opt]= {'control':cb}
                #cb.editingFinished.connect(partial(self.color_btn_clicked, cb))
                self._parameter_layout.addItem(HorizontalSpacerItem(), i, 4)

            if desc == 'bool':
             
                cb = QtWidgets.QCheckBox()
                cb.setObjectName(opt+"_control")
                checked = self.opts_fields[opt]['val']
                cb.setChecked(checked)
                self._parameter_layout.addWidget(cb)
                self.opt_controls[opt]= {'control':cb}
                #cb.editingFinished.connect(partial(self.color_btn_clicked, cb))
                self._parameter_layout.addItem(HorizontalSpacerItem(), i, 4)

            i += 1

        self._parameter_layout.addItem(VerticalSpacerItem(), i, 0)
        self.parameter_widget.setLayout(self._parameter_layout)
        self._body_layout = QtWidgets.QHBoxLayout()
        self._body_layout.addWidget(self.parameter_widget, 0)
        self._layout.addLayout(self._body_layout)
        self._layout.addWidget(HorizontalLine())
        self._layout.addWidget(self.button_widget)
        self.setLayout(self._layout)
        
        self.style_widgets()

    def style_widgets(self):
        
        self.setStyleSheet("""
            #options_control_button_widget FlatButton {
                min-width: 70;
                max-width: 70;
            }
        """)

    def color_btn_clicked(self, btn):
        previous_color = btn.palette().color(1)
        new_color = QtWidgets.QColorDialog.getColor(previous_color, self)
        if new_color.isValid():
            color = str(new_color.name())
        else:
            color = str(previous_color.name())
    
        btn.setStyleSheet('background-color:' + color)

    

    def set_params(self,params):
        oc = self.opt_controls
        for opt in params: 
            if opt in oc:
                control=oc[opt]['control']
                val = params[opt]
                if val is not None:
                    if isinstance( control, FlatButton):
                        set_btn_color(control,val)
                    elif isinstance( control, NumberTextField):
                        control.setValue(val)
                    elif isinstance( control, QtWidgets.QCheckBox):
                        control.setChecked(val)

    def apply(self):
        params = {}
        oc = self.opt_controls
        of = self.opts_fields
        for opt in of:
            desc = of[opt]['desc']
            if desc == 'color':
                val =  oc[opt]['control'].palette().color(1)
                color = (val.red(), val.green(), val.blue())
                color = rgb2hex(*color)
                params[opt]= color
            elif desc == 'num':
                val = oc[opt]['control'].value()
                params[opt]= val
            elif desc == 'bool':
                checked = oc[opt]['control'].isChecked()
                params[opt]= checked
        self.apply_clicked_signal.emit(params)

def set_btn_color(btn, color):
    typ = type(color).__name__
    if typ == 'tuple':
        color = rgb2hex(*color)
    btn.setStyleSheet('background: '+color)

def rgb2hex(r,g,b):
    return f'#{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}'


def hex2rgb(hexcode):
    return tuple(map(ord,hexcode[1:].decode('hex')))