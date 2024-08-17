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


class RoiPreferencesWidget(QWidget):

    apply_clicked_signal = QtCore.pyqtSignal(dict)

    def __init__(self, fields, title='Options control'):
        super().__init__()
        self.title = title
        self.opts_fields = fields
        self.setupUi()
   
    def raise_widget(self):
        self.show()
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
        self.activateWindow()
        self.raise_()    

    def setupUi(self):
        self._layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel('Default ROI hwhm (n channels)\n hwhm = a0 + E(KeV) * a1')
        self._layout.addWidget(label)
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
        self._parameter_layout.addWidget(QtWidgets.QLabel('Value'), 0, 1)
        
        self.opt_controls = {}
        
        i = 1
        for opt in self.opts_fields:
            self._parameter_layout.addWidget(QtWidgets.QLabel(self.opts_fields[opt]['label']), i, 0)
            cb = DoubleSpinBoxAlignRight()
            cb.setMaximum(1000)
            cb.setObjectName(opt+"_control")
            val = self.opts_fields[opt]['val']
            set_cb_value(cb,val)
            self._parameter_layout.addWidget(cb)
            self.opt_controls[opt]= {'control':cb}
            
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

    

    def set_params(self,params):
        oc = self.opt_controls
        for opt in params: 
            if opt in oc:
                control=oc[opt]['control']
                val = params[opt]
                if val is not None:
                    set_cb_value(control,val)

    def apply(self):
        params = {}
        oc = self.opt_controls
        for opt in oc:
            val =  oc[opt]['control'].value()
            
            params[opt]= val
        self.apply_clicked_signal.emit(params)

def set_cb_value(btn, color):
    btn.setValue(float(color))

def rgb2hex(r,g,b):
    return f'#{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}'


def hex2rgb(hexcode):
    return tuple(map(ord,hexcode[1:].decode('hex')))