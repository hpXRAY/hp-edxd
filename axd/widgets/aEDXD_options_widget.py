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


from PyQt6 import QtCore, QtGui, QtWidgets

from  PyQt6.QtWidgets import QMainWindow, QApplication, QInputDialog, QWidget, QLabel
from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight, HorizontalLine, NumberTextField
from hpm.widgets.PltWidget import plotWindow
from functools import partial
from axd.models.aEDXD_functions import is_e

class aEDXDOptionsWidget(QWidget):

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
        self._parameter_layout.addWidget(QtWidgets.QLabel('Parameter'), 0, 1)
        self._parameter_layout.addWidget(QtWidgets.QLabel('Step'), 0, 3)
        self.opt_controls = {}
        
        i = 1
        for opt in self.opts_fields:
            self._parameter_layout.addWidget(QtWidgets.QLabel(self.opts_fields[opt]['label']), i, 0)
            self._parameter_layout.addWidget(QtWidgets.QLabel(self.opts_fields[opt]['unit']), i, 2)
            val=self.opts_fields[opt]['val']
            e = is_e(val)
            if e:
                o = NumberTextField()
            else:
                o = DoubleSpinBoxAlignRight()
            o.setObjectName(opt+"_control")
            o.setValue(val)
            o.setToolTip(self.opts_fields[opt]['desc'])
            o.setMinimum (self.opts_fields[opt]['step']) 
            o.setMinimumWidth(100)
            if not e:
                o.setSingleStep (self.opts_fields[opt]['step'])
                o_step = DoubleMultiplySpinBoxAlignRight()
                o_step.setObjectName(opt+"_control_step")
                o_step.setMinimum (0)
                step = self.opts_fields[opt]['step']
                o_step.setValue(self.opts_fields[opt]['step']) 
            if isinstance( self.opts_fields[opt]['val'], int):
                o.setDecimals(0)
                o.setMinimum(1)
                if not e:
                    o_step.setMinimum(1)
                    o_step.setDecimals(0)
            o_step.valueChanged.connect(partial(self.update_step, opt))
            self.opt_controls[opt]= {'val':o,'step':o_step}
            self._parameter_layout.addWidget(o, i, 1)
            self._parameter_layout.addWidget(o_step, i, 3)
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
        self.retranslateUi(self)
        self.style_widgets()

    def style_widgets(self):
        
        self.setStyleSheet("""
            #options_control_button_widget FlatButton {
                min-width: 70;
                max-width: 70;
            }
        """)

    def update_step(self, opt):
        o = self. opt_controls[opt]['val']
        o_step = self. opt_controls[opt]['step']
        value = o_step.value()
        o.setSingleStep(value)

    def retranslateUi(self, aEDXDWidget):
        pass

    def set_params(self,params):
        oc = self.opt_controls
        for opt in params: 
            if opt in oc:
                control=oc[opt]['val']
                val = params[opt]
                if val is not None:
                    control.setValue(val)

    def apply(self):
        params = {}
        oc = self.opt_controls
        for opt in oc:
            control=oc[opt]['val']
            params[opt]= float(control.value())
        self.apply_clicked_signal.emit(params)