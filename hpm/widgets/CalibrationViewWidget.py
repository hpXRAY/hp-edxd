
from PyQt6 import QtWidgets

from .CustomWidgets import NumberTextField, LabelAlignRight, CleanLooksComboBox, SpinBoxAlignRight, \
    DoubleSpinBoxAlignRight, FlatButton

class CalibrationParametersWidget(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._layout = QtWidgets.QGridLayout()

        self._layout.addWidget(LabelAlignRight('Cal_Offset:'), 0, 0)
        self.Cal_Offset_txt = NumberTextField()
        
     
        self._layout.addWidget(self.Cal_Offset_txt, 0, 1)
        

        self._layout.addWidget(LabelAlignRight('Cal_Slope:'), 1, 0)
        self.Cal_Slope_txt = NumberTextField()
        
        self._layout.addWidget(self.Cal_Slope_txt, 1, 1)
       

        self._layout.addWidget(LabelAlignRight('Cal_Quad:'), 2, 0)
        self.Cal_Quad_txt = NumberTextField()
        self._layout.addWidget(self.Cal_Quad_txt, 2, 1)



        self._layout.addWidget(LabelAlignRight('2 theta:'), 3, 0)
        self.two_theta = NumberTextField()
        self._layout.addWidget(self.two_theta, 4, 1)
        self._layout.addWidget(QtWidgets.QLabel('degree'), 3, 2)

        

        self.update_btn = FlatButton('update')
        self._layout.addWidget(self.update_btn, 10, 0, 1, 4)

        self._layout.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding),
                             11, 0, 1, 4)

        self.setLayout(self._layout)
