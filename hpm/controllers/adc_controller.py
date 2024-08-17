
from PyQt6 import QtCore, QtGui, QtWidgets
from mypyeqt.pvWidgets import pvQDoubleSpinBox, pvQLineEdit, pvQLabel, pvQMessageButton, pvQOZButton

class adcController():

    def __init__(self, record_name='16bmb:aim_adc1', file_options=None, environment_file=None):
        self.widget = adcControlWidget()
        

    



class adcControlWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)


    def make_AcquisitionButtons_groupbox(self):

        groupBoxAcq = QtWidgets.QGroupBox()

        _groupBoxAcqLayout = QtWidgets.QGridLayout(groupBoxAcq)
    
        btnOn = pvQMessageButton()
        btnOff = pvQMessageButton()
        btnErase = pvQMessageButton()

        _groupBoxAcqLayout.addWidget(btnOn,0,0)
        _groupBoxAcqLayout.addWidget(btnOff,0,1)
        _groupBoxAcqLayout.addWidget(btnErase,1,0)
        groupBoxAcq.setLayout(_groupBoxAcqLayout)

        return groupBoxAcq

    def make_TimeElapsed_groupbox(self):

        groupBoxElapsed = QtWidgets.QGroupBox()
        
        _groupBoxElapsedLayout = QtWidgets.QGridLayout(groupBoxElapsed)
       
        lblLiveTime_lbl = QtWidgets.QLabel("Live",groupBoxElapsed)
        lblRealTime_lbl = QtWidgets.QLabel("Real",groupBoxElapsed)
        
        lblLiveTime = pvQLabel()
        lblRealTime = pvQLabel()

        PRTM_pv = pvQDoubleSpinBox()
        PLTM_pv = pvQDoubleSpinBox()

        _groupBoxElapsedLayout.addWidget(lblLiveTime_lbl,0,0)
        _groupBoxElapsedLayout.addWidget(lblRealTime_lbl,1,0)

        _groupBoxElapsedLayout.addWidget(lblLiveTime,0,1)
        _groupBoxElapsedLayout.addWidget(lblRealTime,1,1)

        _groupBoxElapsedLayout.addWidget(PLTM_pv,0,2)
        _groupBoxElapsedLayout.addWidget(PRTM_pv,1,2)

        groupBoxElapsed.setLayout(_groupBoxElapsedLayout)

        return groupBoxElapsed
       