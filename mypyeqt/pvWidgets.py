# -*- coding: utf8 -*-

from PyQt6 import uic, QtWidgets,QtCore
from PyQt6.QtCore import QObject, pyqtSignal, Qt
import pyqtgraph as pg
from mypyeqt.CustomWidgets import HorizontalSpacerItem, VerticalSpacerItem, FlatButton
from functools import partial
import time
#from pyeqt.pv_model import PV
from PyQt6.QtWidgets import QWidget, QLabel, QLineEdit, QComboBox, QDoubleSpinBox, QCheckBox, QPushButton, QProgressBar

from mypyeqt.pvServer import pvServer
import numpy as np
from epics import PV, caget, caput
from mypyeqt.epicsMonitor import epicsMonitor

#from pyeqt.ImageViewWidget import ImageViewWidget

class custom_signal(QtCore.QObject):
    signal = QtCore.pyqtSignal() 


    def __init__(self, debounce_time=None):
        super().__init__()
        self.debounce_time = debounce_time
        self.emitted_timestamp = None


    
    def emit(self):
        if self.debounce_time is None:
            self.signal.emit()
            return
        else:
            # the following is the de-bouncing code
            if self.emitted_timestamp is not None:
                elapsed_since_last_emit = time.time() - self.emitted_timestamp
                #print ('elapsed_since_last_emit: '+ str(elapsed_since_last_emit))
            else:
                elapsed_since_last_emit = -1
            self.emitted_timestamp = time.time()
            if elapsed_since_last_emit >= 0 and elapsed_since_last_emit < self.debounce_time:
                #print('signal skipped')
                pass
            else:
                self.signal.emit()

class pvQWidget(QObject):

    def __init__(self,pvName=None, autoconnect = True, as_string = False):
       
        self.setEnabled( False)
        self.signal = custom_signal(debounce_time=.1)
        self.signal.signal.connect(self.put_value)
        self.widget_type = None
        self.pv_name = pvName
        self.put_value = None
        #self.DESC = ''
        self.connected = False
        self.pv_server = None
        if autoconnect and self.pv_name is not None:
            self.connect(as_string=as_string)

    def connect(self, pvName=None, as_string = False):
        self.disconnect()
        if not self.connected:
            if pvName is None:
                pvName = self.pv_name
    
            if pvName is not None:    
                self.pv_name = pvName
                if self.pv_server is None:
                    self.pv_server = pvServer()
                self.pv = self.pv_server.get_pv(pvName)
                #info = self.pv.info
                #print(info)
                
                self.monitor = self.pv_server.get_pv_monitor(pvName)
                self.status = self.pv.status
                if self.status is not None:
                    self.setEnabled( True)
                    val = self.pv.get(as_string = as_string)
                    self.updateView(val)
                self.connect_pv()

  
    def disconnect(self):
        if self.connected:
            self.disconnect_pv()

    def re_connect(self):
        if not self.connected:
            self.connect_pv()
            
    def connect_pv(self):
        if not self.connected:
            self.monitor.callback_triggered.connect(self.monitorCallback)
            self.monitor.connection_callback_triggered.connect(self.connectionChanged)
            self.setToolTip(self.pv_name)
            self.connectionChanged(True)

    def disconnect_pv(self):
        if self.connected:
            self.monitor.connection_callback_triggered.disconnect(self.connectionChanged)
            self.monitor.callback_triggered.disconnect(self.monitorCallback)
            self.setToolTip('')
            self.connectionChanged(False)
           
    def connectionChanged(self, conn):
        # prototype function, can be offerridden 
        # to show whether widgets are connected 
        # conn is a bool
        self.connected = conn
        self.setEnabled(conn)
        
    def WidgetValueChangedCallback(self,value):
        if self.connected:
            # value must be a string !!!
            #if value != self.put_value:
            self.put_value = value
            self.signal.emit()
                #print('caput ' + self.pv_name + ' ' +value)
       
    # set value is called when monitor detects a change
    def monitorCallback(self, value):
        if self.connected:
            self.updateView(value)
            #print(self.pv_name + " " + str(value))

    def updateView(self, value):
        # prototype, children override this funciton
        pass

    def put_value(self):
        if self.connected:
            # value must be a string !!!
            value = self.put_value
            if value is not None:
                self.pv.put(value)
                #print('caput ' + self.pv_name + ' ' +value)


class pvQDoubleSpinBox(QDoubleSpinBox, pvQWidget):
    def __init__(self, parent=None, pvname=None, autoconnect = True):
        QDoubleSpinBox.__init__(self,parent = parent)
        self.widget_type = QDoubleSpinBox
        widget = self
        QDoubleSpinBox.setGroupSeparatorShown(widget, True)
        minimum = -10000000000000000.0
        maximum = 10e16      
                    
        QDoubleSpinBox.setMinimum(widget, minimum)
        QDoubleSpinBox.setMaximum(widget, maximum)
        '''QDoubleSpinBox.setDecimals(widget, 3)
        QDoubleSpinBox.setSingleStep(widget, 0.001)'''

        pvQWidget.__init__(self, pvname, autoconnect)

        QDoubleSpinBox.setKeyboardTracking(widget, False)

        self.valueChanged.connect(self.valueChangedCallbackPreTreat)
        


    def valueChangedCallbackPreTreat(self, val):
        val = round(val,3)
        val = str(val)
        self.WidgetValueChangedCallback(val )


    # set value is called when monitor detects a change
    def updateView(self, value):
        
        # make sure that the db record has a PREC field set, 
        # otherwise the prescision of the values returned by the monitor is arbitrary
        value = float(value)
        widget = self
        QDoubleSpinBox.blockSignals(widget, True)
        QDoubleSpinBox.setValue(widget, float(value))
        QDoubleSpinBox.blockSignals(widget, False)  
        

class pvQLineEdit(QLineEdit, pvQWidget):
    def __init__(self, parent=None, pvname=None, autoconnect = True, nelem=40):
        widget = self
        self.nelem = nelem-1
        QLineEdit.__init__(widget,parent = parent)
        self.widget_type = QLineEdit
        pvQWidget.__init__(widget, pvname, autoconnect, as_string=True)
        self.editingFinished.connect(lambda:self.valueChangedCallbackPreTreat(self.text()))
         

    def valueChangedCallbackPreTreat(self, val):
        if len(val)>self.nelem:
            val = str(val)[:self.nelem]
            self.updateView(val)
        QLineEdit.blockSignals(self, True)
        self.WidgetValueChangedCallback(val )    
        QLineEdit.blockSignals(self, False)      

    def updateView(self, value):
        widget = self
        QLineEdit.blockSignals(widget, True)
        QLineEdit.setText(widget, str(value))
        QLineEdit.blockSignals(widget, False)    



class pvQLabel(QLabel, pvQWidget):
    def __init__(self, parent=None, pvname=None, autoconnect = True, as_num= False, round=2, average=1):
        widget = self
        self.as_num = as_num
        self.round = round
        
        if self.as_num:
            self.av_list = []

        self.value = None
        QLabel.__init__(widget,parent = parent)
        self.widget_type = QLabel
        pvQWidget.__init__(widget, pvname, autoconnect,as_string=True)

        
    def updateView(self, value):
        widget = self
        if self.as_num:
            val = float(value)
            if self.average > 1:
                if len(self.av_list) >= self.average:
                    self.av_list.pop(0)
                self.av_list.append(val)
                value  = sum(self.av_list) / len(self.av_list)
            value = round(value,self.round)
        QLabel.setText(widget, str(value))

class pvQProgressBar(QProgressBar, pvQWidget):
    def __init__(self, parent=None, pvname=None, autoconnect = True, average=1):
        widget = self
        self.as_num = True
        self.round = 0
        self.average = average
        self.color = 0
        self.previous_color = 0
        self.StyleSheet_colors = ['''#dead_time_indicator {
                                        text-align: center;
                                        min-height: 19px;
                                        max-height: 19px;
                                        border-radius: 3px;
                                        background-color: #6F6F6F;
                                    }
                                    #dead_time_indicator::chunk {
                                    border-radius: 3px;
                                    background-color: #31D27E;
                                    margin: 1px;
                                    width: 5px; 
                                }''',
                                '''#dead_time_indicator {
                                        text-align: center;
                                        min-height: 19px;
                                        max-height: 19px;
                                        border-radius: 3px;
                                        background-color: #6F6F6F;
                                    }
                                    #dead_time_indicator::chunk {
                                    border-radius: 3px;
                                    background-color: #E8CE36;
                                    margin: 1px;
                                    width: 5px; 
                                }''',
                                '''#dead_time_indicator {
                                        text-align: center;
                                        min-height: 19px;
                                        max-height: 19px;
                                        border-radius: 3px;
                                        background-color: #6F6F6F;
                                    }
                                    #dead_time_indicator::chunk {
                                    border-radius: 3px;
                                    background-color: #E83636;
                                    margin: 1px;
                                    width: 5px; 
                                }''']
        

        if self.as_num:
            self.av_list = []
        
        self.value = None
        QProgressBar.__init__(widget,parent = parent)
        self.setStyleSheet(self.StyleSheet_colors[self.color])
        self.widget_type = QProgressBar
        pvQWidget.__init__(widget, pvname, autoconnect,as_string=True)
        
        
    def updateView(self, value):
        widget = self
        if self.as_num:
            val = float(value)
            if self.average > 1:
                if len(self.av_list) >= self.average:
                    self.av_list.pop(0)
                self.av_list.append(val)
                value  = sum(self.av_list) / len(self.av_list)
            value = round(value,self.round)

            if value<20:
                self.color = 0
            elif value >=20 and value < 40:
                self.color = 1
            else:
                self.color = 2
            if self. previous_color != self.color:
                self.previous_color = self.color
                self.setStyleSheet(self.StyleSheet_colors[self.color])
        old_val = QProgressBar.value(widget)
        if old_val != value:
            QProgressBar.setValue(widget, value)

    

        

class pvQMessageButton(QPushButton, pvQWidget):
    def __init__(self, parent=None, pvname=None, message = '', autoconnect = True):
        widget = self
        QPushButton.__init__(widget,parent=parent)
        self.widget_type = QPushButton
        pvQWidget.__init__(widget, pvname)
 
        self.setMessage(message)
        
        self.clicked.connect(self.valueChangedCallbackPreTreat)

    def setMessage(self, message =''):
        self.message = str(message)
        QPushButton.setText(self, self.message)

    def valueChangedCallbackPreTreat(self, val):
        val = str(self.message)
        self.WidgetValueChangedCallback(val )    

class pvQOZButton(QPushButton, pvQWidget):
    def __init__(self, parent=None, pvname=None, autoconnect = True):
        widget = self
        QPushButton.__init__(widget,parent = parent)
        self.widget_type = QPushButton
        pvQWidget.__init__(widget, pvname, autoconnect)
        onam = pvname+'.ONAM'
        znam = pvname+'.ZNAM'
        self.onam = caget(onam)
        self.znam = caget(znam)
        QPushButton.setCheckable(widget, True)
        self.clicked.connect(self.valueChangedCallbackPreTreat)
        
    def valueChangedCallbackPreTreat(self, val):
        if val :
            val = '1'
        else:
            val = '0'
        self.WidgetValueChangedCallback(val )    

    def updateView(self, value):
        if type(value) == str:
            value = value == self.onam or value == '1'
        widget = self
        QPushButton.blockSignals(widget, True)
        QPushButton.setChecked(widget, value)
        QPushButton.blockSignals(widget, False)    
        

class pvQCheckBox(QCheckBox, pvQWidget):

    def __init__(self, parent=None, pvname=None, autoconnect = True):
        QCheckBox.__init__(self,parent = parent)
        self.widget_type = QCheckBox
        pvQWidget.__init__(self, pvname, autoconnect)
        onam = pvname+'.ONAM'
        znam = pvname+'.ZNAM'
        desc = pvname+'.DESC'
        self.desc = caget(desc)
        self.onam = caget(onam)
        self.znam = caget(znam)

        widget = self
        QCheckBox.setCheckable(widget, True)
        QCheckBox.setText(widget, self.desc)
        
        self.clicked.connect(self.valueChangedCallbackPreTreat)
       

    def valueChangedCallbackPreTreat(self, val):
        if val :
            val = '1'
        else:
            val = '0'
        self.WidgetValueChangedCallback(val )  

    def updateView(self, value):
        if type(value) == type(''):
            value = value == self.onam or value == '1'
        widget = self
        QCheckBox.blockSignals(widget, True)
        QCheckBox.setChecked(widget, value)
        QCheckBox.blockSignals(widget, False)    

class pvStripChart(pg.PlotWidget, pvQWidget):
    def __init__(self, parent=None, pvname=None, autoconnect = True, NLEN=100):
        widget = self
        pg.PlotWidget.__init__(widget, parent=parent)
        self.widget_type = pg.PlotWidget
        
        pvQWidget.__init__(widget, pvname, autoconnect)
        self.NLEN = NLEN

        self.plotForeground = pg.PlotDataItem([], title="",
                antialias=True, pen=pg.mkPen(color=(255,255,255), width=1) ,parent=widget)
        self.addItem(self.plotForeground)
        self.data_array = []
        
    def updateView(self, value):
        widget = self
        value = float(value)
        
        self.data_array.append(value)
        if len(self.data_array)>self.NLEN:
            self.data_array = self.data_array[-self.NLEN:]
        self.plotForeground.setData(self.data_array)

#not ready yet


        
'''
class pvQComboBox(QComboBox, pvQWidget):
    def __init__(self, parent=None, myPV):
        QComboBox.__init__(self)
        pvQWidget.__init__(self, myPV)
        widget = self
        items = myPV._items
        for item in items:
            QComboBox.addItem(widget, item)
        val = myPV._val            
        self.setValue(widget, [val])
        self.currentTextChanged.connect(self.valueChangedCallback)

    def setValue(self, tag, value):
        pvQWidget.setValue(self, value)
        value = self.val
        self.blockSignals(True)
       
        widget = self
        current_value = widget.currentText()
        if value != current_value:
            QComboBox.setCurrentText(widget, str(value))
        self.blockSignals(False)    

'''



'''
class pvImage(ImageViewWidget, pvQWidget):
    def __init__(self, pvname=None, autoconnect = True):
       
        widget = self
        self.start = time.time()
        self.pv_base = pvname
        pvname_array = self.pv_base+'ArrayData'
        self.array_PV = PV(pvname_array)

        pvname_ArraySize0 = self.pv_base+'ArraySize0_RBV'
        pvname_ArraySize1 = self.pv_base+'ArraySize1_RBV'
        self.ArraySize0 = PV(pvname_ArraySize0)
        self.ArraySize1 = PV(pvname_ArraySize1)


        pvname = self.pv_base + 'TimeStamp_RBV'
        self.frame_ind = 0
        ImageViewWidget.__init__(widget)
        
        pvQWidget.__init__(widget, pvname, autoconnect)

        
    def updateView(self, value):
        
        
        
        wait_till = self.start+0.5
        if  time.time() > wait_till:
            widget = self
        
            value = float(value)
            #print(value)
            
            ArraySize0 = self.ArraySize0.get()
            ArraySize1 = self.ArraySize1.get()
            now = time.time()
            img = self.array_PV.get()
            end = time.time()
            elapsed = end - now
            print(elapsed)
            if img is not None:
                #img = np.asarray(img)
                
                img = img.reshape((ArraySize1, ArraySize0))
                img = img.transpose()
                
                ImageViewWidget.setData(widget, [img])

                ImageViewWidget.displayImage(widget, 0)
        
            self.start = time.time()


class pvQPlainTextEdit(QPlainTextEdit, pvQWidget):
    def __init__(self, parent=None, pvname=None, autoconnect = True, nelem=40):
        widget = self
        self.nelem = nelem-1
        QPlainTextEdit.__init__(widget,parent = parent)
        self.widget_type = QPlainTextEdit
        pvQWidget.__init__(widget, pvname, autoconnect, as_string=True)
        self.textChanged.connect(lambda:self.valueChangedCallbackPreTreat(self.text()))
         

    def valueChangedCallbackPreTreat(self, val):
        if len(val)>self.nelem:
            val = str(val)[:self.nelem]
            self.updateView(val)
        QPlainTextEdit.blockSignals(self, True)
        self.WidgetValueChangedCallback(val )    
        QPlainTextEdit.blockSignals(self, False)      

    def updateView(self, value):
        widget = self
        QPlainTextEdit.blockSignals(widget, True)
        QPlainTextEdit.setText(widget, str(value))
        QPlainTextEdit.blockSignals(widget, False)  
'''