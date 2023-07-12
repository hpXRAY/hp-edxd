# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hpMCA.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph.widgets.PlotWidget import PlotWidget
from mypyeqt.pvWidgets import pvQDoubleSpinBox, pvQLineEdit, pvQLabel, pvQMessageButton, pvQOZButton, pvQProgressBar

from hpm.widgets.collapsible_widget import CollapsibleBox, EliderLabel

from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, HorizontalLine, IntegerTextField, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight, CheckableFlatButton

from .. import icons_path

import os

class Ui_hpMCA(object):
    def setupUi(self, hpMCA):
        #hpMCA.setObjectName("hpMCA")
        hpMCA.resize(1300, 715)
        self.centralwidget = QtWidgets.QWidget(hpMCA)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_4.setObjectName("verticalLayout_4")

        self._Layout = QtWidgets.QHBoxLayout()
        self._Layout.setObjectName("_Layout")



        self.ControlsLayout = QtWidgets.QVBoxLayout()
        self.ControlsLayout.setObjectName("ControlsLayout")

        

        
        
        self.groupBoxAcq = QtWidgets.QGroupBox()
        
        self._groupBoxAcqLayout = QtWidgets.QGridLayout(self.groupBoxAcq)
        
        self.btnOn = pvQMessageButton()
        self.btnOff = pvQMessageButton()
        self.btnErase = pvQMessageButton()
        self.dead_time_indicator = pvQProgressBar(average=20)
        
        self.dead_time_indicator.setObjectName('dead_time_indicator')
        

        self.btnOn.setMaximumWidth(75)
        self.btnOff.setMaximumWidth(75)
        self.btnErase.setMaximumWidth(75)
        self.dead_time_indicator.setMaximumWidth(75)

        self._groupBoxAcqLayout.addWidget(self.btnOn,0,0)
        self._groupBoxAcqLayout.addWidget(self.btnOff,0,1)
        self._groupBoxAcqLayout.addWidget(self.btnErase,1,0)
        self._groupBoxAcqLayout.addWidget(self.dead_time_indicator,1,1)
        self.groupBoxAcq.setLayout(self._groupBoxAcqLayout)


        self.groupBoxElapsed = QtWidgets.QGroupBox()
        self.groupBoxElapsed.setMaximumWidth(205)
        self._groupBoxElapsedLayout = QtWidgets.QGridLayout(self.groupBoxElapsed)
        self._groupBoxElapsedLayout.setSpacing(2)
        self._groupBoxElapsedLayout.setContentsMargins(12,12,7,7)
        self.groupBoxAcq.setMaximumWidth(205)

        self.lblLiveTime_lbl = QtWidgets.QLabel(self.groupBoxElapsed)
        self.lblLiveTime_lbl.setMaximumWidth(40)
        self.lblRealTime_lbl = QtWidgets.QLabel(self.groupBoxElapsed)
        self.lblRealTime_lbl.setMaximumWidth(40)
        self.lblLiveTime = pvQLabel()
        self.lblRealTime = pvQLabel()
        self.PRTM_pv = pvQDoubleSpinBox()
        self.PLTM_pv = pvQDoubleSpinBox()
        self.PRTM_pv.setMaximumWidth(70)
        self.PLTM_pv.setMaximumWidth(70)
        
        self.PRTM_pv.setMinimum(0)
        self.PLTM_pv.setMinimum(0)
        #self.PRTM_pv.setDecimals(1)
        #self.PLTM_pv.setDecimals(1)

        self.PLTM_0 = pvQMessageButton()
        self.PLTM_1 = pvQMessageButton()
        self.PLTM_2 = pvQMessageButton()
        self.PLTM_3 = pvQMessageButton()
        self.PLTM_4 = pvQMessageButton()
        self.PLTM_5 = pvQMessageButton()
        self.PLTM_0.setMaximumWidth(28)
        self.PLTM_1.setMaximumWidth(28)
        self.PLTM_2.setMaximumWidth(28)
        self.PLTM_3.setMaximumWidth(28)
        self.PLTM_4.setMaximumWidth(28)
        self.PLTM_5.setMaximumWidth(28)
        self.PLTM_0.setMaximumHeight(15)
        self.PLTM_1.setMaximumHeight(15)
        self.PLTM_2.setMaximumHeight(15)
        self.PLTM_3.setMaximumHeight(15)
        self.PLTM_4.setMaximumHeight(15)
        self.PLTM_5.setMaximumHeight(15)


        self.PLTM_btns = QtWidgets.QWidget()
        self._PLTM_btns_layout = QtWidgets.QGridLayout()
        self._PLTM_btns_layout.setSpacing(2)
        self._PLTM_btns_layout.setContentsMargins(74,0,0,12)
        self._PLTM_btns_layout.addWidget( self.PLTM_0 ,0,0)
        self._PLTM_btns_layout.addWidget( self.PLTM_1 ,0,1)
        self._PLTM_btns_layout.addWidget( self.PLTM_2 ,0,2)
        self._PLTM_btns_layout.addWidget( self.PLTM_3 ,1,0)
        self._PLTM_btns_layout.addWidget( self.PLTM_4 ,1,1)
        self._PLTM_btns_layout.addWidget( self.PLTM_5 ,1,2)
        self.PLTM_btns.setLayout(self._PLTM_btns_layout)

        self.PRTM_0 = pvQMessageButton()
        self.PRTM_1 = pvQMessageButton()
        self.PRTM_2 = pvQMessageButton()
        self.PRTM_3 = pvQMessageButton()
        self.PRTM_4 = pvQMessageButton()
        self.PRTM_5 = pvQMessageButton()
        self.PRTM_0.setMaximumWidth(28)
        self.PRTM_1.setMaximumWidth(28)
        self.PRTM_2.setMaximumWidth(28)
        self.PRTM_3.setMaximumWidth(28)
        self.PRTM_4.setMaximumWidth(28)
        self.PRTM_5.setMaximumWidth(28)
        self.PRTM_0.setMaximumHeight(15)
        self.PRTM_1.setMaximumHeight(15)
        self.PRTM_2.setMaximumHeight(15)
        self.PRTM_3.setMaximumHeight(15)
        self.PRTM_4.setMaximumHeight(15)
        self.PRTM_5.setMaximumHeight(15)

        self.PRTM_btns = QtWidgets.QWidget()
        self._PRTM_btns_layout = QtWidgets.QGridLayout()
        self._PRTM_btns_layout.setSpacing(2)
        self._PRTM_btns_layout.setContentsMargins(74,0,0,5)
        self._PRTM_btns_layout.addWidget( self.PRTM_0 ,0,0)
        self._PRTM_btns_layout.addWidget( self.PRTM_1 ,0,1)
        self._PRTM_btns_layout.addWidget( self.PRTM_2 ,0,2)
        self._PRTM_btns_layout.addWidget( self.PRTM_3 ,1,0)
        self._PRTM_btns_layout.addWidget( self.PRTM_4 ,1,1)
        self._PRTM_btns_layout.addWidget( self.PRTM_5 ,1,2)
        self.PRTM_btns.setLayout(self._PRTM_btns_layout)


        

        self.lbl_elapsed_header = QtWidgets.QLabel()
        #self._groupBoxElapsedLayout.addWidget(self.lbl_elapsed_header,0,2)
        self._groupBoxElapsedLayout.addWidget(self.lblLiveTime_lbl,1,0)

        

        self._groupBoxElapsedLayout.addWidget(self.lblRealTime_lbl,3,0)
        self._groupBoxElapsedLayout.addWidget(self.lblLiveTime,1,1)

        self._groupBoxElapsedLayout.addWidget(self.PLTM_btns,2,1,1,2)

        self._groupBoxElapsedLayout.addWidget(self.lblRealTime,3,1)

        self._groupBoxElapsedLayout.addWidget(self.PRTM_btns,4,1,1,2)

        self._groupBoxElapsedLayout.addWidget(self.PLTM_pv,1,2)
        self._groupBoxElapsedLayout.addWidget(self.PRTM_pv,3,2)

        self.groupBoxElapsed.setLayout(self._groupBoxElapsedLayout)
        
        
        


        self.groupBoxROIs = QtWidgets.QGroupBox()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBoxROIs.sizePolicy().hasHeightForWidth())
        self.groupBoxROIs.setSizePolicy(sizePolicy)
        self.groupBoxROIs.setMinimumSize(QtCore.QSize(205, 0))
        self.groupBoxROIs.setMaximumSize(QtCore.QSize(205, 120))
        self.groupBoxROIs.setObjectName("groupBoxROIs")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.groupBoxROIs)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.ROIBtnsLayout = QtWidgets.QGridLayout()
        self.ROIBtnsLayout.setObjectName("ROIBtnsLayout")
        self.btnROIadd = QtWidgets.QPushButton(self.groupBoxROIs)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnROIadd.sizePolicy().hasHeightForWidth())
        self.btnROIadd.setSizePolicy(sizePolicy)
        self.btnROIadd.setMinimumSize(QtCore.QSize(75, 0))
        self.btnROIadd.setCheckable(True)
        self.btnROIadd.setObjectName("btnROIadd")
        self.ROIBtnsLayout.addWidget(self.btnROIadd, 0, 0, 1, 1)
        self.btnROIdelete = QtWidgets.QPushButton(self.groupBoxROIs)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnROIdelete.sizePolicy().hasHeightForWidth())
        self.btnROIdelete.setSizePolicy(sizePolicy)
        self.btnROIdelete.setMinimumSize(QtCore.QSize(75, 0))
        self.btnROIdelete.setObjectName("btnROIdelete")
        self.ROIBtnsLayout.addWidget(self.btnROIdelete, 0, 1, 1, 1)
        self.btnROIclear = QtWidgets.QPushButton(self.groupBoxROIs)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnROIclear.sizePolicy().hasHeightForWidth())
        self.btnROIclear.setSizePolicy(sizePolicy)
        self.btnROIclear.setMinimumSize(QtCore.QSize(75, 0))
        self.btnROIclear.setObjectName("btnROIclear")
        self.ROIBtnsLayout.addWidget(self.btnROIclear, 1, 0, 1, 1)
        self.verticalLayout_8.addLayout(self.ROIBtnsLayout)
        self.ROIPrevNextLayout = QtWidgets.QHBoxLayout()
        self.ROIPrevNextLayout.setObjectName("ROIPrevNextLayout")
        self.btnROIprev = QtWidgets.QPushButton(self.groupBoxROIs)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnROIprev.sizePolicy().hasHeightForWidth())
        self.btnROIprev.setSizePolicy(sizePolicy)
        self.btnROIprev.setMaximumSize(QtCore.QSize(30, 23))
        self.btnROIprev.setObjectName("btnROIprev")
        self.ROIPrevNextLayout.addWidget(self.btnROIprev)
        self.lineROI = QtWidgets.QLineEdit(self.groupBoxROIs)
        self.lineROI.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.lineROI.setObjectName("lineROI")
        self.ROIPrevNextLayout.addWidget(self.lineROI)
        self.btnROInext = QtWidgets.QPushButton(self.groupBoxROIs)
        self.btnROInext.setMaximumSize(QtCore.QSize(30, 23))
        self.btnROInext.setObjectName("btnROInext")
        self.ROIPrevNextLayout.addWidget(self.btnROInext)
        self.verticalLayout_8.addLayout(self.ROIPrevNextLayout)
        
        self.groupBoxXRF = QtWidgets.QGroupBox()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBoxXRF.sizePolicy().hasHeightForWidth())
        self.groupBoxXRF.setSizePolicy(sizePolicy)
        self.groupBoxXRF.setMinimumSize(QtCore.QSize(205, 0))
        self.groupBoxXRF.setMaximumSize(QtCore.QSize(205, 16777215))
        self.groupBoxXRF.setObjectName("groupBoxXRF")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBoxXRF)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.XRFPrevNextLayout = QtWidgets.QHBoxLayout()
        self.XRFPrevNextLayout.setObjectName("XRFPrevNextLayout")
        self.btnKLMprev = QtWidgets.QPushButton(self.groupBoxXRF)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnKLMprev.sizePolicy().hasHeightForWidth())
        self.btnKLMprev.setSizePolicy(sizePolicy)
        self.btnKLMprev.setMaximumSize(QtCore.QSize(30, 23))
        self.btnKLMprev.setObjectName("btnKLMprev")
        self.XRFPrevNextLayout.addWidget(self.btnKLMprev)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.groupBoxXRF)
        self.lineEdit_2.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.XRFPrevNextLayout.addWidget(self.lineEdit_2)
        self.btnKLMnext = QtWidgets.QPushButton(self.groupBoxXRF)
        self.btnKLMnext.setMaximumSize(QtCore.QSize(30, 23))
        self.btnKLMnext.setObjectName("btnKLMnext")
        self.XRFPrevNextLayout.addWidget(self.btnKLMnext)
        self.verticalLayout_6.addLayout(self.XRFPrevNextLayout)
        
        self.groupBoxVerticalScale = QtWidgets.QGroupBox()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBoxVerticalScale.sizePolicy().hasHeightForWidth())
        self.groupBoxVerticalScale.setSizePolicy(sizePolicy)
        self.groupBoxVerticalScale.setMinimumSize(QtCore.QSize(205, 0))
        self.groupBoxVerticalScale.setMaximumSize(QtCore.QSize(205, 16777215))
        self.groupBoxVerticalScale.setObjectName("groupBoxVerticalScale")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.groupBoxVerticalScale)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.LogScaleLayout = QtWidgets.QHBoxLayout()
        self.LogScaleLayout.setObjectName("LogScaleLayout")
        self.radioLog = QtWidgets.QRadioButton(self.groupBoxVerticalScale)
        self.radioLog.setMaximumSize(QtCore.QSize(50, 16777215))
        self.radioLog.setChecked(True)
        self.radioLog.setObjectName("radioLog")
        self.LogScaleLayout.addWidget(self.radioLog)
        self.radioLin = QtWidgets.QRadioButton(self.groupBoxVerticalScale)
        self.radioLin.setChecked(False)
        self.radioLin.setObjectName("radioLin")
        self.LogScaleLayout.addWidget(self.radioLin)

        self.baseline_subtract = QtWidgets.QPushButton()
        self.baseline_subtract.setText("- bg")
        self.baseline_subtract.setEnabled(False)
        self.baseline_subtract.setCheckable(True)
        self.baseline_subtract.setChecked(False)
        self.baseline_subtract.setMaximumWidth(45)
        self.LogScaleLayout.addWidget(self.baseline_subtract)

        self.verticalLayout_7.addLayout(self.LogScaleLayout)

        
        
        self.groupBoxHorizontalScale = QtWidgets.QGroupBox()
        
        self.groupBoxHorizontalScale.setMinimumSize(QtCore.QSize(0, 0))
        self.groupBoxHorizontalScale.setMaximumSize(QtCore.QSize(205, 400))
        self.groupBoxHorizontalScale.setObjectName("groupBoxHorizontalScale")
        self._groupBoxHorizontalScale_layout = QtWidgets.QVBoxLayout(self.groupBoxHorizontalScale)
        self._groupBoxHorizontalScale_layout.setObjectName("_groupBoxHorizontalScale_layout")
        self.HorizontalScaleLayout = QtWidgets.QHBoxLayout()
        self.HorizontalScaleLayout.setSpacing(0)
        
        self.HorizontalScale_btn_group = QtWidgets.QButtonGroup()
        self.radioE = QtWidgets.QPushButton(self.groupBoxHorizontalScale)

        self.radioE.setObjectName("radioE")
        self.HorizontalScaleLayout.addWidget(self.radioE)
        self.radioq = QtWidgets.QPushButton(self.groupBoxHorizontalScale)
        self.radioq.setObjectName("radioq")
        self.HorizontalScaleLayout.addWidget(self.radioq)
        
        self.radiod = QtWidgets.QPushButton(self.groupBoxHorizontalScale)
        self.radiod.setObjectName("radiod")
        self.HorizontalScaleLayout.addWidget(self.radiod)
        self.radiotth = QtWidgets.QPushButton(self.groupBoxHorizontalScale)
        self.radiotth.setObjectName("radiotth")
        self.HorizontalScaleLayout.addWidget(self.radiotth)
        self.radioChannel = QtWidgets.QPushButton(self.groupBoxHorizontalScale)
        self.radioChannel.setObjectName("radioChannel")
        self.HorizontalScaleLayout.addWidget(self.radioChannel)
        self.radioE.setCheckable(True)
        self.radioq.setCheckable(True)
        self.radioChannel.setCheckable(True)
        self.radiod.setCheckable(True)
        self.radiotth.setCheckable(True)
        self.HorizontalScale_btn_group.addButton(self.radioE)
        self.HorizontalScale_btn_group.addButton(self.radioq)
        self.HorizontalScale_btn_group.addButton(self.radioChannel)
        self.HorizontalScale_btn_group.addButton(self.radiod)
        self.HorizontalScale_btn_group.addButton(self.radiotth)
        self.radioChannel.setChecked(True)
        

        self._groupBoxHorizontalScale_layout.addLayout(self.HorizontalScaleLayout)
        

        # Save data file
        
        self.groupBoxSaveDataFile = QtWidgets.QGroupBox()
        self.groupBoxSaveDataFile.setMaximumWidth(205)
        self._groupBoxSaveDataFileLayout = QtWidgets.QVBoxLayout(self.groupBoxSaveDataFile)
        self._groupBoxSaveDataFileLayout.setContentsMargins(12,7,7,7)
        self._groupBoxSaveDataFileLayout.setSpacing(7)

        self._groupBoxSaveDataFileLayout.addWidget(QtWidgets.QLabel('Save in:'))

        self.folder_browse_widget = QtWidgets.QWidget()
        self._folder_browse_widget_layout = QtWidgets.QHBoxLayout(self.folder_browse_widget)
        self._folder_browse_widget_layout.setContentsMargins(0,0,0,0)
        self.folder_lbl = EliderLabel(mode=QtCore.Qt.ElideLeft)
        
        self._folder_browse_widget_layout.addWidget(self.folder_lbl)
        self.folder_browse_btn = QtWidgets.QPushButton('...')
        self.folder_browse_btn.setMaximumWidth(30)
        self._folder_browse_widget_layout.addWidget(self.folder_browse_btn)
        self._groupBoxSaveDataFileLayout.addWidget(self.folder_browse_widget)

    
        self.file_name_ebx = QtWidgets.QComboBox()
        self.file_name_ebx.setEditable(True)
        
        
        self._groupBoxSaveDataFileLayout.addWidget(QtWidgets.QLabel("File name:"))
        self._groupBoxSaveDataFileLayout.addWidget(self.file_name_ebx)

        self.save_file_btn = QtWidgets.QPushButton('Save')
        self.save_file_btn.setMaximumWidth(75)
        self._groupBoxSaveDataFileLayout.addWidget(self.save_file_btn)

        self._groupBoxSaveDataFileLayout.addWidget(QtWidgets.QLabel('Last saved file:'))
        self.last_saved_lbl = EliderLabel('', mode=QtCore.Qt.ElideLeft)
        self.last_saved_lbl.setAlignment(QtCore.Qt.AlignLeft)
        self._groupBoxSaveDataFileLayout.addWidget(self.last_saved_lbl)


        ## naming options

        self.groupBoxFileNamingOptions = QtWidgets.QGroupBox()
        self.groupBoxFileNamingOptions.setMaximumWidth(205)
        self._groupBoxFileNamingOptionsLayout = QtWidgets.QVBoxLayout(self.groupBoxFileNamingOptions)
        self._groupBoxFileNamingOptionsLayout.setContentsMargins(12,7,7,7)
        self._groupBoxFileNamingOptionsLayout.setSpacing(3)

        

        self.naming_options_lbl = QtWidgets.QWidget()
        self._naming_options_lbl_layout = QtWidgets.QHBoxLayout(self.naming_options_lbl)
        self._naming_options_lbl_layout.setContentsMargins(0,0,0,0)
        self._naming_options_lbl_layout.setSpacing(0)
        self._naming_options_lbl_layout.addWidget(QtWidgets.QLabel('Naming options'))

        self._groupBoxFileNamingOptionsLayout.addWidget(self.naming_options_lbl)

        self.incr_file_name_widget = QtWidgets.QWidget()
        self._incr_file_name_widget_layout = QtWidgets.QHBoxLayout(self.incr_file_name_widget)
        self._incr_file_name_widget_layout.setContentsMargins(0,5,0,0)
        self._incr_file_name_widget_layout.setSpacing(0)
        self.increment_file_name_cbx = QtWidgets.QCheckBox('Increment file name')
        self._incr_file_name_widget_layout.addWidget(self.increment_file_name_cbx)
        
        self._groupBoxFileNamingOptionsLayout.addWidget(self.incr_file_name_widget)

        self.starting_num_widget = QtWidgets.QWidget()
        self._starting_num_widget_layout = QtWidgets.QHBoxLayout(self.starting_num_widget)
        self._starting_num_widget_layout.setContentsMargins(0,5,0,0)
        self._starting_num_widget_layout.setSpacing(7)
        start_num_lbl = QtWidgets.QLabel('Next number:')
        start_num_lbl.setMinimumWidth(100)
        start_num_lbl.setAlignment(QtCore.Qt.AlignRight)
        self.starting_num_int = IntegerTextField()
        self.starting_num_int.setText('1')
        self.starting_num_int.setMinimumWidth(50)
        self.starting_num_int.setMaximumWidth(50)
        self._starting_num_widget_layout.addWidget(start_num_lbl)
        self._starting_num_widget_layout.addWidget(self.starting_num_int)
        self._starting_num_widget_layout.addSpacerItem(HorizontalSpacerItem())
        
        self._groupBoxFileNamingOptionsLayout.addWidget(self.starting_num_widget)


        self.minimum_digits_widget = QtWidgets.QWidget()
        self._minimum_digits_widget_layout = QtWidgets.QHBoxLayout(self.minimum_digits_widget)
        self._minimum_digits_widget_layout.setContentsMargins(0,5,0,0)
        self._minimum_digits_widget_layout.setSpacing(7)
        min_digits_lbl = QtWidgets.QLabel('Minimum digits:')
        min_digits_lbl.setMinimumWidth(100)
        min_digits_lbl.setAlignment(QtCore.Qt.AlignRight)
        self.min_digits_int = QtWidgets.QComboBox()
        self.min_digits_int.addItems(['1','2','3','4','5','6'])
        self.min_digits_int.setMinimumWidth(50)
        self.min_digits_int.setMaximumWidth(50)
        self._minimum_digits_widget_layout.addWidget(min_digits_lbl)
        self._minimum_digits_widget_layout.addWidget(self.min_digits_int)
        self._minimum_digits_widget_layout.addSpacerItem(HorizontalSpacerItem())
        
        self._groupBoxFileNamingOptionsLayout.addWidget(self.minimum_digits_widget)

        self.add_date_widget = QtWidgets.QWidget()
        self._add_date_widget_layout = QtWidgets.QHBoxLayout(self.add_date_widget)
        self._add_date_widget_layout.setContentsMargins(0,5,0,0)
        self._add_date_widget_layout.setSpacing(0)
        self.add_date_cbx = QtWidgets.QCheckBox('Add date')
        self._add_date_widget_layout.addWidget(self.add_date_cbx)
        
        self._groupBoxFileNamingOptionsLayout.addWidget(self.add_date_widget)

        self.date_format_widget = QtWidgets.QWidget()
        self._date_format_widget_layout = QtWidgets.QHBoxLayout(self.date_format_widget)
        self._date_format_widget_layout.setContentsMargins(0,5,0,0)
        self._date_format_widget_layout.setSpacing(7)
        date_format_lbl = QtWidgets.QLabel('Format:')
        #date_format_lbl.setMinimumWidth(55)
        date_format_lbl.setAlignment(QtCore.Qt.AlignRight)
        self.date_format_cmb = QtWidgets.QComboBox()
        self.date_format_cmb.addItems( ['YYYYMMDD',
                                        'YYYY-MM-DD',
                                        'YYYY-Month-DD',
                                        'Month-DD-YYYY'])

        self.date_format_cmb.setMinimumWidth(134)
        self.date_format_cmb.setMaximumWidth(134)
        self._date_format_widget_layout.addWidget(date_format_lbl)
        self._date_format_widget_layout.addWidget(self.date_format_cmb)
        self._date_format_widget_layout.addSpacerItem(HorizontalSpacerItem())

        self._groupBoxFileNamingOptionsLayout.addWidget(self.date_format_widget)

        self.add_time_widget = QtWidgets.QWidget()
        self._add_time_widget_layout = QtWidgets.QHBoxLayout(self.add_time_widget)
        self._add_time_widget_layout.setContentsMargins(0,5,0,0)
        self._add_time_widget_layout.setSpacing(0)
        self.add_time_cbx = QtWidgets.QCheckBox('Add time')
        self._add_time_widget_layout.addWidget(self.add_time_cbx)
        
        self._groupBoxFileNamingOptionsLayout.addWidget(self.add_time_widget)

        self.time_format_widget = QtWidgets.QWidget()
        self._time_format_widget_layout = QtWidgets.QHBoxLayout(self.time_format_widget)
        self._time_format_widget_layout.setContentsMargins(0,5,0,0)
        self._time_format_widget_layout.setSpacing(7)
        time_format_lbl = QtWidgets.QLabel('Format:')
        #time_format_lbl.setMinimumWidth(55)
        time_format_lbl.setAlignment(QtCore.Qt.AlignRight)
        self.time_format_cmb = QtWidgets.QComboBox()
        self.time_format_cmb.addItems( ['hh-mm-ss (24h)',
                                        'hh-mm-ss AM/PM'])
        self.time_format_cmb.setMinimumWidth(134)
        self.time_format_cmb.setMaximumWidth(134)
        self._time_format_widget_layout.addWidget(time_format_lbl)
        self._time_format_widget_layout.addWidget(self.time_format_cmb)
        self._time_format_widget_layout.addSpacerItem(HorizontalSpacerItem())

        self._groupBoxFileNamingOptionsLayout.addWidget(self.time_format_widget)

        self.dt_format_widget = QtWidgets.QWidget()
        self.dt_format_widget.setMaximumHeight(50)
        self._dt_format_widget_layout = QtWidgets.QHBoxLayout(self.dt_format_widget)
        self._dt_format_widget_layout.setContentsMargins(0,5,0,0)
        self._dt_format_widget_layout.setSpacing(7)
        dt_format_lbl = QtWidgets.QLabel('Place Date/Time in:')
        #dt_format_lbl.setMinimumWidth(100)
        dt_format_lbl.setAlignment(QtCore.Qt.AlignRight)
        self.dt_format_rad = QtWidgets.QWidget()
        self.dt_format_rad.setMinimumWidth(60)
        self.dt_format_rad.setMaximumWidth(60)
        self._dt_format_rad_layout = QtWidgets.QVBoxLayout(self.dt_format_rad)
        self._dt_format_rad_layout.setContentsMargins(0,0,0,0)
    
        self.prefix_rad = QtWidgets.QRadioButton('Prefix')
        self.suffix_rad = QtWidgets.QRadioButton('Suffix')
        
        self._dt_format_rad_layout.addWidget(self.prefix_rad)
        self._dt_format_rad_layout.addWidget(self.suffix_rad)
        self.prefix_rad.setChecked(True)

        self._dt_format_widget_layout.addWidget(dt_format_lbl)
        self._dt_format_widget_layout.addWidget(self.dt_format_rad)
        self._dt_format_widget_layout.addSpacerItem(HorizontalSpacerItem())

        self._groupBoxFileNamingOptionsLayout.addWidget(self.dt_format_widget)

        self._groupBoxFileNamingOptionsLayout.addWidget(QtWidgets.QLabel('Autoexport'))

        self.pattern_types_gc = QtWidgets.QWidget()
        self.xy_cb = QtWidgets.QCheckBox('.xy')
        self.chi_cb = QtWidgets.QCheckBox('.chi')
        self.dat_cb = QtWidgets.QCheckBox('.dat')
        self.fxye_cb = QtWidgets.QCheckBox('.fxye')
        self.png_cb = QtWidgets.QCheckBox('.png')
        self._pattern_types_gb_layout = QtWidgets.QGridLayout(self.pattern_types_gc)
        self._pattern_types_gb_layout.setContentsMargins(0,0,0,0)
        self._pattern_types_gb_layout.setSpacing(7)
        self._pattern_types_gb_layout.addWidget(self.xy_cb,0,0)
        self._pattern_types_gb_layout.addWidget(self.chi_cb,0,1)
        self._pattern_types_gb_layout.addWidget(self.png_cb,0,2)
        self._pattern_types_gb_layout.addWidget(self.dat_cb,1,0)
        self._pattern_types_gb_layout.addWidget(self.fxye_cb,1,1)


        self._groupBoxFileNamingOptionsLayout.addWidget(self.pattern_types_gc)
        

        # scroll area stuff
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setStyleSheet("QScrollArea { border: 0px;}")
        self.scroll.setMaximumWidth(225)
        
        self.content = QtWidgets.QWidget()
        self.scroll.setWidget(self.content)
        self.scroll.setWidgetResizable(True)
        self.vlay = QtWidgets.QVBoxLayout(self.content)
        self.vlay.setContentsMargins(0,0,0,0)

        
        
        box_dict = {'acquisition':['Acquisition',self.groupBoxAcq, 'expanded'],
                    'elapsed_time':['Elapsed time',self.groupBoxElapsed, 'expanded'],
                    'save_data':['Save data file',self.groupBoxSaveDataFile, 'collapsed'],
                    'file_name_options':['File saving options',self.groupBoxFileNamingOptions, 'collapsed'],
                    
                    'rois':['Regions of interest',self.groupBoxROIs, 'expanded'],
                    'xrf':['Fluorescence markers',self.groupBoxXRF, 'collapsed'],
                    'v_scale':['Vertical scale',self.groupBoxVerticalScale, 'collapsed'],
                    'h_scale':['Horizontal scale',self.groupBoxHorizontalScale, 'collapsed']}

        for b in box_dict:
            box = CollapsibleBox(box_dict[b][0])
            self.vlay.addWidget(box)
            lay = QtWidgets.QVBoxLayout()
            lay.setContentsMargins(0,0,0,0)
            
            widget = box_dict[b][1]
            lay.addWidget(widget)
            state = box_dict[b][2] 
            box.setContentLayout(lay, state)
            
            
        self.vlay.addStretch()

        self.ControlsLayout.addWidget(self.scroll)

        #spacerItem = QtWidgets.QSpacerItem(20, 1000, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        #self.ControlsLayout.addItem(spacerItem)

        # end scroll erea stuff

        self._Layout.addLayout(self.ControlsLayout)


        self.DisplayWidget = QtWidgets.QWidget()
        self._DisplayLayout = QtWidgets.QVBoxLayout(self.DisplayWidget)
        self._DisplayLayout.setContentsMargins(0,0,0,0)
        self._DisplayLayout.setSpacing(0)
        self._DisplayLayout.setObjectName("_DisplayLayout")

        self.live_or_file_widget = QtWidgets.QWidget()
        self._live_or_file_widget_layout = QtWidgets.QHBoxLayout(self.live_or_file_widget)
        self._live_or_file_widget_layout.setSpacing(0)
        self._live_or_file_widget_layout.setContentsMargins(0,0,0,0)

        self.live_or_file_view_tab_bar = QtWidgets.QButtonGroup()
        self.live_view_btn = QtWidgets.QPushButton("Live")
        self.live_view_btn.setObjectName('live_view_btn')
        self.live_view_btn.setCheckable(True)
        self.live_view_btn.setChecked(False)
        self.live_view_btn.setEnabled(False)
        self.live_view_btn.setMinimumWidth(90)
        self.file_view_btn = QtWidgets.QPushButton("File")
        self.file_view_btn.setObjectName('file_view_btn')
        self.file_view_btn.setCheckable(True)
        self.file_view_btn.setChecked(False)
        self.file_view_btn.setEnabled(False)
        self.file_view_btn.setMinimumWidth(90)
        self.live_or_file_view_tab_bar.addButton(self.live_view_btn)
        self.live_or_file_view_tab_bar.addButton(self.file_view_btn)
        self._live_or_file_widget_layout.addWidget(self.live_view_btn)
        self._live_or_file_widget_layout.addWidget(self.file_view_btn)
        self._live_or_file_widget_layout.addSpacerItem(HorizontalSpacerItem())

        self._DisplayLayout.addWidget(self.live_or_file_widget)
        

        self.plot_toolbar_top_widget = QtWidgets.QWidget()
        self.plot_toolbar_top_widget.setObjectName('plot_toolbar_top_widget')
        self.plot_toolbar_top_widget.setStyleSheet( '''
                                                    #plot_toolbar_top_widget {
                                                        background-color: #101010;
                                                    }
                                                        ''')
        self._plot_toolbar_top_widget_layout = QtWidgets.QHBoxLayout(self.plot_toolbar_top_widget)
        self._plot_toolbar_top_widget_layout.setSpacing(15)
        self._plot_toolbar_top_widget_layout.setContentsMargins(8,2,8,2)
        self.plot_toolbar_top_widget.setMinimumHeight(21)
        self.plot_toolbar_top_widget.setMaximumHeight(21)

        

        self.save_pattern_btn = FlatButton()
        self.save_pattern_btn.setToolTip("Save Pattern")
        self.as_overlay_btn = FlatButton('As Overlay')
        self.as_bkg_btn = FlatButton('As Bkg')
        self.load_calibration_btn = FlatButton('Load Calibration')
        self.calibration_lbl = QtWidgets.QLabel('None')

        
        '''
        self._plot_toolbar_top_widget_layout.addWidget(self.as_overlay_btn)
        self._plot_toolbar_top_widget_layout.addWidget(self.as_bkg_btn)
        self._plot_toolbar_top_widget_layout.addSpacerItem(HorizontalSpacerItem())
        self._plot_toolbar_top_widget_layout.addWidget(self.load_calibration_btn)
        self._plot_toolbar_top_widget_layout.addWidget(self.calibration_lbl)'''



        self._DisplayLayout.addWidget(self.plot_toolbar_top_widget)
        
        self.plot_toolbar_right_widget = QtWidgets.QWidget()
        self.plot_toolbar_right_widget.setObjectName('plot_toolbar_right_widget')
        self.plot_toolbar_right_widget.setStyleSheet( '''
                                                    #plot_toolbar_right_widget {
                                                        background-color: #101010;
                                                    }
                                                        ''')
        self._plot_toolbar_right_widget_layout = QtWidgets.QVBoxLayout(self.plot_toolbar_right_widget)
        self._plot_toolbar_right_widget_layout.setSpacing(4)
        self._plot_toolbar_right_widget_layout.setContentsMargins(0,0,0,6)
        self.plot_toolbar_right_widget.setMinimumWidth(10)
        self.plot_toolbar_right_widget.setMaximumWidth(10)

        self.tth_btn = CheckableFlatButton(u"2θ")
        self.q_btn = CheckableFlatButton('Q')
        self.d_btn = CheckableFlatButton('d')
        self.unit_btn_group = QtWidgets.QButtonGroup()
        self.unit_btn_group.addButton(self.tth_btn)
        self.unit_btn_group.addButton(self.q_btn)
        self.unit_btn_group.addButton(self.d_btn)
        self.background_btn = CheckableFlatButton('bg')
        self.background_inspect_btn = CheckableFlatButton('I')
        #self.antialias_btn = CheckableFlatButton('AA')
        self.auto_range_btn = CheckableFlatButton('A')

        '''self._plot_toolbar_right_widget_layout.addWidget(self.tth_btn)
        self._plot_toolbar_right_widget_layout.addWidget(self.q_btn)
        self._plot_toolbar_right_widget_layout.addWidget(self.d_btn)
        self._plot_toolbar_right_widget_layout.addSpacerItem(VerticalSpacerItem())
        self._plot_toolbar_right_widget_layout.addWidget(self.background_btn)
        self._plot_toolbar_right_widget_layout.addWidget(self.background_inspect_btn)
        self._plot_toolbar_right_widget_layout.addSpacerItem(VerticalSpacerItem())
        #self._plot_toolbar_right_widget_layout.addWidget(self.antialias_btn)
        self._plot_toolbar_right_widget_layout.addSpacerItem(VerticalSpacerItem())
        self._plot_toolbar_right_widget_layout.addWidget(self.auto_range_btn)'''


        self.plot_widget = QtWidgets.QWidget()
        self.plot_widget.setObjectName('plot_widget')
        self.plot_widget.setStyleSheet( '''
                                                    #plot_widget {
                                                        background-color: #101010;
                                                    }
                                                        ''')
        self._plot_widget_layout = QtWidgets.QHBoxLayout(self.plot_widget)
        self._plot_widget_layout.setSpacing(0)
        self._plot_widget_layout.setContentsMargins(10,0,5,10)
        
        
        '''self.pg = PltWidget(self.plot_widget,
                        toolbar_widgets=[self.plot_toolbar_right_widget,
                                        self.plot_toolbar_top_widget])
        self.pg.setMinimumSize(QtCore.QSize(205, 100))
        self.pg.setObjectName("pg")

        self._plot_widget_layout.addWidget(self.pg)
        self._plot_widget_layout.addWidget(self.plot_toolbar_right_widget)'''


        self._DisplayLayout.addWidget(self.plot_widget)

        self.CursorsLayout = QtWidgets.QHBoxLayout()
        self.CursorsLayout.setContentsMargins(0,6,0,6)
        self.CursorsLayout.setObjectName("CursorsLayout")
        self.indexLabel = QtWidgets.QLabel()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.indexLabel.sizePolicy().hasHeightForWidth())
        self.indexLabel.setSizePolicy(sizePolicy)
        self.indexLabel.setMinimumSize(QtCore.QSize(150, 0))
        self.indexLabel.setMaximumSize(QtCore.QSize(250, 16777215))
        self.indexLabel.setText("")
        self.indexLabel.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.indexLabel.setObjectName("indexLabel")
        self.CursorsLayout.addWidget(self.indexLabel)
        self.cursorLabel = QtWidgets.QLabel()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cursorLabel.sizePolicy().hasHeightForWidth())
        self.cursorLabel.setSizePolicy(sizePolicy)
        self.cursorLabel.setMinimumSize(QtCore.QSize(150, 0))
        self.cursorLabel.setMaximumSize(QtCore.QSize(250, 16777215))
        self.cursorLabel.setText("")
        self.cursorLabel.setObjectName("cursorLabel")
        self.CursorsLayout.addWidget(self.cursorLabel)
        self._DisplayLayout.addLayout(self.CursorsLayout)
        self._Layout.addWidget(self.DisplayWidget)



        self.verticalLayout_4.addLayout(self._Layout)
        hpMCA.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(hpMCA)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1236, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuControl = QtWidgets.QMenu(self.menubar)
        self.menuControl.setObjectName("menuControl")
        self.menuDisplay = QtWidgets.QMenu(self.menubar)
        self.menuDisplay.setObjectName("menuDisplay")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        hpMCA.setMenuBar(self.menubar)
        self.actionBackground = QtWidgets.QAction(hpMCA)
        self.actionBackground.setObjectName("actionBackground")
        self.actionSave_As = QtWidgets.QAction(hpMCA)
        self.actionSave_As.setEnabled(False)
        self.actionSave_As.setObjectName("actionSave_As")
        self.actionPrint = QtWidgets.QAction(hpMCA)
        self.actionPrint.setObjectName("actionPrint")
        self.actionPreferences = QtWidgets.QAction(hpMCA)
        self.actionPreferences.setObjectName("actionPreferences")
        self.actionEvironment = QtWidgets.QAction(hpMCA)
        self.actionEvironment.setObjectName("actionEvironment")
        self.actionEvironment.setEnabled(False)
        self.actionMultiSpectra = QtWidgets.QAction(hpMCA)
        self.actionMultiSpectra.setObjectName("actionMultiSpectra")
        self.actionAmorphous = QtWidgets.QAction(hpMCA)
        self.actionAmorphous.setObjectName("actionAmorphous")
        self.actionExit = QtWidgets.QAction(hpMCA)
        self.actionExit.setObjectName("actionExit")
        self.actionPresets = QtWidgets.QAction(hpMCA)
        self.actionPresets.setObjectName("actionPresets")
        self.actionCalibrate_energy = QtWidgets.QAction(hpMCA)
        self.actionCalibrate_energy.setEnabled(False)
        self.actionCalibrate_energy.setObjectName("actionCalibrate_energy")
        self.actionPreferences_2 = QtWidgets.QAction(hpMCA)
        self.actionPreferences_2.setObjectName("actionPreferences_2")
        self.actionManualTth = QtWidgets.QAction(self)
        self.actionManualTth.setText("Set 2theta...")
        self.actionManualTth.setEnabled(False)
        self.actionManualWavelength = QtWidgets.QAction(self)
        self.actionManualWavelength.setText("Set wavelength...")
        self.actionManualWavelength.setEnabled(False)
        self.actionLoadCalibration = QtWidgets.QAction(self)
        self.actionLoadCalibration.setText("Load energy calibration...")
        self.actionLoadCalibration.setEnabled(False)

        self.actionShowCalibration = QtWidgets.QAction(self)
        self.actionShowCalibration.setText("All calibration parameters")
        self.actionShowCalibration.setEnabled(False)
        
        self.actionJCPDS = QtWidgets.QAction(hpMCA)
        self.actionJCPDS.setEnabled(False)
        self.actionJCPDS.setObjectName("actionJCPDS")
        self.actionFit_peaks = QtWidgets.QAction(hpMCA)
        self.actionFit_peaks.setObjectName("actionFit_peaks")
        self.actionAbout = QtWidgets.QAction(hpMCA)
        self.actionAbout.setObjectName("actionAbout")
        self.actionOpen_detector = QtWidgets.QAction(hpMCA)
        self.actionOpen_detector.setObjectName("actionOpen_detector")
        self.actionOpen_file = QtWidgets.QAction(hpMCA)
        self.actionOpen_file.setObjectName("actionOpen_file")
        self.actionOpen_folder = QtWidgets.QAction(hpMCA)
        self.actionOpen_folder.setObjectName("actionOpen_folder")
        self.actionCalibrate_2theta = QtWidgets.QAction(hpMCA)
        self.actionCalibrate_2theta.setEnabled(False)
        self.actionCalibrate_2theta.setObjectName("actionCalibrate_2theta")
        self.actionFluor = QtWidgets.QAction(hpMCA)
        self.actionFluor.setEnabled(False)
        self.actionFluor.setObjectName("actionFluor")
        self.actionROIs = QtWidgets.QAction(hpMCA)
        self.actionROIs.setEnabled(False)
        self.actionROIs.setObjectName("actionROIs")
        self.actionOverlay = QtWidgets.QAction(hpMCA)
        self.actionOverlay.setEnabled(False)
        self.actionOverlay.setObjectName("actionOverlay")
        self.actionExport_pattern = QtWidgets.QAction(hpMCA)
        self.actionExport_pattern.setEnabled(False)
        self.actionExport_pattern.setObjectName("actionExport_pattern")
        self.actionSave_next = QtWidgets.QAction(hpMCA)
        self.actionSave_next.setEnabled(False)
        self.actionSave_next.setObjectName("actionSave_next")
        self.actionLatticeRefinement = QtWidgets.QAction(hpMCA)
        self.actionLatticeRefinement.setEnabled(False)
        self.actionLatticeRefinement.setObjectName("actionLatticeRefinement")
        self.actionDisplayPrefs = QtWidgets.QAction(hpMCA)
        self.actionDisplayPrefs.setObjectName("actionDisplayPrefs")
        self.actionRoiPrefs = QtWidgets.QAction(hpMCA)
        self.actionRoiPrefs.setObjectName("actionRoiPrefs")
        self.menuFile.addAction(self.actionOpen_file)
        self.menuFile.addAction(self.actionOpen_folder)
        self.menuFile.addAction(self.actionOpen_detector)
        self.menuFile.addAction(self.actionOverlay)
        #self.menuFile.addAction(self.actionSave_next)
        self.menuFile.addAction(self.actionSave_As)
        self.menuFile.addAction(self.actionExport_pattern)
        self.menuFile.addAction(self.actionPreferences)
        self.menuFile.addAction(self.actionEvironment)
        self.menuFile.addAction(self.actionMultiSpectra)
        #self.menuFile.addAction(self.actionAmorphous)

        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menuControl.addAction(self.actionCalibrate_energy)
        self.menuControl.addAction(self.actionCalibrate_2theta)
        self.menuControl.addAction(self.actionManualTth)
        self.menuControl.addAction(self.actionManualWavelength)
        self.menuControl.addAction(self.actionLoadCalibration)
        #self.menuControl.addAction(self.actionShowCalibration)
        self.menuDisplay.addAction(self.actionJCPDS)
        self.menuDisplay.addAction(self.actionFluor)
        self.menuDisplay.addAction(self.actionROIs)
        self.menuDisplay.addAction(self.actionLatticeRefinement)
        self.menuDisplay.addAction(self.actionDisplayPrefs)
        self.menuDisplay.addAction(self.actionRoiPrefs)
        self.menuHelp.addAction(self.actionAbout)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuControl.menuAction())
        self.menubar.addAction(self.menuDisplay.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(hpMCA)
        QtCore.QMetaObject.connectSlotsByName(hpMCA)
        self.style_widgets()

    def style_widgets(self):
        self.tth_btn.setChecked(True)
        #self.antialias_btn.setChecked(True)
        self.auto_range_btn.setChecked(True)

        self.DisplayWidget.setStyleSheet("""
            #cursorLabel {
                
                color: #00EE00;
            }
       
	    """)

        self.groupBoxElapsed.setStyleSheet("""
            pvQMessageButton{
                
                padding: 0px;
            }
       
	    """)
        
        self.groupBoxHorizontalScale.setStyleSheet("""
            QPushButton{
                
                border-radius: 0px;
            }
            #radioE {

                border-top-left-radius:5px;
                border-bottom-left-radius:5px;
            }
            #radioChannel {

                border-top-right-radius:5px;
                border-bottom-right-radius:5px;
            }
       
	    """)

        self.plot_toolbar_top_widget.setStyleSheet("""
            #pattern_frame, #plot_toolbar_top_widget, QLabel {
                background: #101010;
                color: yellow;
            }
            #pattern_right_control_widget QPushButton{
                padding: 0px;
	            padding-right: 1px;
	            border-radius: 3px;
            }
	    """)

        self.plot_toolbar_right_widget.setStyleSheet("""
            #plot_toolbar_right_widget, QLabel {
                background: #101010;
                color: yellow;
            }
            #plot_toolbar_right_widget QPushButton{
                padding: 0px;
	            padding-right: 1px;
	            border-radius: 3px;
            }
	    """)

        right_controls_button_width = 25
        self.tth_btn.setMaximumWidth(right_controls_button_width)
        self.q_btn.setMaximumWidth(right_controls_button_width)
        self.d_btn.setMaximumWidth(right_controls_button_width)
        self.background_btn.setMaximumWidth(right_controls_button_width)
        self.background_inspect_btn.setMaximumWidth(right_controls_button_width)
        #self.antialias_btn.setMaximumWidth(right_controls_button_width)
        self.auto_range_btn.setMaximumWidth(right_controls_button_width)

        self.save_pattern_btn.setIcon(QtGui.QIcon(os.path.join(icons_path, 'save.ico')))
        self.save_pattern_btn.setIconSize(QtCore.QSize(13, 13))
        self.save_pattern_btn.setMaximumWidth(right_controls_button_width)

    def retranslateUi(self, hpMCA):
        _translate = QtCore.QCoreApplication.translate
        hpMCA.setWindowTitle(_translate("hpMCA", "hpMCA"))

        


        
        self.btnOn.setText(_translate("hpMCA", "On"))
        self.btnOff.setText(_translate("hpMCA", "Off"))
        self.btnErase.setText(_translate("hpMCA", "Erase"))
        
        self.lbl_elapsed_header.setText(_translate("hpMCA", "Presets"))
        self.lblLiveTime_lbl.setText(_translate("hpMCA", "Live"))
        self.lblRealTime_lbl.setText(_translate("hpMCA", "Real"))
        self.lblLiveTime.setText(_translate("hpMCA", "0"))
        self.lblRealTime.setText(_translate("hpMCA", "0"))
        
        self.btnROIadd.setText(_translate("hpMCA", "Add"))
        self.btnROIdelete.setText(_translate("hpMCA", "Delete"))
        self.btnROIclear.setText(_translate("hpMCA", "Clear All"))
        self.btnROIprev.setText(_translate("hpMCA", "<"))
        self.btnROInext.setText(_translate("hpMCA", ">"))
       
        self.btnKLMprev.setText(_translate("hpMCA", "<"))
        self.btnKLMnext.setText(_translate("hpMCA", ">"))
        
        self.radioLog.setText(_translate("hpMCA", "Log"))
        self.radioLin.setText(_translate("hpMCA", "Linear"))
        
        self.radioE.setText(_translate("hpMCA", "E"))
        self.radioq.setText(_translate("hpMCA", "q"))
        self.radioChannel.setText(_translate("hpMCA", "Channel"))
        self.radiod.setText(_translate("hpMCA", "d"))
        self.radiotth.setText(_translate("hpMCA", f'2\N{GREEK SMALL LETTER THETA}'))
        self.menuFile.setTitle(_translate("hpMCA", "File"))
        self.menuControl.setTitle(_translate("hpMCA", "Control"))
        self.menuDisplay.setTitle(_translate("hpMCA", "Display"))
        self.menuHelp.setTitle(_translate("hpMCA", "Help"))
        self.actionBackground.setText(_translate("hpMCA", "Background"))
        self.actionSave_As.setText(_translate("hpMCA", "Save As"))
        self.actionPrint.setText(_translate("hpMCA", "Print"))
        self.actionPreferences.setText(_translate("hpMCA", "Preferences"))
        self.actionEvironment.setText(_translate("hpMCA", "Environment"))
        self.actionMultiSpectra.setText(_translate("hpMCA", "Multiple spectra"))
        self.actionAmorphous.setText(_translate("hpMCA", "Amorphous S(q)"))
        
        self.actionExit.setText(_translate("hpMCA", "Exit"))
        self.actionPresets.setText(_translate("hpMCA", "Presets..."))
        self.actionCalibrate_energy.setText(_translate("hpMCA", "Calibrate energy..."))
        self.actionPreferences_2.setText(_translate("hpMCA", "Preferences..."))
        self.actionJCPDS.setText(_translate("hpMCA", "Phase"))
        self.actionFit_peaks.setText(_translate("hpMCA", "Fit peaks..."))
        self.actionAbout.setText(_translate("hpMCA", "About"))
        self.actionOpen_detector.setText(_translate("hpMCA", "Open detector..."))
        self.actionOpen_file.setText(_translate("hpMCA", "Open file..."))
        self.actionOpen_folder.setText(_translate("hpMCA", "Open folder..."))
        self.actionCalibrate_2theta.setText(_translate("hpMCA", "Calibrate 2theta..."))
        self.actionFluor.setText(_translate("hpMCA", "Fluorescence"))
        self.actionROIs.setText(_translate("hpMCA", "Regions of interest"))
        self.actionOverlay.setText(_translate("hpMCA", "Overlay"))
        self.actionExport_pattern.setText(_translate("hpMCA", "Export pattern"))
        self.actionSave_next.setText(_translate("hpMCA", "Save next"))
        self.actionLatticeRefinement.setText(_translate("hpMCA", "Lattice refinement"))
        self.actionDisplayPrefs.setText(_translate("hpMCA", "Colors options"))
        self.actionRoiPrefs.setText(_translate("hpMCA", "ROI options"))


