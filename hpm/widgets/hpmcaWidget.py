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



import os, os.path, sys, platform, copy

from PyQt6 import uic, QtWidgets,QtCore
from PyQt6.QtCore import QObject, pyqtSignal, Qt
from PyQt6.QtWidgets import QMainWindow

from hpm.widgets.PltWidget import PltWidget

#qtCreatorFile = 'hpMCA.ui'
#Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
_platform = platform.system()

from hpm.widgets.hpMCA_ui import Ui_hpMCA
from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, HorizontalLine, IntegerTextField, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight, CheckableFlatButton


class hpMCAWidget(QMainWindow, Ui_hpMCA):
    file_dragged_in_signal = pyqtSignal(str)
    folder_dragged_in_signal = pyqtSignal(str)
    key_signal = pyqtSignal(str)
    def __init__(self, app):
        QMainWindow.__init__(self)
        Ui_hpMCA.__init__(self)
        self.app = app
        desktop = QtWidgets.QDesktopWidget()
        dpi = desktop.physicalDpiX()
        self.retina_display = False
        if dpi > 96:
            self.retina_display = True
        self.setupUi(self)
        
        self._plot_toolbar_top_widget_layout.addSpacerItem(HorizontalSpacerItem())
        self.element_number_cmb = QtWidgets.QComboBox()
        self.element_number_cmb.setStyleSheet("QComboBox { color: white; }")
        self.element_number_lbl = QtWidgets.QLabel('Detector')
        self._plot_toolbar_top_widget_layout.addWidget(self.element_number_lbl)
        self._plot_toolbar_top_widget_layout.addWidget(self.element_number_cmb)
        

        self.pg = PltWidget(self.plot_widget,
                        toolbar_widgets=[self.plot_toolbar_right_widget,
                                        self.plot_toolbar_top_widget], retina_display=self.retina_display)
        self.pg.setMinimumSize(QtCore.QSize(205, 100))
        self.pg.setObjectName("pg")

        self._plot_widget_layout.addWidget(self.pg)
        self._plot_widget_layout.addWidget(self.plot_toolbar_right_widget)

        self.scales_btns = {'E':self.radioE,
                            'q':self.radioq,
                            'd':self.radiod,
                            'Channel':self.radioChannel,
                            '2 theta':self.radiotth}
        
        self.add_menu_items()
        self.setAcceptDrops(True)

    def hide_det_cmb(self, hide : bool):
        if hide:
            self.element_number_cmb.hide()
            self.element_number_lbl.hide()
        else:
            self.element_number_cmb.show()
            self.element_number_lbl.show()

    def closeEvent(self, QCloseEvent, *event):
        self.app.closeAllWindows()

    def add_menu_items(self):
        
        self.actionPresets = QtWidgets.QAction(self)
        self.actionPresets.setText("Presets...")
        self.actionPresets.setEnabled(False)
        self.menuControl.addAction(self.actionPresets)

        self.actionhklGen = QtWidgets.QAction(self)
        self.actionhklGen.setText("hklGen")
        self.actionhklGen.setEnabled(False)
        #self.menuDisplay.addAction(self.actionhklGen)

    def menu_items_set_enabled(self, enabled):
        self.actionSave_As.setEnabled(enabled)
        self.actionExport_pattern.setEnabled(enabled)
        self.actionJCPDS.setEnabled(enabled)
        self.actionCalibrate_energy.setEnabled(enabled)
        self.actionCalibrate_2theta.setEnabled(enabled)
        self.actionROIs.setEnabled(enabled)
        self.actionOverlay.setEnabled(enabled)
        self.actionFluor.setEnabled(enabled)
        self.actionLatticeRefinement.setEnabled(enabled)
        self.actionManualTth.setEnabled(enabled)
        self.actionPresets.setEnabled(enabled)
        self.actionhklGen.setEnabled(enabled)
        self.actionEvironment.setEnabled(enabled)
        self.actionLoadCalibration.setEnabled(enabled)
        self.actionShowCalibration.setEnabled(enabled)

    def keyPressEvent(self, e):
        sig = None
        if e.key() == Qt.Key_Up:
            sig = 'up'
        if e.key() == Qt.Key_Down:
            sig = 'down'                
        if e.key() == Qt.Key_Delete:
            sig = 'delete'
        if e.key() == Qt.Key_Right:
            sig = 'right'   
        if e.key() == Qt.Key_Left:
            sig = 'left'   
        if e.key() == Qt.Key_Backspace:
            sig = 'delete'  
        if e.key() == Qt.Key_Shift:
            sig = 'shift_press'     
         
        if sig is not None:
            self.key_signal.emit(sig)
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        sig = None
        if e.key() == Qt.Key_Shift:
            sig = 'shift_release' 
        if sig is not None:
            self.key_signal.emit(sig)
        else:
            super().keyPressEvent(e)

    def set_file_naming_settings(self, settings):
        base_name = settings.base_name
        increment_file_name = settings.increment_file_name 
        starting_number = settings.starting_number             
        minimum_digits = settings.minimum_digits            
        add_date = settings.add_date            
        add_time = settings.add_time            
        d_format = settings.d_format            
        t_format = settings.t_format            
        dt_append_possition = settings.dt_append_possition 
        export_xy            = settings.export_xy
        export_chi           = settings.export_chi
        export_dat           = settings.export_dat
        export_fxye          = settings.export_fxye
        export_png           = settings.export_png
        
        self.groupBoxFileNamingOptions.blockSignals(True)
        self.file_name_ebx.setEditText(base_name)
        self.increment_file_name_cbx.setChecked(increment_file_name)
        self.starting_num_int.setText(str(starting_number))
        self.min_digits_int.setCurrentIndex(minimum_digits)
        self.add_date_cbx.setChecked(add_date)
        self.add_time_cbx.setChecked(add_time)
        self.date_format_cmb.setCurrentIndex(d_format)
        self.time_format_cmb.setCurrentIndex(t_format)
        if dt_append_possition == 0:
            self.prefix_rad.setChecked(True)
        elif dt_append_possition == 1:
            self.suffix_rad.setChecked(True)

        self.xy_cb  .setChecked(export_xy)
        self.chi_cb .setChecked(export_chi) 
        self.dat_cb .setChecked(export_dat)
        self.fxye_cb.setChecked(export_fxye)
        self.png_cb .setChecked(export_png)

        self.groupBoxFileNamingOptions.blockSignals(False)


    def get_file_naming_settings(self):
        settings = dict()
        settings['base_name'] = self.file_name_ebx.currentText()
        settings['increment_file_name'] = self.increment_file_name_cbx.isChecked()
        settings['starting_number'] =     int(self.starting_num_int.text())      
        settings['minimum_digits'] =      self.min_digits_int.currentIndex()    
        settings['add_date'] =            self.add_date_cbx.isChecked()
        settings['add_time'] =            self.add_time_cbx.isChecked()
        settings['d_format'] =            self.date_format_cmb.currentIndex()  
        settings['t_format'] =            self.time_format_cmb.currentIndex()  
        settings['dt_append_possition'] = int(self.suffix_rad.isChecked())
        settings['export_xy']=  self.xy_cb  .isChecked()
        settings['export_chi']= self.chi_cb .isChecked() 
        settings['export_dat']= self.dat_cb .isChecked()
        settings['export_fxye']=self.fxye_cb.isChecked()
        settings['export_png']= self.png_cb .isChecked()
        
        return settings

    def set_scales_enabled_states(self, enabled=['E','q','d','Channel']):
        for btn in self.scales_btns:
            self.scales_btns[btn].setEnabled(btn in enabled)

    def get_selected_unit(self):
        horzScale = 'Channel'
        if self.radioE.isChecked() == True:
            horzScale = 'E'
        elif self.radioq.isChecked() == True:
            horzScale = 'q'
        
        elif self.radiod.isChecked() == True:
            horzScale = 'd'
        elif self.radiotth.isChecked() == True:
            horzScale = '2 theta'
        return horzScale

    def set_unit_btn(self, unit):
        if unit in self.scales_btns:
            btn = self.scales_btns[unit]
            btn.setChecked(True)

    ########################################################################################
    ########################################################################################

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls:
            if len(e.mimeData().urls()) != 1:
                e.ignore() 
            else:
                e.accept()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls:
            if len(e.mimeData().urls()) != 1:
                e.ignore() 
            else:
                e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        """
        If user drop files directly onto the widget
        File locations are stored in fname
        Emits file_dragged_in_signal with str arg fname
        :param e:
        :return:
        """
        if e.mimeData().hasUrls:
            e.setDropAction(QtCore.Qt.CopyAction)
            if len(e.mimeData().urls()) != 1:
                e.ignore() 
            else:
                e.accept()
                for url in e.mimeData().urls():   
                    fname = str(url.toLocalFile())
                self.file_dragged_in_signal.emit(fname)
        else:
            e.ignore()  

    

    



    

