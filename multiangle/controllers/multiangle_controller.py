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


import os.path, sys
from PyQt6 import uic, QtWidgets,QtCore, QtGui

from  PyQt6.QtWidgets import QMainWindow, QApplication, QInputDialog, QWidget, QLabel
from PyQt6.QtCore import QObject, pyqtSignal, Qt
import numpy as np
from functools import partial
import json
import copy
from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight
from hpm.widgets.UtilityWidgets import save_file_dialog, open_file_dialog, open_files_dialog
import utilities.hpMCAutilities as mcaUtil

from pathlib import Path
from utilities.HelperModule import calculate_color
from numpy import arange

from hpm.widgets.UtilityWidgets import save_file_dialog, open_file_dialog, open_files_dialog
from multiangle.widgets.multiangle_widget import multiangleWidget
from multiangle.models.multiangle_model import multiangleModel
from multiangle.models.pe_multi_angle_edxd_v2 import MultiangleSweep
from multiangle.models.epics_roi_monitor import epicsRoiMonitor

from multiangle.widgets.multiangle_widget import QCoverageWidget

############################################################

class multiangleController(QObject):
    def __init__(self, app, theme):
        super().__init__()
        self.app = app
        self.model = multiangleModel(self)
        self.sweep_controller = MultiangleSweep(self)
        self.widget = multiangleWidget(self.model)
         
        self.plot_widget = QCoverageWidget(self.model)
        
        self.create_connections()
        self.setStyle(theme)
        self.working_directories = mcaUtil.restore_folder_settings()
        self.epics_roi_monitor = epicsRoiMonitor(record_name='16bmb:aim_adc1')
        self.widget.show()

    def main_widget_closed_callback(self):
        self.app.closeAllWindows()

    def create_connections(self):
        self.widget.parameter_widget.sequence_changed_signal.connect(self.sequence_changed_callback)
        self.widget.clear_btn.clicked.connect(self.clear_callback)
        self.widget.widget_closed.connect(self.main_widget_closed_callback)
        self.widget.delete_btn.clicked.connect(self.delete_callback)
        self.widget.add_btn.clicked.connect(self.add_callback)
        self.widget.save_btn.clicked.connect(self.save_btn_callback)
        self.widget.load_btn.clicked.connect(self.load_btn_callback)
        self.widget.run_btn.clicked.connect(self.run_callback)
        self.widget.test_btn.clicked.connect(self.test_callback)
        self.widget.stop_btn.clicked.connect(self.stop_callback)
        self.sweep_controller.callbackSignal.connect(self.sweep_controller_signal_callback)
        self.widget.plot_btn.clicked.connect(self.plot_btn_callback)

        self.widget.tip_size_sb.valueChanged.connect(self.slit_param_callback)
        self.widget.tip_distance_sb.valueChanged.connect(self.slit_param_callback)
        self.widget.det_distance_sb.valueChanged.connect(self.slit_param_callback)

    def slit_param_callback(self, parameter_list):
        self.model.det_slit_distance = self.widget.det_distance_sb.value()
        self.model.tip_slit_distance = self.widget.tip_distance_sb.value()
        self.model.tip_slit_size = self.widget.tip_size_sb.value()
        self.model.recalc_collimation()
    

    def plot_btn_callback(self, *args, **kwargs):
        self.plot_widget.raise_widget()

    def save_btn_callback(self, *args, **kwargs):
        data = self.model.get_data()
        
        filename = save_file_dialog(
            self.widget, "Save multiangle scan settings",
            filter='Scan settings (*.scan);;Scan settings (*.json)')
        if filename is not '':
            if filename.endswith('.scan') or filename.endswith('.json'):
                self.save_settings(data, filename)

    def ask_to_clear(self):
        row_count = self.model.rowCount()
        if row_count>0:
            qm = QtWidgets.QMessageBox()
            ret = qm.question(self.widget,'Warning', "Clear current rows?", qm.Yes | qm.No)
            if ret == qm.Yes:
                self.clear_callback()

    def load_btn_callback(self, *args, **kwargs):
        filename = open_file_dialog(
            self.widget, "Load multiangle scan settings")
        if filename is not '':
            if filename.endswith('.json') or filename.endswith('.scan'):
                settings = self.load_settings(filename)
                self.ask_to_clear()
                
                for s in settings:
                    d = {'tth':s[0], 'stime':s[7], 'pvsize':s[1], 'phsize':s[2], 'svsize':s[3], 'shsize':s[4], 'dvsize':s[5], 'dhsize':s[6]}
                    self.model.add_row(d)
                self.plot_widget.update()

    def save_settings(self, data, filename):
        data_out = {'multiangle_settings':data}
        try:
            with open(filename, 'w') as outfile:
                json.dump(data_out, outfile,sort_keys=True, indent=4)
                outfile.close()
        except:
            pass

    def load_settings(self, filename):
        ok = False
        try:
            with open(filename) as f:
                openned_file = json.load(f)
            ok = True
        except:
            ok = False
        obj = None
        if ok:
            key = 'multiangle_settings'
            if key in openned_file:
                obj = openned_file[key]
        
        return obj
        

    def sweep_controller_signal_callback(self, *args, **kwargs):
        print(args)

    def run_callback(self, *args, **kwargs):   

        tth_set = self.model.get_data()
        
        print(tth_set)
        self.sweep_controller.set_exp_conditions(tth_set)
        rep = int(self.widget.iterations_control.text())
        self.sweep_controller.set_rep(rep)
        self.sweep_controller.start()

    def test_callback(self, *args, **kwargs):   
        header = self.widget.tth_tv.header()
        sort_column = header.sortIndicatorSection()
        sort_order = header.sortIndicatorOrder()
        #print(sort_order)
        tth_set = self.model.get_data()
      
        print(tth_set)

    def stop_callback(self, *args, **kwargs):
        self.sweep_controller.stop()

    def clear_callback(self, *args, **kwargs):
        """
        Deletes all tth from the GUI
        """
        while self.model.rowCount() > 0:
            self.model.removeRow(self.model.rowCount()-1)
        self.plot_widget.update()

    def delete_callback(self, *args, **kwargs):
        """
        Deletes the currently selected Tth
        """
        cur_ind = self.widget.get_selected_tth_row()
        if cur_ind >= 0:
            self.model.removeRow(cur_ind)
        self.plot_widget.update()

    def add_callback(self, *args, **kwargs):
        """
        Adds new row 
        """
        
        data = self.model.get_data()
        if len(data):
            rows = len(data)
            last_tth = data[-1][0]
            if rows > 1:
                second_last_tth = data[-2][0]
                inc = abs(last_tth - second_last_tth)
                new_tth = max([last_tth,second_last_tth])+inc
            else:
                new_tth = last_tth +3
            s = data[-1]
            d = {'tth':new_tth, 'stime':s[7], 'pvsize':s[1], 'phsize':s[2], 'svsize':s[3], 'shsize':s[4], 'dvsize':s[5], 'dhsize':s[6]}
        else:
            d = {'tth':3, 'stime':30., 'pvsize':0.3, 'phsize':0.1, 'svsize':0.3, 'shsize':0.1, 'dvsize':3.5, 'dhsize':0.1}

        
        self.model.add_row(d)


    def sequence_changed_callback(self, data):
        self.ask_to_clear()
        
        tth = data['tth']
        
        for t in tth:
            d = {'tth':t, 'stime':30., 'pvsize':0.3, 'phsize':0.1, 'svsize':0.3, 'shsize':0.1, 'dvsize':3.5, 'dhsize':0.1}
            self.model.add_row(d)
        self.plot_widget.update()
        
        
    def setStyle(self, Style):
        if Style==1:
            WStyle = 'plastique'
            file = open(os.path.join('resources', "stylesheet.qss"))
            stylesheet = file.read()
            self.app.setStyleSheet(stylesheet)
            file.close()
            self.app.setStyle(WStyle)
        else:
            WStyle = "windowsvista"
            self.app.setStyleSheet(" ")
            #self.app.setPalette(self.win_palette)
            self.app.setStyle(WStyle)