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


import os.path, sys
from PyQt6 import uic, QtWidgets,QtCore, QtGui
from  PyQt6.QtWidgets import QMainWindow, QApplication, QInputDialog, QWidget, QLabel
from PyQt6.QtCore import QObject, pyqtSignal, Qt
import numpy as np
from functools import partial
import json
import copy
from math import isnan
from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight
from pathlib import Path
from utilities.HelperModule import calculate_color
from numpy import arange
from utilities.HelperModule import getInterpolatedCounts
from axd.models.aEDXD_spectra_model import ROI
from axd.widgets.aEDXD_roi_widget import aEDXDRoiWidget, plotFitWindow
from utilities.hpMCAutilities import readconfig


############################################################

class aEDXDPeakCutController(QObject):
    cut_peaks_changed_signal = pyqtSignal()
    roi_updated_signal = pyqtSignal(dict)  

    def __init__(self, model, spectra_model, display_window, files):
        super().__init__()
        self.model = model
        self.spectra_model = spectra_model
        self.display_window = display_window
        self.files = files
        self.roi_window = aEDXDRoiWidget()
        self.plot_fit_window = plotFitWindow()
        self.create_connections()
        self.current_tth = 0
        self.selectedROI = None
        self.plotFitOpen = False

    def create_connections(self):
        self.display_window.spectrum_widget.cut_peak_btn.clicked.connect(lambda:self.cut_peak_clicked(self.files.current_tth,
                                                          self.display_window.spectrum_widget.cut_peak_btn.isChecked() ))
        self.roi_window.delete_btn.clicked.connect(self.remove_btn_click_callback)
        self.roi_window.clear_btn.clicked.connect(self.clear_btn_click_callback)
        self.roi_window.show_cb_state_changed.connect(self.roi_show_cb_callback)
        self.roi_window.roi_selection_changed_signal.connect(self.roi_selection_changed)
        self.roi_window.edit_btn.clicked.connect(self.show_fit)
        self.roi_window.filter_btn.clicked.connect(self.filter_btn_callback)
        self.plot_fit_window.widget_closed.connect(self.plot_fit_closed)
        self.plot_fit_window.vLineRight.sigPositionChangeFinished.connect(self.edit_cursor_drag_done)
        self.plot_fit_window.vLineLeft.sigPositionChangeFinished.connect(self.edit_cursor_drag_done)

        self.files.tth_selection_changed_signal.connect(self.tth_index_changed_callback)
        
        self.roi_window.cut_peak_Wn.valueChanged.connect(self.wn_changed_callback)
        self.roi_window.cut_peak_iter.valueChanged.connect(self.iter_changed_callback)
        self.roi_window.cut_peak_method_baseline_choice.clicked.connect(partial(self.method_changed_callback,'baseline'))
        self.roi_window.cut_peak_method_spline_choice.clicked.connect(partial(self.method_changed_callback,'spline'))

    def wn_changed_callback(self, wn):
        sets = self.get_rois_to_edit()
        self.roi_window.cut_peak_method_baseline_choice.blockSignals(True)
        self.roi_window.cut_peak_method_spline_choice.blockSignals(True)
        self.roi_window.cut_peak_method_baseline_choice.setChecked(True)
        self.roi_window.cut_peak_method_baseline_choice.blockSignals(False)
        self.roi_window.cut_peak_method_spline_choice.blockSignals(False)
        if len(sets):
            wn = round(wn,12)
            for s in sets:
                spectrum = s[0]
                roi = s[1]
                roi.wn = wn
                
                if roi.method != 'baseline':
                    roi.method = 'baseline'
                    iterations= self.roi_window.cut_peak_iter.value()
                    iterations = int(iterations)
                    roi.iterations =iterations
                spectrum.compute_spectrum()
            self.cut_peaks_changed_signal.emit() 
        #print('wn_changed_callback: ' + str(sets))

    def method_changed_callback(self, method):
        sets = self.get_rois_to_edit()
        if len(sets):
            for s in sets:
                spectrum = s[0]
                roi = s[1]
                roi.method = method
                spectrum.compute_spectrum()
            self.cut_peaks_changed_signal.emit() 
        #print('method_changed_callback: ' + str(sets))
        

    def iter_changed_callback(self, iterations):
        iterations = int(iterations)
        sets = self.get_rois_to_edit()
        self.roi_window.cut_peak_method_baseline_choice.blockSignals(True)
        self.roi_window.cut_peak_method_spline_choice.blockSignals(True)
        self.roi_window.cut_peak_method_baseline_choice.setChecked(True)
        self.roi_window.cut_peak_method_baseline_choice.blockSignals(False)
        self.roi_window.cut_peak_method_spline_choice.blockSignals(False)
        if len(sets):
            iterations = round(iterations,12)
            for s in sets:
                spectrum = s[0]
                roi = s[1]
                roi.iterations = iterations
                if roi.method != 'baseline':
                    roi.method = 'baseline'
                    roi.wn = self.roi_window.cut_peak_Wn.value()
                spectrum.compute_spectrum()
            self.cut_peaks_changed_signal.emit() 

        #print('iter_changed_callback: ' + str(sets))

    def get_rois_to_edit(self):
        rois = []
        All = self.roi_window.baseline_params_apply_all.isChecked()
        if not All:
            [spectrum, roi] = self.get_selected_roi()
            if not spectrum is  None and not rois is None:
                rois = [[spectrum, roi]]
        else:
            rois = self.get_all_displayed_rois()

        #print('get_rois_to_edit: ' + str(rois))
        return rois
        
    def get_selected_roi(self):
        params = self.roi_window.get_selected_roi_row()
        tth = params['tth']
        name = params['name']
        roi = None
        spectrum = None
        if tth is not None and name is not None:
            spectrum = self.spectra_model.tth_groups[tth].spectrum
            if len(spectrum.x) :
                roi_ind = spectrum.find_roi_by_name(name)
                roi = spectrum.rois[roi_ind]

        #print('get_selected_roi: ' + str([spectrum, roi]))
        return [spectrum, roi]

    def get_all_displayed_rois(self):
        rois = []
        for i in range(len(self.roi_window.name_items)):

            name = self.roi_window.name_items[i].text()
            tth = float(self.roi_window.tth_items[i].text())
            
            if tth is not None and name is not None:
                spectrum = self.spectra_model.tth_groups[tth].spectrum
                if len(spectrum.x) :
                    roi_ind = spectrum.find_roi_by_name(name)
                    roi = spectrum.rois[roi_ind]
                rois.append([spectrum, roi])

        #print('get_all_displayed_rois: ' + str(rois))
        return rois


    def roi_selection_changed(self, params):
        
        tth = float(params['tth'])
        name = params['name']
        
        roi = None
        if tth is not None and name is not None:
            spectrum = self.spectra_model.tth_groups[tth].spectrum
            if len(spectrum.x) :
                roi_ind = spectrum.find_roi_by_name(name)
                roi = spectrum.rois[roi_ind]
        
        
        if roi is not None:
            wn = roi.wn
            iterations = roi.iterations
            method = roi.method
            self.roi_window.cut_peak_Wn.blockSignals(True)
            self.roi_window.cut_peak_iter.blockSignals(True)
            self.roi_window.cut_peak_Wn.setValue(wn)
            self.roi_window.cut_peak_iter.setValue(iterations)
            self.roi_window.cut_peak_Wn.blockSignals(False)
            self.roi_window.cut_peak_iter.blockSignals(False)
            self.roi_window.cut_peak_method_baseline_choice.blockSignals(True)
            self.roi_window.cut_peak_method_spline_choice.blockSignals(True)
            if method == 'spline':
                self.roi_window.cut_peak_method_spline_choice.setChecked(True)
            if method == 'baseline':
                self.roi_window.cut_peak_method_baseline_choice.setChecked(True)
            self.roi_window.cut_peak_method_baseline_choice.blockSignals(False)
            self.roi_window.cut_peak_method_spline_choice.blockSignals(False)

            if self.plotFitOpen:
                self.updateFitPlot(params)
            self.roi_updated_signal.emit(params)
        
        #print('selection changed: ' + str(params) +' roi:'+ str(roi))


    def edit_cursor_drag_done(self, line):
        pos1 = round(self.plot_fit_window.vLineRight.getPos()[0],3)
        pos2 = round(self.plot_fit_window.vLineLeft.getPos()[0],3)
        left = min([pos1,pos2])
        right = max([pos1,pos2])
        #print('left='+str(left)+', right='+str(right))
        params = self.roi_window.get_selected_roi_row()
        tth = params['tth']
        name = params['name']
        row = params['row']
        if tth !=None:
            spectrum = self.spectra_model.tth_groups[tth].spectrum
            roi_ind = spectrum.find_roi_by_name(name)
            if roi_ind > -1:
                spectrum.set_new_roi_bounds(roi_ind,left,right)
                name = spectrum.rois[roi_ind].name
                params = {'tth': tth, 'name': name}
                self.roi_window.rename_roi(row,name)
                self.updateFitPlot(params,autoscale=False)
                self.cut_peaks_changed_signal.emit() 

    

        #self.update_roi_cursor_lines() 
        #print('selected: ' + str(cur_ind))

    def updateFitPlot(self,params, autoscale = True):
        if self.plotFitOpen:
            tth = params['tth']
            name = params['name']
            if tth is not None and name is not None:
                x_fit=[]
                y_fit=[]
                label=""
                x=[]
                y=[]
                ranges = None
                roi_range= None
                
                spectrum = self.spectra_model.tth_groups[tth].spectrum
                if len(spectrum.x) :
                    roi_ind = spectrum.find_roi_by_name(name)
                    roi = spectrum.rois[roi_ind]
                    data = roi.peak_cutting['tc_binned']
                    x = data[0]
                    y = data[1]
                    left = min(x)
                    right = max(x)
                    r_x =  right-left 
                    data1 = roi.peak_cutting['tc']
                    x_fit = spectrum.x
                    y_fit = spectrum.y
                    x1 = roi.peak_cutting['interp'][0]
                    y1 = roi.peak_cutting['interp'][1]
                    roi_range = [roi.right, roi.left]
                    if autoscale:
                        y_fit_min = min(y)
                        y_fit_max = max(y)
                        r_y = y_fit_max - y_fit_min
                        Ymax = y_fit_max+ r_y
                        Ymin = y_fit_min - r_y
                        Emin = max([left - r_x, 0])
                        Emax= min([right + r_x, len(spectrum.x)-1])
                        ranges = [[Emin,Emax],[Ymin,Ymax]]
                self.plot_fit_window.set_data(x_fit,y_fit,label,x,y,x1,y1,'E','KeV', ranges, roi_range)

    def show_fit(self):
        cur_ind = self.roi_window.get_selected_roi_row()
        if not self.plotFitOpen:
            self.plot_fit_window.raise_widget()
            self.plotFitOpen = True
        self.updateFitPlot(cur_ind)    
    
    def plot_fit_closed(self):
        self.plotFitOpen = False

    def roi_show_cb_callback(self, state):
        tth = float(state['tth'])
        name = state['name']
        checked = state['checked']
        spectrum = self.spectra_model.tth_groups[tth].spectrum
        ind = spectrum.find_roi_by_name(name)
        spectrum.change_roi_use(ind,checked)
        self.cut_peaks_changed_signal.emit() 

    def filter_btn_callback(self, state):
        self.display_rois()
        

    def tth_index_changed_callback(self, index):
        self.display_rois()

    def display_rois(self):
        fltr_state = self.roi_window.filter_btn.isChecked()
        tth = str(self.files.current_tth)
        while self.roi_window.roi_tw.rowCount() > 0:
            self.roi_window.del_roi(self.roi_window.roi_tw.rowCount()-1)
        peaks = self.spectra_model.get_cut_peaks()
        for p in peaks:
            left = '%.3f' % (p['left'])
            right = '%.3f' % (p['right'])
            name = left +'-'+right
            tth_str=str(p['tth'])
            if tth == tth_str or not fltr_state:
                self.roi_window.add_roi(True,name,tth_str)
        self.cut_peaks_changed_signal.emit() 
        
        
    def load_peaks_from_config(self):
        conf = self.model.config_file
        self.spectra_model.set_autoprocess(False)
        if conf != None:
            config_dict = readconfig(self.model.config_file)
            if 'E_cut' in config_dict:
                peaks  = config_dict['E_cut']
                self.spectra_model.load_peaks_from_config(peaks)
        self.spectra_model.set_autoprocess(True)
        self.display_rois()
       

    def remove_btn_click_callback(self):
        """
        Deletes the currently selected roi
        """
        cur_tth = self.roi_window.get_selected_roi_tth()
        cur_name = self.roi_window.get_selected_roi_name()
        if cur_name !='':
            ind = list(self.spectra_model.tth).index(cur_tth)
            group = self.spectra_model.tth_groups[float(cur_tth)]
            spectrum = group.spectrum
            roi_ind = spectrum.find_roi_by_name(cur_name)
            spectrum.del_roi(roi_ind)
        self.display_rois()
        

    def clear_btn_click_callback(self):
        """
        Deletes all rois
        """

        displayed_rois = self. get_all_displayed_rois()
        
        for dr in displayed_rois:
            spectrum, roi = dr

            spectrum.set_auto_process(False)
            ind = spectrum.rois.index(roi)

            spectrum.del_roi(ind)
            spectrum.set_auto_process(True)
        self.display_rois()

    def cut_peak_clicked(self, tth, state):
        if state:
            action = 'Add'
        else:
            action = 'Set'
        if self.roi_window.cut_peak_method_baseline_choice.isChecked():
            method = 'baseline'
        else:
            method = 'spline'
        self.roi_action(action, tth, method)

    def roi_action(self, mode, tth, method = 'spline'):
        if mode == 'Add':
            #self.display_window.spectrum_widget.cut_peak_btn.setText("Set")
            width = 8
            self.roi_construct(mode, width=width)
        if mode == 'Set':
            #self.display_window.spectrum_widget.cut_peak_btn.setText("Add")
            reg = self.roi_construct(mode)
            if not reg[1] == 0:
                if len(self.spectra_model.tth_groups):
                    group = self.spectra_model.tth_groups[tth]
                    roi = {'left':reg[0],'right':reg[1]}
                    wn = self.roi_window.cut_peak_Wn.value()
                    iterations = int(self.roi_window.cut_peak_iter.value())
                    roi['method']=method
                    roi['wn']=wn
                    roi['iterations'] = iterations
                    group.add_cut_roi(roi)
                    self.display_rois()

    def roi_construct(self, mode, **kwargs):
        if mode == 'Add':
            width = kwargs.get('width', None)
            self.display_window.spectrum_widget.fig.win.lin_reg_mode(mode,width=width)
        elif mode == 'Set':
            reg = self.display_window.spectrum_widget.fig.win.lin_reg_mode(mode)
            
            low = reg[0]
            high = reg[1]
            if isnan(low):
                low = 0
            if isnan(high):
                high = 0
            center = int((low+high)/2)
            half_width = int((high - low)/2)
            return [low,high]

    def show_rois(self):
        self.roi_window.raise_widget()   

    