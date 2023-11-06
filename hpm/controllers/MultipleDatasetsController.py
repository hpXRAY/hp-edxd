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


from cmath import isnan
from functools import partial
from this import d
from tkinter import filedialog
#from platform import java_ver
import pyqtgraph as pg
import copy
import numpy as np
import os
from PyQt5 import QtWidgets, QtCore
from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight
from hpm.widgets.UtilityWidgets import save_file_dialog, open_file_dialog, open_files_dialog, open_folder_dialog
from hpm.widgets.MultipleDatasetsWidget import MultiSpectraWidget
from hpm.widgets.GSDCalibrationWidget import GSDCalibrationWidget
from hpm.models.multipleDatasetModel import MultipleSpectraModel
from PyQt5.QtCore import pyqtSignal, QObject
import natsort
from hpm.controllers.DisplayPrefsController import DisplayPreferences
from hpm.models.mcaModel import MCA
from hpm.controllers.MaskController import MaskController
from hpm.models.MaskModel import MaskModel
from hpm.controllers.AmorphousAnalysisController import AmorphousAnalysisController
from hpm.controllers.GSDCalibrationController import GSDCalibrationController

class MultipleDatasetsController(QObject):
    file_changed_signal = pyqtSignal(str)  
    element_changed_signal = pyqtSignal(int)
    channel_changed_signal = pyqtSignal(float)  
    add_rois_signal = pyqtSignal(list)

    def __init__(self, file_save_controller, directories):
        super().__init__()
        self.file_save_controller = file_save_controller
        self.directories = directories
        
        
        self.multi_spectra_model = MultipleSpectraModel()
        self.widget = MultiSpectraWidget()

        self.gsd_calibration_controller =  GSDCalibrationController()
        self.gsd_calibration_widget = self.gsd_calibration_controller.widget

        #self.displayPrefs = DisplayPreferences(self.widget.line_plot_widget) 

        self.aac = AmorphousAnalysisController( file_save_controller,directories)

        self.folder = ''
        self.active = False
        self.selectedENV = 0
        self.selectedENV_persist = 0
        self.envLen = 0
        self.single_file = False

        self.scale = 'Channel'
        self.row_scale = 'Index'
        self.file = ''
        self.row = 0
   
        self.create_signals()
        
    def create_signals(self):
       
        self.widget.widget_closed.connect(self.view_closed)

        self.widget.radioE.clicked.connect(partial (self.HorzScaleRadioToggle, 'E'))
        self.widget.radioq.clicked.connect(partial (self.HorzScaleRadioToggle, 'q'))
        self.widget.radioChannel.clicked.connect(partial (self.HorzScaleRadioToggle, 'Channel'))
        self.widget.radioAligned.clicked.connect(partial (self.HorzScaleRadioToggle, 'Aligned'))

        self.widget.align_btn.clicked.connect(self.align_btn_callback)

        self.widget.show_roi_btn.clicked.connect(self.show_roi_btn_callback)
        self.widget.add_roi_btn.clicked.connect(self.add_roi_btn_callback)
        self.widget.delete_roi_btn.clicked.connect(self.delete_roi_btn_callback)
        self.widget.clear_roi_btn.clicked.connect(self.clear_roi_btn_callback)

        self.widget.linearRegionMovedSignal.connect(self.linearRegionMovedSignal_callback)

        self.widget.amorphous_btn.clicked.connect(self.amorphous_btn_callback)
        '''self.widget.sum_scratch_btn.clicked.connect(self.sum_scratch_callback)
        self.widget.ebg_btn.clicked.connect(self.ebg_data)'''
        self.widget.transpose_btn.clicked.connect(self.transpose_E_2theta)
        self.widget.copy_rois_btn.clicked.connect(self.propagate_rois_to_all_elements)
        self.widget.cal_btn.clicked.connect(self.calibrate_all_elements)

        self.widget.key_signal.connect(self.key_sig_callback)
        self.widget.plotMouseMoveSignal.connect(self.fastCursorMove)
        self.widget.plotMouseCursorSignal.connect(self.CursorClick)
        self.widget.file_list_view.currentRowChanged.connect(self.file_list_selection_changed_callback)
        

        self.widget.prev_btn.clicked.connect(partial(self.key_sig_callback, 'left'))
        self.widget.next_btn.clicked.connect(partial(self.key_sig_callback, 'right'))

        self.widget.cal_gsd_2theta_btn.clicked.connect(self.cal_gsd_2theta_btn_callback)
        self.widget.cal_gsd_2theta_upd_btn.clicked.connect(self.cal_gsd_2theta_upd_btn_callback)
      

        self.gsd_calibration_controller.scale_correction_signal.connect(self.cal_gsd_E_upd_btn_callback)
        

    def set_mca(self, mca, element=0):
        self.multi_spectra_model.set_mca(mca)
        self.setHorzScaleBtnsEnabled()
        self.multispectra_loaded()

    def cal_gsd_2theta_btn_callback(self):
       
        data = self.multi_spectra_model.E
        E_scale = self.multi_spectra_model.E_scale
        self.gsd_calibration_controller.set_2D_data(E_scale, data)
        self.gsd_calibration_controller.widget.raise_widget()

    def cal_gsd_2theta_upd_btn_callback(self):
        tth_calibration = self.gsd_calibration_controller.model.tth_calibrated
        calibration = self.multi_spectra_model.mca.get_calibration()
        for i, cal in enumerate(calibration):
            cal.two_theta = tth_calibration[i]
            cal.set_dx_type('edx')
        self.update_view_after_calibration_change()

    def cal_gsd_E_upd_btn_callback(self, e_corrected):
       
        self.multi_spectra_model.correct_calibration_all_elements(e_corrected)
        self.set_mca(self.multi_spectra_model.mca)

    def set_channel_cursor(self, cursor):
        if len(cursor):
           
            channel = cursor['channel']
            pos = channel
            if self.scale == 'q' or self.scale == 'E':
                ndet = self.multi_spectra_model.mca.n_detectors
                if not ndet > self.row:
                    self.row = 0
                converter = self.multi_spectra_model.mca.get_calibration()[self.row].channel_to_scale
                pos = converter(channel,self.scale)
            elif self.scale == 'Aligned':
                if len(self.multi_spectra_model.calibration_inv):
                    pos = self.multi_spectra_model.channel_to_aligned(channel, self.row)
              
            self.widget.select_value(pos)

    

    def file_list_selection_changed_callback(self, row):
        self.file_changed(row)
        pos = self.widget.cursorPoints[0][1]
        if  np.isnan(pos):
            pos = 0
        
        self.CursorClick([row, pos])
            

    def file_changed(self, index):
        self._file_name_update(self.widget.file_name, index )

    def file_changed_fast(self, index):
        self._file_name_update(self.widget.file_name_fast, index )
    
    def _file_name_update(self, widget:QtWidgets.QLabel, index):
        index = int(round(index))
        if hasattr(self.multi_spectra_model.mca, 'files_loaded'):
            files = self.multi_spectra_model.mca.files_loaded
            file_display = ''
            if index < len(files) and len(files)>1:
                file = files[index]
                file_display = os.path.split(file)[-1]
                
            elif len(files)==1:
                file = files[0]
                file_display = os.path.split(file)[-1] 
                n_det = self.multi_spectra_model.mca.n_detectors
                if n_det > 1:
                    file_display += ' : ' + str(index)
        else:
            file_display = ''
        widget.setText(file_display)

    def element_changed(self, element):
        self.element_changed_signal.emit(int(element))

    def fastCursorMove(self, index):
        index = int(index)
        self.file_changed_fast(index)

    def key_sig_callback(self, sig):
        if self.widget.file_view_tabs.currentIndex() == 0:
            row = self.row
            if sig == 'right' or sig == 'up':
                row += 1
            if sig == 'left' or sig == 'down':
                row -= 1
            pos = self.widget.cursorPoints[0][1]
            if not np.isnan(pos):
                self.CursorClick([row, pos])

    def CursorClick(self, index):
        index, pos = int(index[0]), index[1]
        
        rows = np.shape(self.multi_spectra_model.data)[0]
        if index < rows and index >= 0:
            self.row = index
            self.widget.select_spectrum(index)
            self. element_changed(index)
            self.file_changed(index) # in case the mca in multifile
            
            self.widget.select_value(pos)
            self.gsd_calibration_controller.widget.select_value(pos)
            self.gsd_calibration_controller.widget.plot_flat.set_cursor(pos)
            self.set_channel(index, pos)
            files_loaded = self.multi_spectra_model.mca.files_loaded
            if index < len(files_loaded):
                self.widget.select_file(index)
        
    def set_element(self, element ):

        self.widget.select_spectrum(element)
    
        self.file_changed(element) # in case the mca in multifile

    def set_channel(self, index, pos):
        channel = pos
        if self.scale == 'q' or self.scale == 'E':
            converter = self.multi_spectra_model.mca.get_calibration()[index].scale_to_channel
            channel = converter(pos,self.scale)
        elif self.scale == 'Aligned':
            if len(self.multi_spectra_model.calibration):
                channel = self.multi_spectra_model.aligned_to_channel(pos, self.row)
        self.channel_changed_signal.emit(channel)


    def align_btn_callback(self):
        if len(self.multi_spectra_model.data):
            self.multi_spectra_model.rebin_channels(1)
            self.setHorzScaleBtnsEnabled()
            scale = "Aligned"
            self.update_view(scale)

    def show_roi_btn_callback(self):
        show = self.widget.show_roi_btn.isChecked()
        self.widget.set_alignment_roi_visibility(show)
        

    def add_roi_btn_callback(self):
        pos = self.widget.cursorPoints[0][1]
        roi = self.multi_spectra_model.add_new_alignment_roi(pos)
        self.widget.add_alignment_roi(roi, True)
        self.widget.show_roi_btn.setChecked(True)
        self.widget.set_alignment_roi_visibility(True)

    def linearRegionMovedSignal_callback(self, rois):
        self.multi_spectra_model.set_alignment_rois(rois) 

    def delete_roi_btn_callback(self):
        self.multi_spectra_model.delete_alignment_roi()
        self.widget.set_linear_regions(self.multi_spectra_model.alignment_rois, True)

    def clear_roi_btn_callback(self):
        self.multi_spectra_model.clear_alignment_rois()
        self.widget.set_linear_regions(self.multi_spectra_model.alignment_rois, False)
        self.widget.show_roi_btn.setChecked(False)


    def HorzScaleRadioToggle(self,horzScale):
        
        self.set_unit(horzScale)

    def setHorzScaleBtnsEnabled(self):
        
        scales = self.get_available_scales()
        horzScale = self.widget.get_selected_unit()
        if not horzScale in scales:
            self.widget.set_unit_btn('Channel')
            self.unit = 'Channel'
        self.widget.set_scales_enabled_states(scales)

    def get_available_scales(self):
        all_available_scales = []
        available_scales = []
        rows = np.shape(self.multi_spectra_model.data)[0]
        for row in range(rows):
            all_available_scales.append(self.multi_spectra_model.mca.get_calibration()[row].available_scales)
        
        for scale in all_available_scales:
            for item in scale:
                if not item in available_scales:
                    available_scales.append(item)

        aligned = len(self.multi_spectra_model.rebinned_channel_data) >0
        if aligned:
            available_scales.append('Aligned')
        
        scales = available_scales
        return scales
        

    def set_unit(self,unit):
        self.scale = unit
        if self.multi_spectra_model.mca != None:
            self.rebin_by_unit(unit)

    def rebin_by_unit(self, scale):
        if len(self.multi_spectra_model.data):
            if scale == 'q' or scale == 'E':
                self.multi_spectra_model.rebin_scale(scale) 
                #self.multi_spectra_model.rebin_scratch('Channel', scale)
            self.update_view(scale)    
            #self.mask_controller.mask_model.scale = scale
            #self.mask_controller.plot_mask()

    def set_row_scale(self, label='Index'):
        d = [1,0]
        row_scale = 'Spectrum index'

        if label == 'tth':
            row_scale=f'2\N{GREEK SMALL LETTER THETA}'
            d = self.multi_spectra_model.tth_scale

        self.widget.set_image_row_scale(row_scale, d)
        self.row_scale = row_scale

    def transpose_E_2theta(self):
        if self.multi_spectra_model.is2thetaScan():
            self.multi_spectra_model.energy_to_2theta()

    def update_view (self, scale='Channel'):
        r = [1,0]
        scratch_view = []
        if scale == 'Channel':
            view = self.multi_spectra_model.data
            
            self.widget.radioChannel.setChecked(True)
        elif scale == 'q':
            view = self.multi_spectra_model.q
            r = self.multi_spectra_model.q_scale
            #scratch_view = self.multi_spectra_model.scratch_q
            self.widget.radioq.setChecked(True)
        elif scale == 'E':
            view = self.multi_spectra_model.E
            r = self.multi_spectra_model.E_scale
            #scratch_view = self.multi_spectra_model.scratch_E
            self.widget.radioE.setChecked(True)
        elif scale == 'Aligned':
            view = self.multi_spectra_model.rebinned_channel_data
            self.widget.radioAligned.setChecked(True)
            #r = self.multi_spectra_model.E_scale

        self.widget.set_spectral_data(view)
        self.widget.set_linear_regions(self.multi_spectra_model.alignment_rois, self.widget.show_roi_btn.isChecked())
        #if len(scratch_view):
        #    self.widget.scratch_widget.plot_image(scratch_view)
        #self.mask_controller.mask_model._img_data = view
        #self.mask_controller.update_mask_dimension()
        self.widget.set_image_scale(scale, r)
        
        self.scale = scale
        

    def amorphous_btn_callback(self):
        scales =  self.get_available_scales()
        if 'E' in scales:
            
            self.aac.set_models(self.multi_spectra_model)
            self.aac.show_view()



    def sum_scratch_callback(self):

        if self.scale == 'E':
            data = self.multi_spectra_model.scratch_E
            scale = self.multi_spectra_model.E_scale
        elif self.scale == 'q':
            data = self.multi_spectra_model.scratch_q
            scale = self.multi_spectra_model.q_scale
        elif self.scale == 'Channel':
            data = self.multi_spectra_model.data
            scale = [1,0]
        elif self.scale == 'Aligned':
            data = self.multi_spectra_model.rebinned_channel_data
            scale = [1,0]
        mask = self.mask_model.get_mask()
        out = self.multi_spectra_model.flaten_data(data, mask)
        self.multi_spectra_model.scratch_q_average = out
        x = np.arange(len(out)) * scale[0] + scale[1]
        self.widget.plot_data(x, out)
    

    def ebg_data(self):
        if self.scale == 'E':
            if len(self.multi_spectra_model.scratch_E_average):
                m = self.multi_spectra_model
                for i in range(np.shape(m.E)[0]):
                    m.scratch_E[i] = m.E[i] / m.scratch_E_average
                #self. update_view('E')
                self.widget.scratch_widget.plot_image(m.scratch_E)
        if self.scale == 'q':
            if len(self.multi_spectra_model.scratch_q_average):
                m = self.multi_spectra_model
                for i in range(np.shape(m.q)[0]):
                    m.scratch_q[i] = m.q[i] / m.scratch_q_average
                #self. update_view('q')
    
                self.widget.scratch_widget.plot_image(m.scratch_q)

    def multispectra_loaded(self, scale='Channel'):
        data = self.multi_spectra_model.data
       
        #self.aac.model.set_data(data)
        self.multi_spectra_model.q = np.zeros(np.shape(data))
        self.multi_spectra_model.E = np.zeros(np.shape(data))
        self.multi_spectra_model.rebinned_channel_data = np.zeros(np.shape(data))
        scales = self.get_available_scales()
        horzScale = self.widget.get_selected_unit()
        if horzScale in scales:
            self.rebin_by_unit(horzScale)
            scale = horzScale
        self.update_view(scale)
        files_loaded = self.multi_spectra_model.mca.files_loaded
        files = []
        for f in files_loaded:
            files.append(os.path.basename(f))
        self.widget.reload_files(files)
        self.file_changed(self.row)
        fast_row = self.widget.cursorPoints[1][0]
        self.file_changed_fast(fast_row)
        

    def propagate_rois_to_all_elements(self):
        row = self.row
        rois = self.multi_spectra_model.mca.get_rois_by_det_index(row)
        all_new_rois = self.multi_spectra_model.make_aligned_rois(row, rois)
        self.add_rois_signal.emit(all_new_rois)

    def calibrate_all_elements(self):
        self.multi_spectra_model.calibrate_all_elements(2)
        self.update_view_after_calibration_change()

    def update_view_after_calibration_change(self):
        self.setHorzScaleBtnsEnabled()
        self.multispectra_loaded()
        

    def show_view(self):
        self.active = True
        #self.update_envs()
        self.widget.raise_widget()
        
        
    def view_closed(self):
        self.active = False
       

