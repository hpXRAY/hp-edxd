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



from functools import partial
import collections
import copy
from typing import Iterator
import utilities.centroid
import numpy as np
import os
from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight
from hpm.widgets.UtilityWidgets import save_file_dialog, open_file_dialog, open_files_dialog
from hpm.widgets.RoiWidget import RoiWidget, plotFitWindow

from hpm.models.roiSetsModel import RoiModel

from hpm.models.mcaModel import  McaROI
from PyQt6.QtCore import pyqtSignal, QObject

class RoiController(QObject):

    roi_updated_signal = pyqtSignal(int, str )  
    roi_selection_changed_signal = pyqtSignal(int,str)
    #rois_changed = pyqtSignal()  

    def __init__(self, mcaModel, plotWidget, plotController,mainController):
        super().__init__()
        
        self.set_mca(mcaModel)
        self.mcaController = mainController
        self.roi = []
        self.roi_model = RoiModel()
        self.roi_model_detector = RoiModel()
        self.roi_model_file = RoiModel()
        self.rois_widget = RoiWidget()
        self.plot_fit_window = plotFitWindow()
        
        self.active = False
        self.plotFitOpen = False
        self.win = None
        self.fitPlot = None
        self.dataPlot = None
        self.fitPlots = None
        self.selectedROI = 0
        self.selectedROI_persist = 0
        self.selectedROIset = 0
        self.selectedROIset_persist = 0
        self.roiLen = 0
        self.roiSetsLen = 0
        self.dataLen = self.mca.nchans
        self.pattern_widget = plotWidget
        self.plotController = plotController
        self.unit = 'E'
        self.unit_ = 'KeV'
        self.roi_cursor = []
        color = self.add_roi_cursor_plot()
        self.rois_widget.roi_tw.set_tw_header_unit(self.unit,self.unit_)
       
        #self.phases =dict()
        self.create_signals()
        
    def set_mca(self, mca, element=0):
        self.mca = mca
        calibrations = self.mca.get_calibration()
        if not element < len(calibrations):
            element = 0
        self.calibration = calibrations[element]

    

    def get_calibration(self):
        element = self.mcaController.element
        calibrations = self.mca.get_calibration()
        if not element < len(calibrations):
            element = 0
        self.calibration = calibrations[element]
        return self.calibration
        
    def create_signals(self):

        ui = self.mcaController.widget
        ui.btnROIadd.clicked.connect(lambda:self.roi_action('add'))
        ui.btnROIclear.clicked.connect(lambda:self.roi_action('clear'))
        ui.btnROIdelete.clicked.connect(lambda:self.roi_action('delete'))
        ui.btnROIprev.clicked.connect(lambda:self.roi_action('prev'))
        ui.btnROInext.clicked.connect(lambda:self.roi_action('next'))

        self.plotController.logScaleYUpdated.connect(self.update_log_y_scale)
        self.connect_click_function(self.rois_widget.delete_btn, self.remove_btn_click_callback)
        self.connect_click_function(self.rois_widget.clear_btn, self.clear_rois)
        self.connect_click_function(self.rois_widget.show_fit_btn, self.show_fit)
        self.connect_click_function(self.rois_widget.save_peaks_btn, self.save_peaks)
        

        self.rois_widget.widget_closed.connect(self.view_closed)
        self.rois_widget.roi_tw.currentCellChanged.connect(self.roi_selection_changed)
        self.rois_widget.roi_tw.show_cb_state_changed.connect(self.change_roi_use)
        self.rois_widget.roi_sets_tw.show_cb_state_changed.connect(self.change_roi_set_use)
        self.rois_widget.roi_tw.name_item_changed.connect(self.edit_roi_name)

        self.plot_fit_window.widget_closed.connect(self.plot_fit_closed)
        self.rois_widget.key_signal.connect(self.key_sig_callback)

    def connect_click_function(self, emitter, function):
        emitter.clicked.connect(function)      

    def key_sig_callback(self, sig):
        if sig == 'delete' :
            self.remove_btn_click_callback()

    def show_view(self):
        self.active = True
        
        self.rois_widget.raise_widget()
       
    def view_closed(self):
        self.active = False
        if self.plotFitOpen:
            self.plotFitOpen = False
            self.plot_fit_window.close()

    def update_log_y_scale (self, log_mode):
        self.update_roi_cursor_line()

    def update_unit (self):
        self.unit_ = self.plotController.units[self.unit]
        unit = self.unit
        if unit == '2 theta':
            unit = u'2θ'
        self.rois_widget.roi_tw. set_tw_header_unit(unit,self.unit_)
        
        if self.fitPlots is not None:
            if self.plotFitOpen:
                cur_ind = self.rois_widget.roi_tw. get_selected_roi_row()
                if cur_ind >= 0 :
                    self.updateFitPlot(cur_ind)


    def update_set_rois (self, use_only=False):

        roi_sets = self.roi_model.get_sets()
        oldLen =  copy.copy(self.roiSetsLen)
        self.selectedROIset_persist = copy.copy(self.selectedROIset)
       
        newLen = len(roi_sets)
        self.nroisets = len(roi_sets)
        self.blockSignals(True)
        self.rois_widget.roi_sets_tw.blockSignals(True)
        if newLen == oldLen:
            names = list(roi_sets.keys())
            for name in roi_sets:
                index = names.index(name)
                
                self.update_roi_set_by_ind(index, name, roi_sets[name])
        else:
            while self.rois_widget.roi_sets_tw.rowCount() > 0:
                self.rois_widget.roi_sets_tw.del_roi(self.rois_widget.roi_sets_tw.rowCount()-1,silent=True)
            for name in roi_sets:
        
                self.rois_widget.roi_sets_tw.add_roi(name, roi_sets[name],  silent=True)
        self.blockSignals(False)              
        
        self.rois_widget.roi_sets_tw.blockSignals(False)      
        if self.selectedROIset_persist<len(roi_sets):
            sel = np.clip(self.selectedROIset_persist,0,31)
        else: 
            sel = len(roi_sets)-1
        self.rois_widget.roi_sets_tw. select_roi(sel) 
        self.selectedROIset=sel
        self.roiSetsLen = copy.copy(self.nroisets)
        
    

    def update_rois(self, element, use_only=False):

        if not use_only:
            self.roi = self.roi_model. get_rois_for_use(element)

        unit =self.plotController.get_unit()
        if self.unit !=unit:
            self.unit=unit
            self.update_unit()

        self.update_widgets(self.roi)
        self.update_set_rois()

        
    def update_widgets(self, roi):

        oldLen =  copy.copy(self.roiLen)
        self.selectedROI_persist = copy.copy(self.selectedROI)
        
        newLen = len(roi)
        self.nrois = len(roi)
        
        self.rois_widget.roi_tw.blockSignals(True)
        if newLen == oldLen:
            for r in roi:
                index = roi.index(r)
                centroid, fwhm, counts = self.get_roi_attributes(r, self.unit)
                self.update_roi_by_ind(index, r.use, r.label, '%.3f'%(centroid), '%.3f'%(fwhm), '%d'%(counts), r.fit_ok)
        else:
            while self.rois_widget.roi_tw.rowCount() > 0:
                self.rois_widget.roi_tw.del_roi(self.rois_widget.roi_tw.rowCount()-1,silent=True)
            for r in roi:
                centroid, fwhm, counts = self.get_roi_attributes(r, self.unit)
                self.rois_widget.roi_tw.add_roi(r.use, r.label, 
                                '%.3f'%(centroid), '%.3f'%(fwhm), roi.index(r),'%d'%(counts), r.fit_ok,  silent=True)
               
       
             

        if self.selectedROI_persist<len(roi):
            sel = int(np.clip(self.selectedROI_persist,0,31))
        else: 
            sel = len(roi)-1
        self.rois_widget.roi_tw. select_roi(sel) 
        self.selectedROI=sel
        self.rois_widget.roi_tw.blockSignals(False) 
        self.emit_rois_updated()
        

        self.roiLen = copy.copy(self.nrois)
        if self.plotFitOpen:
            self.show_fit()

    ####################################################################################
    # roi-related interaction with mca model
    ####################################################################################

    def get_roi_model(self):
        mca_type = self.mcaController.Foreground
        roi_model = None
        if mca_type != None:
            if mca_type =='file':
                roi_model = self.roi_model_file
            elif mca_type == 'detector':
                roi_model = self.roi_model_detector
        return roi_model

    def data_updated(self, element):

        mca_type = self.mcaController.Foreground
        if mca_type != None:
            if mca_type == 'file':
                roi_model = self.roi_model_file
            elif mca_type == 'epics':
                roi_model = self.roi_model_detector
            self.roi_model = roi_model

            rois_all = self.mca.get_rois()
            rois = rois_all[element]

            if mca_type == 'file':
                load_from_file = self.rois_widget.lock_rois_btn.isChecked()
                if not load_from_file:
                    file_rois = self.mca.get_file_rois()[element]
                    self.roi_model.set_file_rois(file_rois, element)

            elif mca_type == 'epics':    
                det_rois = self.mca.get_det_rois()[element]
                d_rois = []
                for dr in det_rois:
                    new = True
                    for r in rois:
                        if r == dr:
                            new = False
                    if new:
                        d_rois.append(dr)
                self.roi_model.add_rois(d_rois, element)
            
            

            rois_for_use = self.roi_model.get_rois_for_use(element)
            for r in rois_for_use:
                self.mca.compute_roi(r, element)

            
            # check if rois already in mca, no update if all rois are already in mca
            for_mca = []
            for r in rois_for_use:
                not_in_mca = True
                for r_d in rois:
                    equal = False
                    if mca_type == 'file':
                        equal = r.compare_counts(r_d)
                    if mca_type == 'epics':
                        equal = r == r_d
                    if equal:
                        not_in_mca = False
                if not_in_mca:
                    for_mca.append(r)
            
            if len(for_mca):
                self.set_mca_rois(rois_for_use,element)
            
            self.update_rois(element)

    

    def add_rois_to_mca(self, rois, detector):
        # rois: list of McaRoi objects
        # adding rois to mca should happen only throgh one place

        
        if type(rois) == list:
            for r in rois:
        
                self.mca.compute_roi(r, detector)
                self.roi_model.add_roi(r,detector)
        else:
            self.mca.compute_roi(rois, detector)
            self.roi_model.add_roi(rois,detector)

        rois_for_use = self.roi_model.get_rois_for_use(detector)
        self.set_mca_rois(rois_for_use, detector)

    def del_roi_from_mca(self, ind, det):
        # deleting rois from mca should happen only throgh one place
        self.roi_model. delete_roi(ind, det)
        rois_for_use = self.roi_model.get_rois_for_use(det)
        self.set_mca_rois(rois_for_use, det)

    def clear_mca_rois(self, element):
        # clearing rois from mca should happen only throgh one place
        self.mca.clear_rois(source='controller', detector=element)
        self.roi_model.clear_rois(element)

    def get_rois_for_use(self, element):
        rois = self.mca.get_rois()[element]
        return rois

    def set_mca_rois(self, rois, det):
        self.mca.auto_process_rois = False
        self.mca.set_rois(rois, detector= det, source = 'controller')
        self.mca.auto_process_rois = True

    def save_peaks(self):
        img_filename, _ = os.path.splitext(os.path.basename(self.mca.name))
        filename = save_file_dialog(
            self.mcaController.widget, "Save ROI's Data.",
            os.path.join(self.mcaController.working_directories.savedata,
                        img_filename + '.csv'),
            ('Comma separated values (*.csv)'))
        if filename != '':
            self.mca.save_peaks_csv(filename)

    

    ####################################################################################
    # end roi-related interaction with mca model
    ####################################################################################


    def roi_action(self, action):
        widget = self.mcaController.widget
        if self.mca != None:
            
            if action == 'add':
                if self.plotController.is_cursor_in_range():
                    #self.add_roi_btn()
                    mode = widget.btnROIadd.text()
                    #self.plotController.roi_construct(mode)
                    widths = {'E': 0.7, 'q': 0.1, 'Channel':25, 'd': 0.1}
                    if mode == 'Add':
                        widget.btnROIadd.setText("Set")
                        if self.unit in widths:
                            width = widths[self.unit]
                        else:
                            width = 2
                        self.plotController.roi_construct(mode,width=width)
                    else:
                        widget.btnROIadd.setText("Add")
                        reg = self.plotController.roi_construct(mode)
                        if reg is not None:
                            self.addROIbyChannel(reg[0],reg[1])
                else:
                    widget.btnROIadd.setChecked(False)
            elif action == 'delete':
                self.remove_btn_click_callback()
            elif action == 'clear':
                self.clear_rois()
            elif action == 'next':
                self.navigate_btn_click_callback('next')
            elif action == 'prev':
                self.navigate_btn_click_callback('prev')
            else: 
                pass
        else:
            if action == 'add':
                widget.btnROIadd.setChecked(False)

    def get_roi_attributes(self, roi, unit):
        if self.unit == 'E': 
            centroid = roi.energy
            fwhm = roi.fwhm_E
        elif self.unit == 'q': 
            centroid = roi.q
            fwhm = roi.fwhm_q
        elif self.unit == 'd': 
            centroid = roi.d_spacing
            fwhm = abs(roi.fwhm_d)
        elif self.unit == '2 theta': 
            centroid = roi.two_theta
            fwhm = abs(roi.fwhm_tth)
        else: 
            centroid = roi.centroid
            fwhm = roi.fwhm
        try:
            counts = max(roi.counts)
        except:
            counts = 0
        return centroid, fwhm, counts

    def emit_rois_updated(self):
        selection = self.rois_widget.roi_tw. get_selected_roi_row()
        if selection >-1 and selection < len(self.roi):
            out = self.roi[selection].label
            
        else: 
            out = ''
        self.selectedROI = selection
        self.update_roi_cursor_line()
        self.roi_updated_signal.emit(selection, out)

    def navigate_btn_click_callback(self, action=''):
        n=len(self.roi)
        if n >0:
            if action == 'next':
                move = 1
            elif action == 'prev':
                move = -1
            else:
                move = 0
            curr_ind = self.selectedROI 
            new_ind = curr_ind + move
            
            if new_ind >=0 and new_ind<n:
                pass
            else:  # enable loop-around
                if new_ind<0:
                    new_ind= n+new_ind
                else:
                    new_ind=new_ind-n
            if new_ind != curr_ind:
                self.rois_widget.roi_tw.select_roi(new_ind)    

    def roi_selection_changed(self, row, **kwargs):
        cur_ind = row
        self.selectedROI = row
        if self.plotFitOpen:
            self.updateFitPlot(cur_ind)
        if cur_ind >-1 and cur_ind < len(self.roi): 
            out = self.roi[cur_ind].label
               
        else: out = ''
        self.update_roi_cursor_line() 
        self.roi_selection_changed_signal.emit(cur_ind, out)
    

    def updateFitPlot(self,ind):
        if self.plotFitOpen:
            rois = self.roi
            if ind >=0 and len(rois)>0 and ind < len(rois):
                x = self.roi[ind].channels
                if self.roi[ind].fit_ok:
                    fit = self.roi[ind].yFit
                    x_fit = self.roi[ind].x_yfit
                else:
                    x_fit=[]
                    fit=[]
                label = self.roi[ind].label
                if self.unit == 'E':
                    x_fit = self.calibration.channel_to_energy(x_fit)
                    x = self.calibration.channel_to_energy(x)
                elif self.unit == 'q':
                    x_fit = self.calibration.channel_to_q(x_fit)
                    x = self.calibration.channel_to_q(x)
                elif self.unit == 'd':
                    x_fit = self.calibration.channel_to_d(x_fit)
                    x = self.calibration.channel_to_d(x)
                elif self.unit == '2 theta':
                    x_fit = self.calibration.channel_to_tth(x_fit)
                    x = self.calibration.channel_to_tth(x)
                if label == '':
                    label = '#'+str(ind) 
                label = 'ROI '+label+' fit'
                counts = self.roi[ind].counts
            else:
                x_fit=[]
                fit=[]
                label=""
                x=[]
                counts=[]
            self.plot_fit_window.set_data(x_fit,fit,label,x,counts,self.unit,self.unit_)
            

    def show_fit(self):
        cur_ind = self.rois_widget.roi_tw. get_selected_roi_row()
        if not self.plotFitOpen:
            self.plot_fit_window.raise_widget()
            self.plotFitOpen = True
        self.updateFitPlot(cur_ind)    
    
    def plot_fit_closed(self):
        self.plotFitOpen = False


    def change_roi_use(self, ind, use):
        rois = self.roi
        element = self.mcaController.element
        if ind >=0 and len(rois)>0 and ind < len(rois):
            self.mca.change_roi_use(ind, use, element)  
            self.update_rois(element, use_only=True)

    def change_roi_set_use(self, ind, use):
        element = self.mcaController.element
        sets = list(self.roi_model.get_sets().keys())
        if ind >=0 and len(sets)>0 and ind < len(sets):
            name = sets[ind]
            self.roi_model.change_roi_set_use(name, use, element)  
            rois_for_use = self.roi_model.get_rois_for_use(element)
            self.set_mca_rois( rois_for_use, element)
            self.update_rois(element, use_only=False)
            
    def remove_btn_click_callback(self, *args, **kwargs):
        """
        Deletes the currently selected roi
        """
        element = self.mcaController.element
        ind = self.rois_widget.roi_tw. get_selected_roi_row()
        if ind >= 0:
            rois = self.roi
            if ind >=0 and len(rois)>0 and ind < len(rois):
                self.blockSignals(True)
                self.del_roi_from_mca(ind,element)
                self.roi = self.roi_model.get_rois_for_use(element)
                self.roi_removed(ind)
                
                self.blockSignals(False)
                self.update_rois(element)
                

    def roi_removed(self, ind):
        self.rois_widget.roi_tw. del_roi(ind)
        

    def clear_rois(self, *args, **kwargs):
        """
        Deletes all rois from the GUI
        """
        element = self.mcaController.element
        self.blockSignals(True)
        while self.rois_widget.roi_tw.rowCount() > 0:
            self.roi_removed(self.rois_widget.roi_tw.rowCount()-1)
        self.clear_mca_rois(element)
        self.roiLen = 0
        self.blockSignals(False)
        self.update_rois(element)


    def addROISbyE(self, e_rois):
        E2C = self.get_calibration().energy_to_channel
        rois = []
        for roi in e_rois:
            r={}
            ch=E2C(roi[0])
            hw=roi[1]
            lbl=roi[2]
            r['channel']=ch
            r['halfwidth']=hw
            r['label']=lbl
            rois.append(r)
        self.addReflections(rois)
        
    def addJCPDSReflections(self, reflections, phase):
        element = self.mcaController.element
        self.blockSignals(True)
        rois = []
        # when adding reflections for rois that already exist, remove reflections
        # which the user has deleted previusly. For this we check if the phase already 
        # is present in the rois. Then only the rois for that phase that in the list
        # are replaced by the updated specs (channel, start-end), the rest of the reflectons
        # are ignored.
        

        # first get the current phases in roi_model
        phases = self.roi_model.get_sets(element)

        if phase.name in phases:
            # only keep the intersection between the old and new rois in 
            # a given phase
            current_roi_labels_for_phase = []
            new_rois_labels_for_phase = []
            for r in reflections:
                new_rois_labels_for_phase.append(reflections[r]['label'])

            current_rois = self.roi_model.get_rois_for_use(element)
            for c_roi in current_rois:
                if phase.name in c_roi.label:
                    current_roi_labels_for_phase.append(c_roi.label)
        
            # remove current rois if not in new reflections
            for c_roi in current_roi_labels_for_phase:
                if not c_roi in new_rois_labels_for_phase:
                    self.roi_model.delete_roi_by_name(c_roi, element)

            # then remove new reflections if not in current rois
            for n_roi in new_rois_labels_for_phase:
                if not n_roi in current_roi_labels_for_phase:
                    del reflections[n_roi]

        for name in reflections:
            r = reflections[name]
            rois.append(self.make_roi_by_channel(r['channel'],r['halfwidth'], \
                                                 r['label'],r['name'],r['hkl']))
        rois = self.validate_rois(rois)
    
        self.add_rois_to_mca(rois, detector=element)
        self.blockSignals(False)
        self.update_rois(element)

    def validate_rois(self,rois):
        n = self.mca.nchans
        rois_valid = []
        for roi in rois:
            if roi is not None:
                l = roi.left
                r = roi.right
                if l >=0 and l< n and r >=0 and r< n:
                    rois_valid.append(roi)
        return rois_valid

    def addReflections(self, reflections):
        element = self.mcaController.element
        self.blockSignals(True)
        rois = []
        for r in reflections:
            rois.append(self.make_roi_by_channel(r['channel'],r['halfwidth'],r['label']))
        rois = self.validate_rois(rois)
        self.add_rois_to_mca(rois, detector=element)
        self.blockSignals(False)
        self.update_rois(element)

    def make_roi_by_channel(self, channel, halfWidth=10, label='', jcpds='',hkl=[]):
        cP = channel
        left = cP - halfWidth
        right = cP + halfWidth + 1
        dataLen = self.mca.nchans
        if left >= halfWidth and right <= dataLen-1-halfWidth:
            newRoi = McaROI(left,right,label=label)
            newRoi.jcpds_file = jcpds
            newRoi.hkl=hkl
            return newRoi
        return None

    def addROIbyChannel(self, channel, halfWidth=10, label=''):
        # adds ROI to mca and triggers 
        # a view update to set it as the selected roi in the widgets
        element = self.mcaController.element
        cP = channel
        left = cP - halfWidth
        right = cP + halfWidth + 1
        dataLen = self.mca.nchans
        if left >= halfWidth and right <= dataLen-1-halfWidth:
            newRoi = McaROI(left,right,label=label)
            self.add_rois_to_mca(newRoi, detector=element)
            ind = self.mca.find_roi(newRoi.left,newRoi.right)
            self.selectedROI = ind
            self.update_rois(element)

    def edit_roi_name(self, ind, name):
        rois = self.roi
        element = self.mcaController.element
        if ind >=0 and len(rois)>0 and ind < len(rois):
            
            
            self.roi_model.edit_roi_name(ind, name, element)
            rois = self.roi_model.get_rois_for_use(element)
            self.set_mca_rois(rois, element)
            
            self.update_rois(element)
                
    def update_roi_by_ind(self, ind, use, name, centroid, fwhm, counts, fit_ok):
        self.rois_widget.roi_tw. update_roi(ind, use, name, centroid, fwhm, counts, fit_ok)

    def update_roi_set_by_ind(self, ind, name, use):
        self.rois_widget.roi_sets_tw. update_roi(ind, name, use)

    def roi_name_changed(self, ind, name):
        self.rois_widget.roi_tw. rename_roi(ind, name)


    

    ###############################################################
    #######  roi cursor stuff  ####################################

    def add_roi_cursor_plot(self):
        """
        Adds a roi_cursor to the Pattern view.
        :return:
        """
        label = ''
        positions = [0]
        intensities = [1]
        baseline = [1]
        color = self.pattern_widget.add_roi_cursor(label,
                                              positions,
                                              intensities,
                                              baseline)
        return color

    def show_state_changed(self, ind, state):
        if state:
            self.pattern_widget.show_roi_cursor(ind)
        else:
            self.pattern_widget.hide_roi_cursor(ind)
        pass

    def clear_roi_cursor(self):
        self.show_state_changed(0,False)

    def update_roi_cursor_line(self):
        """
        """
        roi_ind = self.selectedROI
        if len(self.roi) < 1 or roi_ind> (len(self.roi)-1):
            roi_label = ''
            positions = 0
            intensities = 1
            baseline = 1
        else:
            axis_range = self.plotController.getRange()
            y_range = axis_range[1]
            y_max = np.amax(y_range)
            if y_max == 1:
                y_max = np.amax(self.plotController.mca.data[0])
            roi_height = y_max*1.1
            roi = self.roi[roi_ind]
            if self.unit == 'E':
                roi_position = roi.energy
            elif self.unit == 'q':
                roi_position = roi.q
            elif self.unit == 'd':
                roi_position = roi.d_spacing
            elif self.unit == 'Channel':
                roi_position = roi.centroid
            elif self.unit == '2 theta':
                roi_position = roi.two_theta

            roi_label = '%.4f' % (roi_position) + ' '  + self.unit_
            if roi.label != '':
                roi_label = roi_label + ', ' + roi.label 

            #x = self.roi[ind].channels
            #fit = self.roi[ind].yFit
            counts = self.roi[roi_ind].counts
            if len(counts)>0:
                intensities = np.amax(counts)
            else :
                intensities = 1
            baseline = intensities
            log_mode = self.plotController.get_log_mode_y()
            if log_mode:
                baseline = baseline*1.2
                intensities = intensities * 1.9
            else:
                baseline=baseline+y_max*0.015
                intensities=intensities+y_max*0.07
            positions = roi_position
            if intensities <= 1:
                intensities = 1
            if baseline <= 1:
                baseline = 1
        self.pattern_widget.update_roi_cursor_intensities(
                                                    0, 
                                                    [positions], 
                                                    [intensities], 
                                                    [baseline])
        self.pattern_widget.rename_roi_cursor(0,roi_label)
    