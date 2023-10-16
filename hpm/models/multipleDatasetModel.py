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


from fileinput import filelineno

from operator import truediv
import numpy as np
from scipy import interpolate
from PyQt5 import QtCore, QtWidgets
#from pyqtgraph.functions import pseudoScatter
import os
import time
import copy

from .mcareaderGeStrip import *
from .mcaModel import McaCalibration, McaElapsed, McaROI, McaEnvironment, MCA
from utilities.CARSMath import fit_gaussian
from .eCalModel import calc_parabola_vertex, fit_energies
from .mcaComponents import McaROI
import hpm.models.Xrf as Xrf

from hpm.models.MaskModel import MaskModel
from hpm.models.multiple2thetaCalibrationModel import Multiple2ThetaModel

class MultipleSpectraModel(QtCore.QObject):  # 
    def __init__(self,  *args, **filekw):
        
        """
        Creates new Multiple Spectra object.  
    
        Example:
            m = MultipleSpectraModel()
        """
        self.mca : MCA
        self.mca = None
   
        
        self.max_spectra = 500
        self.nchans = 4000

        self.data = []
        self.data_average = []


        self.calibration = {}
        self.calibration_inv = {}

        self.channel_calibration_scales = []

        self.rebinned_channel_data = []
        self.rebinned_channel_average = []

        self.E = []
        self.E_average = []

        self.q = []
        self.q_average = []

        self.current_scale = {'label': 'channel', 'range': [1,0]}
        self.q_scale = [1 , 0]
        self.E_scale = [1 , 0]
        self.tth_scale = [1,0]
        
        self.r = {'files_loaded':[],
                'start_times' :[],
                'data': np.empty((0,0)),
                'calibration':[],
                'elapsed':[]}

        range_1 = [667,1100]
        range_2 = [3000,3480]
        range_3 = [2600,3000]
        self.alignment_rois = []
        self.multi_angle_calibration_model = Multiple2ThetaModel()

    def clear(self):
        self.__init__()

    def set_mca(self, mca, element=0):
        self.clear()
        self.mca = mca

        self.data = np.asarray(mca.get_data())



    def flaten_data(self, data, mask):

        # Compute the average beam profile by averaging all the rows while in energy space. 
        # Then, convert that average profile to q space then you can use that to 
        # normalize all of the data rows.
        # do a weighted average because the high energy / low energy bins will be more noisy 
        out = np.mean(np.ma.array(data, mask=mask), axis=0 )
        return out

    def is2thetaScan(self):
        '''
        Returns True if the mca is a 2theta scan, i.e. each element (detector) is collected 
        at a different 2theta. Otherwise returns False. Essentially checks if 
        the 2theta for each element is diffetent. 
        '''
        return True
        

    def energy_to_2theta(self):
        '''
        transposes the 2D dataset, converts from 2D EDX to 2D ADX
        '''
        
        data_t = np.transpose(self.data)
        s = np.shape(data_t)
        n_det = s[0]
        n_cnan = s[1]
        cal_t = []
        
        calibrations = self.mca.get_calibration()
        old_cal : McaCalibration
        old_cal = calibrations[0]

        tths = []
        for n in range(len(calibrations)):
            cal_n : McaCalibration
            cal_n = calibrations[n]
            tth = cal_n.two_theta
            tths.append(tth)
        offset = tths[0]
        slope = abs(tths[1] - tths[0])

        cal = McaCalibration()
        
        cal.slope = slope
        cal.offset = offset
        cal.set_dx_type('adx')

        for n in range(n_det):
            
            E = old_cal.channel_to_energy(n)
            new_cal_n = copy.deepcopy(cal)
            new_cal_n.wavelength = E
            cal_t.append(new_cal_n)
     
        print(len(cal_t))

    def _rebin_scale(self, data, new_data, scale, new_scale):
        
        rows = len(data)
        tth = np.zeros(rows)
        bins = np.size(self.data[0])
        x = np.arange(bins)
        calibrations = self.mca.get_calibration()
        rebinned_scales = []
    
        if new_scale == 'q':
            for row in range(rows):
                calibration = calibrations[row]
                q = calibration.channel_to_q(x)
                tth[row]= calibration.two_theta
                rebinned_scales.append(q)
        elif new_scale == 'E':
            for row in range(rows):
                calibration = calibrations[row]
                e = calibration.channel_to_energy(x)
                tth[row]= calibration.two_theta
                rebinned_scales.append(e)
        tth_min = np.amin(tth)
        tth_max = np.amax(tth)
        if tth_max != tth_min:
            tth_step = (tth_max-tth_min)/rows
            self.tth_scale = [tth_step, tth_min]

        rebinned_scales = np.asarray(rebinned_scales)
        rebinned_min = np.amin( rebinned_scales)
        rebinned_max = np.amax(rebinned_scales)
     
        rebinned_step = (rebinned_max-rebinned_min)/bins
        if new_scale == 'q':
            self.q_scale = [rebinned_step, rebinned_min]
        elif new_scale == 'E':
            self.E_scale = [rebinned_step, rebinned_min]
        rebinned_new = [x*rebinned_step+rebinned_min]*rows
        self.align_multialement_data(data, new_data, rebinned_scales,rebinned_new )

        #self.align_multialement_data(mask, new_mask , rebinned_scales,rebinned_new ,kind='nearest')
        
    def angle_calibration_gsd_set_data(self):
        data = self.E
        
        self.multi_angle_calibration_model.set_data(data)

    def angle_calibration_gsd_add_pt(self, x,y):
        x_range, y_range = self.multi_angle_calibration_model.add_point(x,y)
        
        return x_range, y_range

    def angle_calibration_gsd_calc(self):
        pass

    def rebin_scale(self, new_scale='q'):
        data = self.data
        
        if new_scale == 'q':
            new_data = self.q
           
       
        elif new_scale == 'E':
            new_data = self.E
            
            
        self._rebin_scale(data, new_data, 'Channel', new_scale)

    def rebin_channels(self, order = 1):
        # This is useful for the germanium strip detector data, 
        # where the rows have to be aligned before processing
        if len(self.alignment_rois)<2:
            return
     
       
        range_1 = self.alignment_rois[0]
        range_2 = self.alignment_rois[1]
        
        if len(self.alignment_rois)>2:
            range_3 = self.alignment_rois[2]

        data = self.data
        bins = np.size(data[0])
        x =  np.arange(bins)
        rows = len(data)
        new_scales = [x]*rows
        if True: #not len(self.channel_calibration_scales):
            
            now = time.time()
            

            max_points_left = np.zeros(rows)
            max_points_right = np.zeros(rows)
            max_points_middle= np.zeros(rows)

            fit_range = 20
            for row in range(rows):
                max_rough = int(range_1[0]+ np.argmax(data[row][slice(*range_1)]))
                if max_rough == 0:
                    continue
                fit_segment_x = x[max_rough-fit_range:max_rough+fit_range]
                fit_segment_y = data[row][max_rough-fit_range:max_rough+fit_range]
                min_y = np.amin(fit_segment_y)
                _ , centroid,_ = fit_gaussian(fit_segment_x,fit_segment_y - min_y)
                max_points_left[row] = centroid

                max_rough = int(range_2[0] + np.argmax(data[row][slice(*range_2)]))
                fit_segment_x = x[max_rough-fit_range:max_rough+fit_range]
                fit_segment_y = data[row][max_rough-fit_range:max_rough+fit_range]
                min_y = np.amin(fit_segment_y)
                _ , centroid,_ = fit_gaussian(fit_segment_x,fit_segment_y- min_y)
                max_points_right[row] = centroid

                if order == 2:
                    max_rough = int(range_3[0] + np.argmax(data[row][slice(*range_3)]))
                    fit_segment_x = x[max_rough-fit_range:max_rough+fit_range]
                    fit_segment_y = data[row][max_rough-fit_range:max_rough+fit_range]
                    min_y = np.amin(fit_segment_y)
                    _ , centroid,_ = fit_gaussian(fit_segment_x,fit_segment_y- min_y)
                    max_points_middle[row] = centroid
            
            max_points_left_ =  max_points_left[max_points_left != 0]
            max_points_right_ = max_points_right[max_points_right != 0]
            max_points_middle_ = max_points_middle[max_points_middle != 0]
            left = np.mean(max_points_left_)
            right = np.mean(max_points_right_)
            middle = np.mean(max_points_middle_)
        
            slope = np.ones(rows)   # relative slopes
            offset = np.zeros(rows)  # relative y-intercepts
            quad = np.zeros(rows)  # quad coefficient

            slope_inv = np.ones(rows)   # relative slopes
            offset_inv = np.zeros(rows)  # relative y-intercepts
            quad_inv = np.zeros(rows)  # quad coefficient
            
            for row in range(rows):
                if max_points_left[row] == 0:
                    continue
                x1 = left
                x2 = right
                
                y1 = max_points_left[row]
                y2 = max_points_right[row]
                
                if order == 1:
                    slope[row] = (y1-y2)/(x1-x2)
                    offset[row] = (x1*y2 - x2*y1)/(x1-x2)
                    slope_inv[row] = (x1-x2)/(y1-y2)
                    offset_inv[row] = (y1*x2 - y2*x1)/(y1-y2)
                elif order == 2:
                    x3 = middle
                    y3 = max_points_middle[row]
                    quad[row],slope[row],offset[row] = calc_parabola_vertex(x1,y1,x2,y2,x3,y3)
                    quad_inv[row],slope_inv[row],offset_inv[row] = calc_parabola_vertex(y1,x1,y2,x2,y3,x3)

            calibration = {}
            calibration['slope'] = slope
            calibration['offset'] = offset
            calibration['quad'] = quad
            calibration_inv = {}
            calibration_inv['slope'] = slope_inv
            calibration_inv['offset'] = offset_inv
            calibration_inv['quad'] = quad_inv
            self.calibration = calibration
            self.calibration_inv = calibration_inv

            self.channel_calibration_scales = self.create_multialement_alighment_calibration(data, calibration)
       
        self.align_multialement_data(data , self.rebinned_channel_data, new_scales, self.channel_calibration_scales )
        
    def create_multialement_alighment_calibration(self, data, calibration):
        rows = len(data)
        bins = np.size(data[0])
        x = np.arange(bins)
        slope = calibration['slope']
        offset = calibration['offset']
        quad = calibration['quad']
        calibration_scales = []
        for row in range(rows): 
            xnew = x * slope[row] + offset[row]
            if quad[row] != 0:
                xnew = xnew + quad[row] * x * x
            calibration_scales.append(xnew)
        return calibration_scales
            
    def align_multialement_data (self,  data, new_data, old_scales, new_scales, kind='linear'):
        rows = len(data)
        
        bins = np.size(data[0])
        x = np.arange(bins)
        for row in range(rows): 
            x = old_scales[row]
            xnew = new_scales[row]
            new_data[row] = self.shift_row(data[row],x, xnew, kind)
        

    def shift_row(self, row,x, xnew, kind='linear'):
        f = interpolate.interp1d(x, row, assume_sorted=True, bounds_error=False, fill_value=0, kind=kind)
        row = f(xnew)
        return row



    def aligned_to_channel(self, aligned, row):
        slope = self.calibration['slope'][row]
        offset = self.calibration['offset'][row]
        channel = aligned * slope + offset
        quad = self.calibration['quad'][row]
        if quad != 0:
            channel = channel + quad * aligned * aligned
        return channel

    def channel_to_aligned(self, channel, row):
        slope = self.calibration_inv['slope'][row]
        offset = self.calibration_inv['offset'][row]
        aligned = channel * slope + offset
        quad = self.calibration_inv['quad'][row]
        if quad != 0:
            aligned = aligned + quad * channel * channel
        return aligned

    def make_aligned_rois(self, row, rois):
        
        all_new_rois = []
        for roi in rois:
            left = roi.left
            right = roi.right
            aligned_left = self.channel_to_aligned(left, row)
            aligned_right = self.channel_to_aligned(right, row)
            roi.left = aligned_left
            roi.right = aligned_right
        for det in range(self.mca.n_detectors):
            new_rois = []
            for roi in rois:
                aligned_left = roi.left
                aligned_right = roi.right
                left = int(round(self.aligned_to_channel(aligned_left,det)))
                right = int(round(self.aligned_to_channel(aligned_right,det)))
                new_roi = McaROI(left,right,label = roi.label)
                new_rois.append(new_roi)
            all_new_rois.append(new_rois)
        return all_new_rois

    def calibrate_all_elements(self, order = 1):
        calibration = self.mca.get_calibration()
        all_rois = self.mca.get_rois()
        det0_rois = all_rois[0]
        energies = []
        for r in det0_rois:
            energy = Xrf.lookup_xrf_line(r.label)
            if (energy == None):
                energy = Xrf.lookup_gamma_line(r.label)
            if (energy != None): 
                energies.append( energy)
        for det in range(self.mca.n_detectors):
            cal = calibration[det]
            rois = all_rois[det]
            for i, roi in enumerate(rois):
                roi.energy = energies[i]
            fit_energies(rois, order,cal)


    def add_new_alignment_roi(self, center):
        new_roi = [int(center-150), int(center+150)]
        self.alignment_rois.append(new_roi)
        return new_roi

    def set_alignment_rois(self, rois):
        self.alignment_rois = rois

    def delete_alignment_roi(self):
        if len(self.alignment_rois):
            del self.alignment_rois[-1]

    def clear_alignment_rois(self):
        self.alignment_rois = []