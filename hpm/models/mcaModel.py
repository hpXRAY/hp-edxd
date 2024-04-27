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



import numpy as np


import utilities.centroid as centroid
import copy
import json
import time

import os

from utilities.filt import spectra_baseline
from .mcaComponents import *
from .mcaIO import *

from epics import caput, caget, PV

class MCA():  # 
    

    """ Device-independent MultiChannel Analyzer (MCA) class """
    def __init__(self, file=None, file_settings = None, *args, **filekw):
        """
        Creates new Mca object.  The data are initially all zeros, and the number
        of channels is 4000.
        
        Keywords:
            file:
                Name of a file to read into the new Mca object with read_file()
                
        Example:
            m = Mca('my_spectrum.dat')
        """
        
        self.dx_type = 'edx'

        self.name = ''
        self.file_name = ''
        self.files_loaded = []
        self.file_saved_timestamp = None
        self.file_settings = file_settings # not used in this based class, used in device dependent class

        self.max_rois = 32
        self.n_detectors = 1
        self.nchans = 4000
        self.data = [np.ones(self.nchans)]
        self.data_processed = np.ones(self.nchans)
        self.baseline = [np.ones(self.nchans)]
        self.baseline_state = 0
        
        self.rois = [[]]
        self.rois_from_file = [[]]
        self.rois_from_det = [[]]
        self.auto_process_rois = True

        self.calibration = [McaCalibration()]
        self.calibration_persistent = []
        #self.calibration_processed = McaCalibration()
        self.elapsed = [McaElapsed()]
        self.presets = McaPresets()
        self.environment = [[]]

        self.wavelength = None  # persistent wavelength for axd mode useful when loading files without wavelength, e.g. *.chi
        #if (file != None):
        #    self.read_file(file, **filekw)
        self.fileIO = mcaFileIO()

        

    ########################################################################
    def get_calibration(self) -> list : 
        """ Returns the Mca calibration, as an McaCalibration object """
        return self.calibration

    ########################################################################
    def set_calibration(self, calibration, det = 0):
        """
        Sets the Mca calibration.
        
        Inputs:
            calibration:
                An McaCalibration object
        """
        if det > len(self.calibration)-1:
            diff = det - (len(self.calibration)-1)
            for i in range(diff):
                self.calibration.append(None)
        self.calibration[det] = calibration

    

    ########################################################################
    def get_presets(self):
        """ Returns the Mca presets, as an McaCPresets object """
        return self.presets

    ########################################################################
    def set_presets(self, presets):
        """
        Sets the Mca presets.
        
        Inputs:
            presets:
                An McaPresets object
        """
        self.presets = presets

    ########################################################################
    def get_elapsed(self):
        """ Returns the Mca elapsed parameters, as an McaElapsed object """
        return self.elapsed

    ########################################################################
    def set_elapsed(self, elapsed):
        """
        Sets the Mca elapsed parameters.
        
        Inputs:
            elapsed:
                An McaElapsed object
        """
        self.elapsed = copy.copy(elapsed)

    ########################################################################
    def get_name(self):
        """ Returns the Mca name as a string """
        return self.name

    def get_name_base(self):
        name = os.path.basename(self.file_name)
        
        return name


    

    ########################################################################
    def set_name(self, name):
        """
        Sets the Mca name.
        
        Inputs:
            name:
                A string
        """
        self.name = name

    ########################################################################
    
    def get_file_rois(self, energy=0):
        """ Returns the Mca ROIS, as a list of McaROI objects """
        rois = copy.copy(self.rois_from_file)
     
        return rois 
    
    def get_det_rois(self, energy=0):
        """ Returns the Mca ROIS, as a list of McaROI objects """
        rois = copy.copy(self.rois_from_det)
       
        return rois 
    
    def get_rois(self, energy=0):
        """ Returns the Mca ROIS, as a list of McaROI objects """
        rois = copy.copy(self.rois)
      
        return rois
    
    def get_rois_by_det_index(self, detector=0):
        """ Returns the Mca ROIS, as a list of McaROI objects """
        rois = copy.copy(self.rois[detector])
      
        return rois

    def get_rois_offline(self, energy=0):
        """ Returns the Mca ROIS, as a list of McaROI objects, dont override with epics mca etc """
        
        rois = copy.copy(self.rois)
        '''
        if (energy != 0):
            for roi in rois:
                roi.left = self.calibration.channel_to_energy(roi.left)
                roi.right = self.calibration.channel_to_energy(roi.right)
        '''
        return rois

    def compute_get_rois(self):
        self.compute_rois()

        rois = copy.copy(self.rois)

    def compute_rois(self):
        for det in self.rois:
                for roi in det:
                    self.compute_roi(roi, det)

    def compute_roi(self, roi, detector=0):
        ## computes and updates the centroid/energy/fwhm of a roi
        ## ideally should be called when a new roi is added or data updated
        ## the goal is to simplify things for users of this class, centroids/energy of roi 
        ## should always be ready for the user 
        #start_time = time.time()
        
        available_scales = self.calibration[detector].available_scales
        [roi, fwhm_chan] = centroid.computeCentroid(self.data[detector], roi, 1)
        roi.fwhm = fwhm_chan
        if 'E' in available_scales:
            roi.fwhm_E = (self.calibration[detector].channel_to_energy(roi.centroid + 
                                            fwhm_chan/2.) - 
                                self.calibration[detector].channel_to_energy(roi.centroid - 
                                            fwhm_chan/2.))
            roi.energy = self.calibration[detector].channel_to_energy(roi.centroid)
        if 'q' in available_scales:
            roi.fwhm_q = (self.calibration[detector].channel_to_q(roi.centroid + 
                                            fwhm_chan/2.) - 
                                self.calibration[detector].channel_to_q(roi.centroid - 
                                            fwhm_chan/2.))
            roi.q = self.calibration[detector].channel_to_q(roi.centroid)
        if 'd' in available_scales:                                    
            roi.fwhm_d = (self.calibration[detector].channel_to_d(roi.centroid + 
                                            fwhm_chan/2.) - 
                                self.calibration[detector].channel_to_d(roi.centroid - 
                                            fwhm_chan/2.))      
            roi.d_spacing = self.calibration[detector].channel_to_d(roi.centroid)
        if '2 theta' in available_scales:                                
            roi.fwhm_tth = (self.calibration[detector].channel_to_tth(roi.centroid + 
                                            fwhm_chan/2.) - 
                                self.calibration[detector].channel_to_tth(roi.centroid - 
                                            fwhm_chan/2.))                             
            roi.two_theta = self.calibration[detector].channel_to_tth(roi.centroid)
        

    ########################################################################
    def set_rois(self, rois, energy=0, detector=0, source='file'):
        """
        Sets the region-of-interest parameters for the MCA.
        The rois information is contained in an object of class McaRoi.
        This routine is not needed if the information in the McaRoi instance
        is already in channel units.  It is needed if the information in the
        .left and .right fields is in terms of energy.

        Inputs:
            rois:
                A list of objects of type McaROI
                
        Keywords:
            energy:
                Set this flag to indicate that the .left and .right fields
                of rois are in units of energy rather than channel number.
                
        Example:
            mca = Mca('mca.001')
            r1 = McaROI()
            r1.left = 5.4
            r1.right = 5.6
            r2 = McaROI()
            r2.left = 6.1
            r2.right = 6.2
            mca.set_rois([r1,r2], energy=1)
        """
        if source == 'file':
            self.rois_from_file[detector] = []
            set_rois = self.rois_from_file
        elif source == 'controller':
            self.rois[detector]  = []
            set_rois = self.rois
        elif source == 'detector':
            self.rois_from_det[detector]  = []
            set_rois = self.rois_from_det
        
        for roi in rois:
            '''if self.auto_process_rois :
                self.compute_roi(roi, 0)'''
            
            '''
            if (energy == 1):
                r.left =  self.calibration.energy_to_channel(r.left, clip=1)
                r.right = self.calibration.energy_to_channel(r.right, clip=1)
            '''
            lbl = roi.label
            if len(lbl)> 12:
                if ' ' in lbl:
                    parts = lbl.split(' ')
                    end = parts[-1]
                    end = end[:3]
                    start = parts[0]
                    start = start[:8]
                    lbl = start + ' ' +end
                else:
                    lbl = lbl[:12]
            roi.label = lbl
            set_rois[detector].append(roi)


    ########################################################################
    
    def add_roi(self, roi, energy=0, detector=0, source='file'):
        """
        This procedure adds a new region-of-interest to the MCA.

        Inputs:
            roi: An object of type mcaROI.
            
        Kyetwords:
            energy:
                Set this flag to 1 to indicate that the .left and .right 
                fields of roi are in units of energy rather than channel 
                number.
                
        Example:
            mca = Mca('mca.001')
            roi = McaROI()
            roi.left = 500
            roi.right = 600
            roi.label = 'Fe Ka'
            mca,add_roi(roi)
        """

        if source == 'file':
            
            set_rois = self.rois_from_file
        elif source == 'controller':
            set_rois = self.rois
        elif source == 'detector':
            set_rois = self.rois_from_det

        r = copy.copy(roi)
        if self.auto_process_rois :
            self.compute_roi(r, detector)
        '''
        if (energy == 1):
            r.left = self.calibration.energy_to_channel(r.left, clip=1)
            r.right = self.calibration.energy_to_channel(r.right, clip=1)
        '''
        set_rois[detector].append(r)

        # Sort ROIs.  This sorts by left channel.
        tempRois = copy.copy(set_rois[detector])
        tempRois.sort()
        
        set_rois[detector] = tempRois
        
    
    def add_rois(self, rois, energy=0, detector = 0, source ='file'):
        """
        Adds multiple new ROIs to the epicsMca.

        Inputs:
            rois:
                List of McaROI objects.
        """

        if source == 'file':
            
            set_rois = self.rois_from_file
        elif source == 'controller':
            set_rois = self.rois

        elif source == 'detector':
            set_rois = self.rois_from_det
       
        n_new_rois = len(rois)
        nrois = len(set_rois[detector])
        new_total = n_new_rois + nrois
        if new_total > self.max_rois:
            extra = new_total - self.max_rois 
            keep = n_new_rois - extra
            rois = rois[:keep]

        for roi in rois:
            r = copy.copy(roi)
            if self.auto_process_rois :
                self.compute_roi(r, detector)
            '''
            if (energy == 1):
                r.left = self.calibration.energy_to_channel(r.left, clip=1)
                r.right = self.calibration.energy_to_channel(r.right, clip=1)
            '''
            set_rois[detector].append(r)

            # Sort ROIs.  This sorts by left channel.
            tempRois = copy.copy(set_rois[detector])
            tempRois.sort()
            
            set_rois[detector] = tempRois

    def change_roi_use(self, ind, use, detector=0):
        roi = self.rois[detector][ind].use = use
            
    
    def clear_rois(self, source, detector):
        self.set_rois([], detector=detector,source=source)

    

    ########################################################################
    def get_data(self):
        """ Returns the data (counts) from the Mca """
        return self.data

    ########################################################################
    def set_data(self, data):
        """
        Copies an array of data (counts) to the Mca.

        Inputs:
            data:
                A Numeric array of data (counts).
        """
        self.data = data
        
        
        
        #if self.auto_process_rois :
        #    for det in self.rois:
        #        for roi in det:
        #            self.compute_roi(roi, det)

    ########################################################################

    def get_baseline(self):
        baselines = []
        for d in self.data:
            
            baselines.append(spectra_baseline(d, 0.04, 50, method='gust'))
       
        self.baselines = baselines
        return self.baselines

    def get_bins(self,detector=0):
        n = len(self.data[detector])
        return np.linspace(0,n-1, n) 

    

    def get_energy(self, detector):
        """
        Returns a list containing the energy of each channel in the MCA spectrum.

        Procedure:
            Simply returns mca.channel_to_energy() for each channel
            
        Example:
            from Mca import *
            mca = Mca('mca.001')
            energy = mca.get_energy()
        """
        channels = np.arange(len(self.data[detector]))
        return self.calibration[detector].channel_to_energy(channels)
    ########################################################################
    def get_roi_counts(self, background_width=1):
        """
        Returns a tuple (total, net) containing the total and net counts of
        each region-of-interest in the MCA.

        Kyetwords:
            background_width:
                Set this keyword to set the width of the background region on either
                side of the peaks when computing net counts.  The default is 1.
                
        Outputs:
            total:  The total counts in each ROI.
            net:    The net counts in each ROI.

            The dimension of each list is NROIS, where NROIS
            is the number of currently defined ROIs for this MCA.  It returns
            and empty list for both if NROIS is zero.
            
        Example:
            mca = Mca('mca.001')
            total, net = mca.get_roi_counts(background_width=3)
            print 'Net counts = ', net
        """
        total = []
        net = []
        nchans = len(self.data)
        for roi in self.rois:
            left = roi.left
            ll = max((left-background_width+1), 0)
            if (background_width > 0):
                bgd_left = sum(self.data[ll:(left+1)]) / (left-ll+1)
            else: bgd_left = 0.
            right = roi.right
            rr = min((right+background_width-1), nchans-1)
            if (background_width > 0):
                bgd_right = sum(self.data[right:rr+1]) / (rr-right+1)
            else: bgd_right = 0.
            total_counts = self.data[left:right+1]
            total.append(sum(total_counts))
            n_sel        = right - left + 1
            bgd_counts   = bgd_left + np.arange(n_sel,dtype=float)/(n_sel-1) * \
                                    (bgd_right - bgd_left)
            net_counts   = total_counts - bgd_counts
            net.append(sum(net_counts))
        return (total, net)
    ########################################################################    

    ########################################################################
    

    ########################################################################
    def find_roi(self, left, right, energy=0, detector =0):
        """
        This procedure finds the index number of the ROI with a specified
        left and right channel number.

        Inputs:
            left:
                Left channel number (or energy) of this ROI
                
            right:
                Right channel number (or energy) of this ROI
            
        Keywords:
            energy:
                Set this flag to 1 to indicate that Left and Right are in units
                of energy rather than channel number.
                
        Output:
            Returns the index of the specified ROI, -1 if the ROI was not found.
            
        Example:
            mca = Mca('mca.001')
            index = mca.find_roi(100, 200)
        """
        l = left
        r = right
        if (energy == 1):
            l = self.calibration.energy_to_channel(l, clip=1)
            r = self.calibration.energy_to_channel(r, clip=1)
        index = 0
        for roi in self.rois[detector]:
            if (l == roi.left) and (r == roi.right): return index
            index = index + 1
        return -1

    ########################################################################
    def delete_roi(self, index, detector=0):
        """
        This procedure deletes the specified region-of-interest from the MCA.

        Inputs:
            index:  The index of the ROI to be deleted, range 0 to len(mca.rois)
            
        Example:
            mca = Mca('mca.001')
            mca.delete_roi(2)
        """
        del self.rois[detector][index]


    ########################################################################

    ########################################################################
    def get_environment(self, detector = 0):
        """
        Returns a list of McaEnvironment objects that contain the environment
        parameters of the Mca.
        """
        
        envs = []
        for env in self.environment:
            val = env.value
            if not np.isscalar(val):
                if len(val)> detector:
                    val = val[detector]
                else:
                    val = ''

            e = McaEnvironment(env.name,val,env.description)
            envs.append(e)
        return envs
        # testing typing on the new laptop stand

    ########################################################################
    def set_environment(self, environment):
        """
        Copies a list of McaEnvironment objects to the Mca object.

        Inputs:
            environment:
                A list of McaEnvironment objects.
        """
        self.environment = environment

    def write_file(self, file, netcdf=0):
        """
        Writes Mca or Med objects to a disk file.
        
        It calls Mca.write_netcdf_file if the netcdf keyword flg is set,

        Note that users who want to read such files with Python are strongly
        encouraged to use Mca.read_file()

        Inputs:
            file:
                The name of the disk file to write.
                
        Keywords:
            netcdf:
                Set this flag to write the file in netCDF format, otherwise
                the file is written in ASCII format.  See the documentation
                for Mca.write_ascii_file and Mca.write_netcdf_file for 
                information on the formats.
    
        Example:
            mca = Mca()
            mca.write_file('mca.001')
        """
        # Call the get_xxx() methods to make sure things are up to date
        data = self.get_data()
        calibration = self.get_calibration()
        elapsed = self.get_elapsed()
        presets = self.get_presets()
        rois = self.get_rois()
        environment = self.get_environment()
        

        if (netcdf != 0):
            pass
            #    write_netcdf_file(file, data, calibration, elapsed, presets, rois, environment)
        else:
            try:
                self.fileIO.write_ascii_file(file, data, calibration, elapsed, presets, rois, environment)
                self.file_name=file
                
            except:
                return [file, False]

        
        return [file, True]

    
    ########################################################################
    
    
    def read_file(self, file, netcdf=0, detector=0):
        """
        Reads a disk file into an MCA object.  If the netcdf=1 flag is set it
        reads a netcdf file, else it assumes the file is ASCII.
        If the data file has multiple detectors then the detector keyword can be
        used to specify which detector data to return.

        Inputs:
            file:
                The name of the disk file to read.
                
        Keywords:
            netcdf:
                Set this flag to read files written in netCDF format, otherwise
                the routine assumes that the file is in ASCII format.
                See the documentation for Mca.write_ascii_file and
                Mca.write_netcdf_file for information on the formats.

            detector:
                Specifies which detector to read if the file has multiple detectors.
           
                
        Example:
            mca = Mca()
            mca.read_file('mca.001')
        """
        if (netcdf != 0):
            pass
            #r = read_netcdf_file(file)
        else:
            if file.endswith('.mca'):
                [r, success] = self.fileIO.read_mca_file(file)
            elif file.endswith('.chi'):
                [r, success] = self.fileIO.read_chi_file(file, wavelength=self.wavelength)
                wavelength = r['calibration'][0].wavelength
                self.wavelength = wavelength
            elif file.endswith('.xy'):
                [r, success] = self.fileIO.read_chi_file(file, wavelength=self.wavelength)
                wavelength = r['calibration'][0].wavelength
                self.wavelength = wavelength
            elif file.endswith('.nxs'):
                [r, success] = self.fileIO.read_nsx_file(file)
                '''wavelength = r['calibration'][0].wavelength
                self.wavelength = wavelength'''
            else:
                [r, success] = self.fileIO.read_ascii_file(file)
     
        if success == True:
            self.file_name=file
            self.files_loaded = [file]
            self.data = r['data']
            self.nchans = len(r['data'][0])
            self.n_detectors=len(self.data)

            calibration = self.calibration_persistent
            if len(calibration) == self.n_detectors:
                for i, cal in enumerate(calibration):
                    self.set_calibration(cal, i)
            else:
                self.clear_persistent_calibration()
                self.calibration = r['calibration']

            self.rois_from_file = [[]]*self.n_detectors
            self.rois = [[]]*self.n_detectors
            self.elapsed = r['elapsed']
            rois = r['rois']
   
            for i , roi in enumerate(rois):
                self.set_rois(roi, detector=i) 
            self.environment = r['environment']
            self.name = os.path.split(file)[-1]
            self.dx_type = r['dx_type']
    
        return([file,success])

    def load_calibration(self, filename):
        res, success = self.fileIO.read_ascii_file_calibration(filename)
        if success:
            n_det = res['n_detectors']
            if n_det == self.n_detectors:
                calibration = res['calibration']
                self.calibration_persistent = calibration
                for i, cal in enumerate(calibration):
                    self.set_calibration(cal, i)

    def clear_persistent_calibration(self):
        
        self.calibration_persistent = []

    def save_peaks_csv(self,file):

        self.fileIO.write_peaks(file, self.rois[0])

    def export_pattern(self, filename, unit='channel', unit_='#',detector=0,header=None, subtract_background=False):
        """
        Saves the current data pattern.
        :param filename: where to save
        :param header: you can specify any specific header
        :param subtract_background: whether or not the background set will be used for saving or not
        """
        y = self.data[detector]
        channels = np.arange(len(y))

        if filename.endswith('.fxye'):
            unit = 'q'
            unit_=  "A^-1"
        if unit == 'E':
            x = self.calibration[detector].channel_to_energy(channels)
        elif unit == 'q':
            x = self.calibration[detector].channel_to_q(channels)
        elif unit == 'd':
            x = self.calibration[detector].channel_to_d(channels)
        else: x = channels

        if filename.endswith('.xy'):
            self.fileIO.write_pattern(filename,x,y,unit,unit_, header=self._create_xy_header(unit))
        elif filename.endswith('.fxye'):
            self.fileIO.write_pattern(filename,x,y,unit,unit_, header=self._create_fxye_header(filename,"q_A^-1"))
        else:
            self.fileIO.write_pattern(filename,x,y,unit,unit_)

    def _create_xy_header(self, unit):
        """
        Creates the header for the xy file format (contains information about calibration parameters).
        :return: header string
        """
        header = self.make_headers()
        header = header.replace('\r\n', '\n')
        header = '# ' + unit + '\t I'
        return header

    def _create_fxye_header(self, filename, unit):
        """
        Creates the header for the fxye file format (used by GSAS and GSAS-II) containing the calibration information
        :return: header string
        """
        header = 'Generated file ' + filename + ' using hpMCA\n'
        header = header + self.make_headers()
        
        lam = 1
        if unit == 'q_A^-1':
            con = 'CONQ'
        else:
            con = 'CONS'

        header = header + '\nBANK\t1\tNUM_POINTS\tNUM_POINTS ' + con + '\tMIN_X_VAL\tSTEP_X_VAL ' + \
                 '{0:.5g}'.format(lam * 1e10) + ' 0.0 FXYE'
        return header

    def make_headers(self, hdr="#", has_mask=None, has_dark=None, has_flat=None,
                     polarization_factor=None, normalization_factor=None,
                     metadata=None):
        """
        :param hdr: string used as comment in the header
        :type hdr: str
        :param has_dark: save the darks filenames (default: no)
        :type has_dark: bool
        :param has_flat: save the flat filenames (default: no)
        :type has_flat: bool
        :param polarization_factor: the polarization factor
        :type polarization_factor: float

        :return: the header
        :rtype: str
        """
        

        header_lst = "" 
        '''["Mask applied: %s" % has_mask,
                       "Dark current applied: %s" % has_dark,
                       "Flat field applied: %s" % has_flat,
                       "Polarization factor: %s" % polarization_factor,
                       "Normalization factor: %s" % normalization_factor]'''

        if metadata is not None:
            header_lst += ["", "Headers of the input frame:"]
            header_lst += [i.strip() for i in json.dumps(metadata, indent=2).split("\n")]
        header = "\n".join(["%s %s" % (hdr, i) for i in header_lst])

        return header

        
