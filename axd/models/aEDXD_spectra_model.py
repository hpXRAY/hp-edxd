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

import time
from numpy import asarray
from numpy import abs, append, nonzero, array
from axd.models.aEDXD_functions import *
import copy
from PyQt5.QtCore import QObject, pyqtSignal
from hpm.models.mcaModel import MCA
import os, os.path, sys, platform
from hpm.widgets.UtilityWidgets import save_file_dialog, open_file_dialog, open_files_dialog
from utilities.BackgroundExtraction import extract_background

from .. import data_path, output_path 
from utilities.filt import spectra_baseline

class Spectra():

    def __init__(self):
        #super().__init__()
        self.bin_size = 8
        self.inputdatadirectory = ''
        self.outputsavedirectory = ''
        self.max_peaks = 32
        self.tth_groups = {}
        self.tth = []
        

    def set_autoprocess(self, state):
        for tth in self.tth_groups:
            spectrum = self.tth_groups[tth].spectrum
            spectrum.set_auto_process(state)
            
    def get_dataarray(self):
        tth = self.tth
        tth_groups = self.tth_groups
        dataarray = []
        ttharray = []
        for i, t in enumerate(tth):
            tth_group = tth_groups[t]
            spectrum = tth_group.spectrum
            if len(spectrum.x):
                ttharray.append(t)
                dataarray.append(copy.deepcopy([spectrum.x,spectrum.y_cut]))
        return np.asarray(dataarray), np.asarray(ttharray)

    def save_file(self, outputfile):
        data, tth = self.get_dataarray()
        outputsavedirectory = os.path.dirname(outputfile)
        self.outputsavedirectory =  outputsavedirectory
        base = os.path.basename(outputfile)
        b0 = base.split('.')[0]

        for i in range(len(data)):
            
            fname = os.path.join(outputsavedirectory,b0 +'_'+str("%03d" % i)+'.dat')
            
            try:
                with open(fname,'w') as f:
                    f.write("# EDXD spectrum: E(keV), I(abr.)\n")
                    f.write("# 2th = "+str(tth[i])+"\n")
                    f.write("# %d\n" % len(data[i][0]))
                    for energy,intensity in zip(data[i][0], data[i][1]):
                        f.write("%(e)12.4f %(i)12d\n" % {"e":energy,"i":intensity})
                f.close()
            except Exception as e:
                print(e)
                print("\nThe file has not been saved!")  

    def load_files_from_config(self, file_groups, file_uses = []):
        """ loads data into TthGroup objects
            inputs:
            config - a dict in classic aEDXD format containing 'mcadata' key wiht info about files and their 2th values:
            {'mcadata':[['file1','file2','file3', 15.0],
                        ['file4','file5','file6', 18.0]}
        """
        self.tth_groups = {}
        self.tth = []
        
        for i, filegroup in enumerate(file_groups):
            files = filegroup[:-1]

            filepahts = []
            for f in files:
                
                full = os.path.join(self.inputdatadirectory, f)
              
                filepahts.append(full)
            
            if len(filepahts):
                
                tth = filegroup[-1]
                self.tth.append(tth)
                self.tth.sort()
                if tth in self.tth_groups:
                    group = self.tth_groups[tth]
                else:
                    group = TthGroup(tth,self.bin_size)
                    self.tth_groups[tth]= group
                
                if len(file_uses)>1:
                    uses = file_uses[i][:-1]
                else:
                    uses = []

                group.add_files(filepahts, uses)
           

    def add_files(self, tth, filenames):
        if tth in self.tth_groups:
            group = self.tth_groups[tth]
            group.add_files(filenames)

    def add_tth(self, tth):
        if not tth in self.tth_groups:
            group = TthGroup(tth,self.bin_size)
            group.bin_size = self.bin_size
            self.tth_groups[tth]= group
            self.tth.append(tth)

    def remove_file(self, tth, filename):
        if tth in self.tth_groups:
            group = self.tth_groups[tth]
            group.remove_file(filename)

    def remove_tth(self, tth):
        if tth in self.tth_groups:
            del self.tth_groups[tth]
            ind = self.tth.index(tth)
            del self.tth[ind]

    def set_file_use(self, tth, filename, use):
        if tth in self.tth_groups:
            group = self.tth_groups[tth]
            group.set_file_use(filename, use)
            
    def load_peaks_from_config(self, peaks):
        for peak in peaks:
            tth  = peak['tth']
            if tth in self.tth_groups:
                group = self.tth_groups[tth]
                group.add_cut_roi(peak)

    def get_cut_peaks(self):
        """
        returns all cut peaks in the spectra 
        [{'name':'%.3f-%.3f', 'tth':15}, ...]
        """
        tth = self.tth
        peaks = []
        for t in tth:
            g = self.tth_groups[t]
            s= g.spectrum
            rois = s.rois
            for r, roi in enumerate(rois):
                left = round(roi.left, 3)
                right = round(roi.right, 3)
                peak_params = {'tth':t, 'left':left,'right':right}
                if roi.method == 'baseline':
                    peak_params['method'] = roi.method
                    peak_params['wn'] = roi.wn
                    peak_params['iterations']=roi.iterations
                peaks.append(peak_params)
        return peaks

    def get_file_list(self):
        """
        exports a classic mcadata dict item:
        'mcadata':[['file1','file2','file3', 15.0],
                        ['file4','file5','file6', 18.0]}
        """
        tth = self.tth
        file_groups = []
        for t in tth:
            g = self.tth_groups[t]
            files = list(g.files.keys())
            base_files = []
            for f in files:
                base_files.append(os.path.basename(f))
            file_group = base_files + [t]
            file_groups.append(file_group)
        return file_groups

    def get_file_use_list(self):
        """
        exports a classic mcadata dict item:
        'mcadata':[['file1','file2','file3', 15.0],
                        ['file4','file5','file6', 18.0]}
        """
        tth = self.tth
        file_groups = []
        for t in tth:
            g = self.tth_groups[t]
            files = list(g.files.keys())
            use_files = []
            for f in files:
                fl = g.files[f]
                if fl.use:
                    use = 1
                else:
                    use = 0
                use_files.append(use)
            file_group = use_files + [t]
            file_groups.append(file_group)
        return file_groups

    def clear_files(self):
        while len(list(self.tth_groups.keys())):
            tth = list(self.tth_groups.keys())[0]
            del self.tth_groups[tth]
        while len(self.tth):
            del self.tth[-1]

    def set_bin_size(self, bin_size):
        tth = self.tth
        tth_groups = self.tth_groups
        self.bin_size = bin_size
        for i, t in enumerate(tth):
            tth_group = tth_groups[t]
            tth_group.set_bin_size(bin_size)


class FileData():
    def __init__(self, energy, intensity):
        self.use = True
        self.energy = energy
        self.intensity = intensity

class TthGroup():
    
    def __init__(self, tth=15, bins=8):
        
        self.bin_size = bins
        self.tth = tth
        self.spectrum = Spectrum()
        self.mca_adc_shapingtime = 4e-6
        self.files = {}
        self._energy = []
        self._intensity = []
        self._energy_binned = []
        self._intensity_binned = []
        self.spectrum_raw = [[],[]]
        
        #base = os.path.basename(filename)

    def add_files(self, filenames, uses=[]):
        for i, filename in enumerate(filenames):
            
            energy, intensity = read_file(filename,self.mca_adc_shapingtime)
            f = FileData(energy,intensity)
            if len(uses)== len(filenames):
                f.use = uses[i] == 1
            self.files[filename] = f
            
        self.combine_specra()
        

    def remove_file(self, filename):
        filenames = list(self.files.keys())
        for i, f in enumerate(filenames):
            if filename in f:
                f_key = filenames[i]
                del self.files[f_key]
                self.combine_specra()
                break     

    def set_file_use(self, filename, use):
        filenames = list(self.files.keys())
        for i, f in enumerate(filenames):
            if filename in f:
                f_key = filenames[i]
                self.files[f_key].use = use
                self.combine_specra()
                break  

    def combine_specra(self ):
        group = self.files
        energy = []
        intensity = []
        for filename in group:
            s = group[filename]
            if s.use:
                e = s.energy
                i = s.intensity
                energy.append(e)
                intensity.append(i)
        if len(energy):
            energy = np.mean(energy,0)
            intensity = np.sum(intensity,0)
        
        [self._energy, self._intensity] = [energy, intensity]
        self.spectrum_raw = copy.deepcopy([energy, intensity])

        self.bin()

    def bin(self):
        bsize = self.bin_size
        # binning the data
        xb = fastbin(self._energy,bsize) # average energy for binned data
        yb = fastbin(self._intensity,bsize)*bsize # for better counting statistics  WHYYYYYYY !!!!???????
        [self._energy_binned, self._intensity_binned]=  [xb,yb]
        x = copy.deepcopy(self._energy_binned)
        y = copy.deepcopy(self._intensity_binned)
        self.spectrum.bin_size = bsize
        self.spectrum.set_data(x, y)

    def set_bin_size(self, bsize):
        self.bin_size = bsize
        self.bin()

    def add_cut_roi(self, roi_params):
        left = roi_params['left']
        right = roi_params['right']
        roi = ROI(left,right)
        if 'method' in roi_params:
            roi.method = roi_params['method']
            if roi.method == 'baseline':
                if 'wn' in roi_params:
                    roi.wn = roi_params['wn']
                if 'iterations' in roi_params:
                    roi.iterations = roi_params['iterations']
        self.spectrum.add_roi(roi)

    

class Spectrum(QObject):
    ''' this class is based on some relevant parts of MCA class by M. Rivers '''
    spectrum_changed_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.x = []
        self.y = []
        self.bg = []
        self.unscaled = []
        self.y_cut = []
        self.auto_process = False
        self.max_rois = 32
        self.rois = []
        self.bin_size = 8
        self.peak_cutting = {}
        
    def set_auto_process(self,mode=False):
        self.auto_process =mode
        if self.auto_process:
            self.compute_spectrum()
  
    def __len__(self):
        return len(self.rois)

    def set_data(self, x, y):
        self.x = x

        
        self.y = y
        self.y_cut = y


        
        if self.auto_process:
            self.compute_spectrum()

    def compute_spectrum(self, ind=None):
        if len(self.y):
            self.y_cut=copy.deepcopy(self.y)
            if len(self.rois):
                for i, roi in enumerate(self.rois):
                    if roi.use:
                        #if ind !=None or ind == i:
                        self.compute_roi(roi)
            
            self.unscaled = [self.x,self.y_cut/self.bin_size]
       

    def compute_roi(self, roi):
        xb = self.x
        yb = self.y_cut

        #
        if len(xb):
            # TODO peak cut side regions size, binnin size, smoothing options
            t = [roi.left, roi.right]
            ei = np.abs(float(t[0])- xb).argmin()
            ef = np.abs(float(t[1])- xb).argmin()
            txa1 = xb[ei-11:ei+7]
            txa = fastbin(txa1,4)
            txa2 = xb[ef:ef+16]
            txb = fastbin(txa2,4)
            txc = np.append(txa, txb)
            tya1=yb[ei-11:ei+7]
            tya2=yb[ef:ef+16]
            tya = fastbin(tya1,4)
            tyb = fastbin(tya2,4)
            tyc = np.append(tya, tyb)
            roi.peak_cutting['tc']= [np.append(txa1, txa2),np.append(tya1, tya2)]
            roi.peak_cutting['tc_binned']= [txc,tyc]
            #fastbin
            # this is the part that needs to be improved 
            # spline interploation will be replaced by smooth background function calculation
            x_interp = xb[ei-1:ef+1]
            if roi.method == 'baseline':
                wn = roi.wn
                iterations = roi.iterations
                interp = spectra_baseline(yb[ei-1:ef+1],wn,iterations)
                
                yb[ei-1:ef+1] = interp
            elif roi.method == 'spline':
                spl = interpolate.UnivariateSpline(
                        txc,tyc,s = roi.smooth_width)
                #,w=1/np.sqrt(tyc)
                interp = spl(x_interp)
                yb[ei-1:ef+1] = interp
            roi.peak_cutting['interp'] = [x_interp,interp]
            self.y_cut = yb

    def find_roi_by_name(self, name):
        index = 0
        for roi in self.rois:
            if (name == roi.name) : return index
            index = index + 1
        return -1

    def set_new_roi_bounds(self, ind, l, r):
        roi = self.rois[ind]
        roi.left = l
        roi.right = r
        left = '%.3f' % (l)
        right = '%.3f' % (r)
        roi.name = left +'-'+right
        self.compute_spectrum(ind)

    def del_roi(self, ind):
        del self.rois[ind]
        if self.auto_process:
            self.compute_spectrum()

    ########################################################################
    def add_roi(self, roi, energy=0, detector=0):
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
            roi = ROI()
            roi.left = 500
            roi.right = 600
            roi.label = 'Fe Ka'
        """
        r = copy.copy(roi)
        name = '%.3f-%.3f' % (r.left,r.right)
        r.name= name

        exists = self.find_roi_by_name(name) > -1
        # don't add duplicate rois
        if not exists:
            if self.auto_process :
                self.compute_roi(r)
            
            self.rois.append(r)

            # Sort ROIs.  This sorts by left channel.
            tempRois = copy.copy(self.rois)
            tempRois.sort()
            
            self.rois = tempRois

            if self.auto_process:
                self.compute_spectrum()

    def change_roi_use(self, ind, use):
        roi = self.rois[ind].use = use
        if self.auto_process:
            self.compute_spectrum()
    
    def clear_rois(self):
        
        self.set_rois([])
        if self.auto_process:
            self.compute_spectrum()

    def set_rois(self, rois, energy=0, detector=0):
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
        self.rois = [[]]
        
        for roi in rois:
            if self.auto_process :
                self.compute_roi(roi, 0)
            
            '''
            if (energy == 1):
                r.left =  self.calibration.energy_to_channel(r.left, clip=1)
                r.right = self.calibration.energy_to_channel(r.right, clip=1)
            '''
            self.rois.append(roi)
    ########################################################################
    def add_rois(self, rois, energy=0, detector = 0):
        """
        Adds multiple new ROIs to the epicsMca.

        Inputs:
            rois:
                List of McaROI objects.
        """
       
        n_new_rois = len(rois)
        nrois = len(self.rois)
        new_total = n_new_rois + nrois
        if new_total > self.max_rois:
            extra = new_total - self.max_rois 
            keep = n_new_rois - extra
            rois = rois[:keep]

        for roi in rois:
            
            r = copy.copy(roi)
            if self.auto_process :
                self.compute_roi(r, detector)
            '''
            if (energy == 1):
                r.left = self.calibration.energy_to_channel(r.left, clip=1)
                r.right = self.calibration.energy_to_channel(r.right, clip=1)
            '''
            self.rois.append(r)

            # Sort ROIs.  This sorts by left channel.
            tempRois = copy.copy(self.rois)
            tempRois.sort()
            
            self.rois = tempRois

    def get_rois(self, energy=0):
        """ Returns the Mca ROIS, as a list of McaROI objects """
        
        rois = copy.copy(self.rois)
        '''
        if (energy != 0):
            for roi in rois:
                roi.left = self.calibration.channel_to_energy(roi.left)
                roi.right = self.calibration.channel_to_energy(roi.right)
        '''
        return rois


class ROI():
    """
    Class that defines a Region-Of-Interest (ROI)
    Fields
        .left      # Left channel or energy
        .right     # Right channel or energy
        .centroid  # Centroid channel or energy
        .fwhm      # Width
        .bgd_width # Number of channels to use for background subtraction
        .use       # Flag: should the ROI should be used for energy calibration
        .preset    # Is this ROI controlling preset acquisition
        .label     # Name of the ROI
        .d_spacing # Lattice spacing if a diffraction peak
        .energy    # Energy of the centroid for energy calibration
    """
    def __init__(self, left=0., right=0., centroid=0., fwhm=0., bgd_width=0,
                    use=1, preset=0, label='', d_spacing=0., energy=0., q = 0.):
        """
        Keywords:
            There is a keyword with the same name as each attribute that can be
            used to initialize the ROI when it is created.
        """
        self.name = ''
        self.left = left
        self.right = right
        self.centroid = centroid
        self.bgd_width = bgd_width
        self.use = use
        self.label = label
        self.yFit = []
        self.x_yfit = []
        self.channels = []
        self.counts = []
        self.method = 'spline'
        self.smooth_width=40
        self.iterations=50
        self.wn = 0.2
        self.cheb_order=50
        self.peak_cutting={}
        
    def __lt__(self, other): 
        """
        Comparison operator.  The .left field is used to define ROI ordering
            Updated for Python 3 (python 2 used __cmp__ syntax)
        """ 
        return self.left < other.left
    def __eq__(self, other):
        """
        Equals operator. useful for epics MCA, only update the record in left, right or label changes.
        """
        eq = self.left == other.left and  self.right == other.right and  self.label == other.label
        return eq
    

def dataparse(mcadata, inputdatadirectory, mca_adc_shapingtime):
    datalist=[]; tth=[]; filelist=[]
    for i in range(len(mcadata)):
        files2read = []
        for files in mcadata[i][:-1]:
            files2read.append(inputdatadirectory+files)
        tthi = mcadata[i][-1]
        energy = []
        intensity = [] #np.zeros(len(lines[firstdataline_ndx:]))
        for filename in files2read:
            energyi, intensityi = read_file(filename,mca_adc_shapingtime)
            energy.append(energyi)
            intensity.append(intensityi)
        energy = np.mean(energy,0)
        #intensity = np.sum(intensity,0)
        datalist.append([energy,intensity]) # count as measured
        tth.append(tthi)
        filelist.append(files2read)

    return datalist, tth, filelist

def read_file(filename, mca_adc_shapingtime):
    f = open(filename,'r')
    lines = f.readlines()
    for line in lines:
        if 'REAL_TIME:' in line: rt_ndx = lines.index(line)
        if 'LIVE_TIME:' in line: lt_ndx = lines.index(line)
        if 'CAL_OFFSET:' in line: e_offset_ndx = lines.index(line)
        if 'CAL_SLOPE:' in line: e_slope_ndx = lines.index(line)
        if 'TWO_THETA:' in line: two_theta_ndx = lines.index(line)
        if 'DATA:' in line: 
            firstdataline_ndx = lines.index(line)+1
            break
    real_time = float(lines[rt_ndx].split()[1])
    live_time = float(lines[lt_ndx].split()[1])
    e_offset = float(lines[e_offset_ndx].split()[1])
    e_slope = float(lines[e_slope_ndx].split()[1])
    two_theta = float(lines[two_theta_ndx].split()[1])
    intensityi = np.array(lines[firstdataline_ndx:],dtype=float)
    # correct the dead time here: 
    dead_time = real_time-live_time
    y = intensityi/live_time # counting rate
    lost_counts = (1-np.exp(-y*mca_adc_shapingtime))*y*dead_time
    intensityi = intensityi+lost_counts
    energyi = e_offset + e_slope*np.arange(len(intensityi)) # keV
    f.close()
    return energyi, intensityi

def get_tth_from_file(filename):
    f = open(filename,'r')
    lines = f.readlines()
    for line in lines:
        if 'TWO_THETA:' in line: 
            two_theta_ndx = lines.index(line)
            break
    f.close()
    two_theta = float(lines[two_theta_ndx].split()[1])
    return two_theta

def dataread(mcadata, inputdatadirectory):
    datalist=[]; tth=[]; filelist=[]
    for i in range(len(mcadata)):
        files2read = []
        for files in mcadata[i][:-1]:
            files2read.append(inputdatadirectory+files)
        tthi = mcadata[i][-1]
        energy = []
        intensity = []
        for files in files2read:
            try:
                readin = np.genfromtxt(files,dtype=float,comments='#')
                energyi = readin[:,0]
                intensityi = readin[:,1]
                energy.append(energyi)
                intensity.append(intensityi)
            except (ValueError, TypeError, IndexError) as e:
                print(e)
        energy = np.mean(energy,0)
        #intensity = np.sum(intensity,0)
        datalist.append([energy,intensity]) # count as measured
        tth.append(tthi)
        filelist.append(files2read)
    return datalist, tth, filelist 

def get_slope_range(y):
    
    x_r = len(y)
    y_0 = y[0] 
    y_f = y[-1] 
    y_r = y_f-y_0
    slope = y_r/x_r
    x_range = range(x_r)
    y_range = np.asarray(x_range)*slope
    return y_range