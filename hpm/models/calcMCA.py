
import numpy as np
#from epics.clibs import *
import copy
import time
import utilities.hpMCAutilities as hpUtil
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QMessageBox
from hpm.models.mcaModel import *
from PyQt5 import QtCore, QtWidgets
from .nxsexport_batch import read_nxs
from .mcaModel import McaCalibration, McaElapsed, McaROI, McaEnvironment

import natsort




class multiFileMCA(MCA):
    """
    Creates a new multiFileMCA. multiFileMCA acts like a regular multi-detector MCA, but can be
        created either by reading a single multi-detector mca file, or by 
        reading multiple single-detector mca files.

    Keywords:
        record_Name:
        The name of the multielement MCA file for the MCA object being created.

        file_options:

        record_name_file:
        
        environment_file:
        This keyword can be used to specify the name of a file which
        contains the names of EPICS process variables which should be saved
        in the header of files written with Mca.write_file().
        If this keyword is not specified then this function will attempt the
        following:
            - If the system environment variable MCA_ENVIRONMENT is set then
                it will attempt to open that file
            - If this fails, it will attempt to open a file called
                'catch1d.env' in the current directory.
                This is done to be compatible with the data catcher program.
        This is an ASCII file with each line containing a process variable
        name, followed by a space and a description field.

    Example:

    """
    def __init__(self, *args,  **kwargs):
        
        super().__init__()
        
  
        self.name = ''

        self.files_loaded = []
        self.verbose = False
        self.mcaRead = None
        
        self.max_rois = 24           
        self.max_spectra = 500
        self.initOK = False           
        self.handle_new_file = lambda new_file: print(f"New file detected: {new_file}")
        self.watch_folder = None 

    ########################################################################
    
    
    
    def read_files(self, *args, **kwargs):
        """
        Reads multiple disk files into a single multi-detector MCA object.  
        The netcdf input is depricated and is not used.
        If the data file has multiple detectors then the detector keyword can be
        used to specify which detector data to return.

        Inputs:
            file:
                The name of the disk file to read.
                
        Keywords:
            netcdf:(deprecated, not used, always input as 0)
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
        success = False
       

        if 'folder' in kwargs:
            folder = kwargs['folder']
        
            if folder == '':
                return
            paths = []
            #files_filtered = []
        
            if os.path.exists(folder):
                files = natsort.natsorted([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and not f.startswith('.')]) 
                for f in files:
                    if f.endswith('.hpmca') or f.endswith('.chi') or f.endswith('.mca') or f.endswith('.xy') or f.endswith('.nxs') or f[-3:].isnumeric() :
                        file = os.path.join(folder, f) 
                        paths.append(file)  
                        #files_filtered.append(f)
                filenames = paths
        else:
            if 'paths' in kwargs:
                paths = kwargs['paths']
                if len(paths) < 1:
                    return
                
                folder = os.path.split( paths[0])[0]
                filenames = paths
        
            else:
                filenames = []

        if 'progress_dialog' in kwargs:
            progress_dialog = kwargs['progress_dialog']
        else:
            progress_dialog = QtWidgets.QProgressDialog()

        if 'replace' in kwargs:
            replace = kwargs['replace']
        else:
            replace = True
            

        if len(filenames):
            if len(filenames) > 15:
                progress_dialog.setMaximum(len(filenames))
                progress_dialog.show()
            firstfile = filenames[0]
            if firstfile.endswith('.hpmca')or firstfile[-3:].isnumeric():
                [r, success] =self.read_ascii_files_2d(filenames, progress_dialog=progress_dialog)
            if firstfile.endswith('.nxs'):
                [r, success] =self.read_nxs_files_2d(filenames, progress_dialog=progress_dialog)
            elif firstfile.endswith('.chi') or firstfile.endswith('.xy'):
                [r, success] = self.read_chi_files_2d(filenames, progress_dialog=progress_dialog)
                wavelength = r['calibration'][0].wavelength
                self.wavelength = wavelength
            elif  firstfile.endswith('.mca'):
                print('mca 2d not implemented')

            progress_dialog.close()
        
        if success == True:

            if replace:
                self.file_name=folder
                self.data = r['data']
                self.nchans = len(r['data'][0])
                self.n_detectors=r['n_detectors']

                calibration = self.calibration_persistent
                if len(calibration) == self.n_detectors:
                    for i, cal in enumerate(calibration):
                        self.set_calibration(cal, i)
                else:
                    self.clear_persistent_calibration()
                    self.calibration = r['calibration']

                self.rois_from_file = r['rois']
                self.rois = r['rois']
                self.elapsed = r['elapsed']
                
                self.environment = r['environment']
                self.name = os.path.split(folder)[-1]
                self.dx_type = r['dx_type']

                self.files_loaded = r['files_loaded']
            
            else:
                #self.file_name=folder
                self.data = np.concatenate((self.data, r['data']), axis=0) 
                #self.nchans = len(r['data'][0])
                self.n_detectors += r['n_detectors']

                calibration = self.calibration_persistent
                if len(calibration) == self.n_detectors:
                    for i, cal in enumerate(calibration):
                        self.set_calibration(cal, i)
                else:
                    self.clear_persistent_calibration()
                    self.calibration +=  r['calibration']

                self.rois_from_file += r['rois']
                self.rois += r['rois']
                self.elapsed += r['elapsed']
                
                #self.environment += r['environment']
                #self.name = os.path.split(folder)[-1]
                #self.dx_type = r['dx_type']

                self.files_loaded += r['files_loaded']
    
        print('added')
        return([paths,success])

        
    def read_ascii_files_2d(self, paths, *args, **kwargs):      
        """
        Reads multiple disk files.  The file format is a tagged ASCII format.
        The file contains the information from the Mca object which it makes sense
        to store permanently, but does not contain all of the internal state
        information for the Mca.  This procedure reads files written with
        write_ascii_file().
        This mehtod can be several times faster than reading individual mca files.

        Inputs:
            paths:
                The names of the disk files to read.
                
        Outputs:
            Returns a dictionary of the following type
                'files_loaded':     files_loaded
                'start_times' :     start_times
                'data'        :     mca counts in a form of a 2-D numpy array 
                                    (dimension 1: file index
                                     dimension 2: channel index) 
            
        Example:
            m = read_ascii_files_2d(['mca.hpmca'])
            
        """
        success = False
        if 'progress_dialog' in kwargs:
            progress_dialog = kwargs['progress_dialog']
        else:
            progress_dialog = QtWidgets.QProgressDialog()

        paths = paths [:self.max_spectra]
        nfiles = len (paths)   
        
        files_loaded = []
        environment = []
        calibration = [McaCalibration()]
        elapsed = [McaElapsed()]
        rois = [[]]
        max_rois = 12
        times = []
        n_detectors = nfiles
        
        for det in range(1, n_detectors):
            elapsed.append(McaElapsed())
            calibration.append(McaCalibration())
            rois.append([])
        
        nchans = self.nchans
        QtWidgets.QApplication.processEvents()
        for d, file in enumerate(paths):
            
            if d % 5 == 0:
                #update progress bar only every 5 files to save time
                progress_dialog.setValue(d)
                QtWidgets.QApplication.processEvents()
            try:
                fp = open(file, 'r')
            except:
                continue
            line = ''
            e = 0
            while(1):
                line = fp.readline()
                if (line == ''): break
                pos = line.find(' ')
                if (pos == -1): pos = len(line)
                tag = line[0:pos]
                value = line[pos:].strip()
                values = value.split()
                if (tag == 'VERSION:'):
                    pass
                elif (tag == 'ELEMENTS:'):
                    pass
                elif (tag == 'DATE:'):  
                    start_time = value
                    times.append(start_time)
                elif (tag == 'CHANNELS:'):
                    if d == 0:

                        nchans = int(value)
                        self.nchans = nchans
                        data = np.zeros([nfiles, self.nchans])
                elif (tag == 'ROIS:'):
                    nrois = int(values[0])
                    
                    for r in range(nrois):
                        rois[d].append(McaROI())
                elif (tag == 'REAL_TIME:'):
                    elapsed[d].start_time = start_time
                    elapsed[d].real_time = float(values[0])
                elif (tag == 'LIVE_TIME:'):  
                    elapsed[d].live_time = float(values[0])
                elif (tag == 'CAL_OFFSET:'):
                    calibration[d].offset = float(values[0])
                elif (tag == 'CAL_SLOPE:'):
                    calibration[d].slope = float(values[0])
                elif (tag == 'CAL_QUAD:'):  
                    calibration[d].quad = float(values[0])
                elif (tag == 'TWO_THETA:'):
                    calibration[d].two_theta = float(values[0])
                    calibration[d].set_dx_type('edx')
                    calibration[d].units = "keV"
                    data_type = int
                    dx_type = 'edx'
                elif (tag == 'WAVELENGTH:'):
                    calibration[d].wavelength = float(values[d])
                    calibration[d].set_dx_type('adx')
                    calibration[d].units = 'degrees'
                    data_type = float
                    dx_type = 'adx'
                elif (tag == 'ENVIRONMENT:'):
                    if d == 0:
                        env = McaEnvironment()
                        p1 = value.find('=')
                        env.name = value[0:p1]
                        p2 = value[p1+2:].find('"')
                        env.value = []
                        for i in range(n_detectors):
                            env.value.append('')
                        env.description = value[p1+2+p2+3:-1]
                        environment.append(env)
                    env = environment[e]
                    p1 = value.find('=')
                    p2 = value[p1+2:].find('"')
                    val= value[p1+2: p1+2+p2]
                    env.value[d]= val
                    e = e +1
                elif (tag == 'DATA:'):
                    for chan in range(nchans):
                        line = fp.readline()
                        counts = line.split()
                        data[d][chan]=data_type(counts[0])
                else:
                    for i in range(max_rois):
                        roi = 'ROI_'+str(i)+'_'
                        if (tag == roi+'LEFT:'):
                            if (i < nrois):
                                rois[d][i].left = int(values[0])
                                #break
                        elif (tag == roi+'RIGHT:'):
                            if (i < nrois):
                                rois[d][i].right = int(values[0])
                                #break
                        elif (tag == roi+'LABEL:'):
                            labels = value.split('&')
                            if (i < nrois):
                                rois[d][i].label = labels[0].strip()

            files_loaded.append(os.path.normpath(file))
                
            fp.close()
            if progress_dialog.wasCanceled():
                break
        QtWidgets.QApplication.processEvents()
       
        # Built dictionary to return
        r = {}
        r['n_detectors'] = n_detectors
        r['calibration'] = calibration
        r['elapsed'] = elapsed
        r['rois'] = rois
        r['data'] = data
        r['environment'] = environment
        r['dx_type'] = dx_type
        r['files_loaded'] = files_loaded

        
        success = True
        
        return [r, success]

    def read_chi_files_2d(self, paths, *args, **kwargs):
        #fit2d or dioptas chi type file

        if 'progress_dialog' in kwargs:
            progress_dialog = kwargs['progress_dialog']
        else:
            progress_dialog = QtWidgets.QProgressDialog()

        if 'wavelength' in kwargs:
            wavelength = kwargs['wavelength']
        else:
            wavelength = None

        '''if wavelength == None:
            basefile=os.path.basename(paths[0])
            wavelength = xyPatternParametersDialog.showDialog(basefile,'wavelength',.4)
        '''

        paths = paths [:self.max_spectra]
        nfiles = len (paths)   

        nchans = self.find_chi_file_nelements(paths[0])
        data = np.zeros([nfiles, nchans])
        n_detectors = nfiles
        files_loaded = []
        times = []
        self.nchans = nchans
        QtWidgets.QApplication.processEvents()
        for d, file in enumerate(paths):
            if d % 5 == 0:
                #update progress bar only every 5 files to save time
                progress_dialog.setValue(d)
                QtWidgets.QApplication.processEvents()
        
            file_text = open(file, "r")

            a = True
            row = 0
            while a:
                file_line = file_text.readline()
                if not file_line.startswith ('#'):
                    columns = file_line.split()
                    if len(columns):
                        data[d][row]=float(columns[1])
                    row +=1
                if row >= nchans-1:
                    a = False
            files_loaded.append(os.path.normpath(file))
            file_text.close()
          
            
            if progress_dialog.wasCanceled():
                break
        QtWidgets.QApplication.processEvents()
        # Built dictionary to return

        calibration = []
        elapsed = []
        rois = []
        environment = []
        skiprows = 4
        file1 = np.loadtxt(paths[0], skiprows=skiprows)
        x = file1.T[0]
        coeffs = self.compute_tth_calibration_coefficients(x)
        for n in range(n_detectors):
            cal = McaCalibration(offset=coeffs[0],
                                               slope=coeffs[1],
                                               quad=0, 
                                               two_theta= np.mean(x),
                                               units='degrees',
                                               wavelength=wavelength)
            cal.set_dx_type('adx')
            calibration.append(cal)
            elapsed.append(McaElapsed())
            rois.append([])
        r = {}
        r['n_detectors'] = n_detectors
        r['calibration'] = calibration
        r['elapsed'] = elapsed
        r['rois'] = rois
        r['data'] = data
        r['environment'] = environment
        r['wavelength'] = wavelength
        r['dx_type'] = 'adx'

        r['files_loaded'] = files_loaded
        success = True
        return [r, success]

    def find_chi_file_nelements(self, file):
        file_text = open(file, "r")
        a = True
        comment_rows = 0
        first_data_line = 0
        line_n = 0
        while a:
            file_line = file_text.readline()
            
            if not file_line:
                #print("End Of File")
                a = False
            else:
                if file_line.startswith("#"):
                    comment_rows +=1
                else:
                    if first_data_line == 0 :      
                        if  file_line.split()[0].isdigit():
                            first_data_line = line_n
            line_n +=1
        nelem = line_n - first_data_line
        return nelem

    def compute_tth_calibration_coefficients(self, tth):
        chan = np.linspace(0,len(tth)-1,len(tth))[::50]
        tth = tth[::50]
        weights = np.ones(len(tth)) 
        coeffs = CARSMath.polyfitw(chan, tth, weights, 1)
        return coeffs


    def read_nxs_files_2d(self, paths, *args, **kwargs):
        #fit2d or dioptas chi type file

        if 'progress_dialog' in kwargs:
            progress_dialog = kwargs['progress_dialog']
        else:
            progress_dialog = QtWidgets.QProgressDialog()

        if 'wavelength' in kwargs:
            wavelength = kwargs['wavelength']
        else:
            wavelength = None

        '''if wavelength == None:
            basefile=os.path.basename(paths[0])
            wavelength = xyPatternParametersDialog.showDialog(basefile,'wavelength',.4)
        '''

        paths = paths [:self.max_spectra]
        nfiles = len (paths)   

        det_to_read = '0'
        two_theta= [5.00588, 2.99675]
        calibrations =          [McaCalibration(offset=9.261257763597*1e-3,
                                                slope=40.145783749552*1e-3,
                                                quad=-0.000002461285*1e-3, 
                                                two_theta= two_theta[0],
                                                units='keV',
                                                wavelength=None),
                                McaCalibration(offset=10.428138841333*1e-3,
                                                slope=40.171133274681*1e-3,
                                                quad=-0.000011833096*1e-3, 
                                                two_theta= two_theta[0],
                                                units='keV',
                                                wavelength=None)]
        first_data = read_nxs(paths[0],det_to_read)

        nchans = len(first_data[0])
        data = np.zeros([nfiles, nchans])
        n_detectors = nfiles
        files_loaded = []
        times = []
        self.nchans = nchans
        QtWidgets.QApplication.processEvents()
        for d, file in enumerate(paths):
            if d % 5 == 0:
                #update progress bar only every 5 files to save time
                progress_dialog.setValue(d)
                QtWidgets.QApplication.processEvents()
    
            d_file_data = read_nxs(file,det_to_read)
            data[d][:]=d_file_data[0][:]
           
            files_loaded.append(os.path.normpath(file))
           
            
            if progress_dialog.wasCanceled():
                break
        QtWidgets.QApplication.processEvents()
        # Built dictionary to return

        calibration = []
        elapsed = []
        rois = []
        environment = []
        skiprows = 4
        
        for n in range(n_detectors):
            cal = calibrations[0]
            cal.set_dx_type('edx')
            calibration.append(cal)
            elapsed.append(McaElapsed())
            rois.append([])
        r = {}
        r['n_detectors'] = n_detectors
        r['calibration'] = calibration
        r['elapsed'] = elapsed
        r['rois'] = rois
        r['data'] = data
        r['environment'] = environment
        r['wavelength'] = wavelength
        r['dx_type'] = 'adx'

        r['files_loaded'] = files_loaded
        success = True
        return [r, success]