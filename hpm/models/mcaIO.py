import numpy as np
import utilities.CARSMath as CARSMath
from math import sqrt, sin, pi
from hpm.models.mcareaderGeStrip import *
from hpm.widgets.UtilityWidgets import xyPatternParametersDialog
from hpm.models.mcareader import McaReader
from hpm.models.nxsexport_batch import read_nxs

import os

########################################################################
class mcaFileIO():
    def __init__(self):
        pass


        
    def read_mca_file (self, file, tth=15):  #amptek type file

        
        r = {}
        loaded = False
        mca_type = -1
        test_1 = test_read_mca_file_type1(file)
        if test_1:
            mca_type = 1
        else:
            test_0 = self.test_read_mca_file_type0(file)
            if test_0:
                mca_type = 0
        
        if mca_type == 0:
            [r, loaded] = self.read_mca_file_type0(file, tth)
        elif mca_type == 1:
            [r, loaded] = read_mca_file_type1(file, tth)
            
        return [r, loaded]

        
        
    def test_read_mca_file_type0(self, file):
        mcafile = McaReader(file)
        
        ans = mcafile.get_live_time()
        type_0 = ans != None
        return type_0

    def read_mca_file_type0 (self, file, tth=15):  #amptek type file

        mcafile = McaReader(file)
        elapsed = McaElapsed()
        r = {}
        loaded = False
        

        elapsed.live_time = mcafile.get_live_time()
        elapsed.real_time = mcafile.get_real_time()
        elapsed.start_time = mcafile.get_start_time()
        calibration = McaCalibration(dx_type='edx')
        
        cal_func = mcafile.get_calibration_function()
        if cal_func is not None:
            calibration.offset, calibration.slope = cal_func
        file_rois = mcafile.get_rois()
        rois = []
        for r in file_rois:
            roi = McaROI()
            roi.left = int(r[0])
            roi.right = int(r[1])
            rois.append(roi)
        data = mcafile.get_data()
        basefile=os.path.basename(file)
        #tth = xyPatternParametersDialog.showDialog(basefile,'tth',15)
        calibration.two_theta= tth
        
        r['n_detectors'] = 1
        r['calibration'] = [calibration]
        r['elapsed'] = [elapsed]
        r['rois'] = [rois]
        r['data'] = [data]
        r['environment'] = []
        r['dx_type'] = 'edx'
        loaded = True
        return [r, loaded]
        #mcafile.plot()

 

    def read_ascii_file(self, file, *args, **kwargs):
        """
        Reads a disk file.  The file format is a tagged ASCII format.
        The file contains the information from the Mca object which it makes sense
        to store permanently, but does not contain all of the internal state
        information for the Mca.  This procedure reads files written with
        write_ascii_file().

        Inputs:
            file:
                The name of the disk file to read.
                
        Outputs:
            Returns a dictionary of the following type:
            'n_detectors': int,
            'calibration': [McaCalibration()],
            'elapsed':     [McaElapsed()],
            'rois':        [[McaROI()]]
            'data':        [Numeric.array]
            'environment': [[McaEnvironment()]]
            
        Example:
            m = read_ascii_file('mca.001')
            m['elapsed'][0].real_time

        Modification by RH Dec. 30 2021
        Version 3.1A
        Added a distionction between EDX and ADX files.
        For ADX files a WAVELENGTH field is written rather than TWO_THETA.
        For ADX data is written as float, for EDX the as int.
        """
        try:
            fp = open(file, 'r')
        except:
            return [None, False]

        if 'progress_dialog' in kwargs:
            progress_dialog = kwargs['progress_dialog']
        else:
            progress_dialog = QtWidgets.QProgressDialog()

        line = ''
        start_time = ''
        max_rois = 0
        data = None
        
        environment = []
        n_detectors = 1  # Assume single element data
        elapsed = [McaElapsed()]
        calibration = [McaCalibration()]
        rois = [[]]
        dx_type = ''
        data_type = int
        try:
            
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
                elif (tag == 'DATE:'):  
                    start_time = value
                elif (tag == 'ELEMENTS:'):
                    n_detectors  = int(value)
                    
                    for det in range(1, n_detectors):
                        elapsed.append(McaElapsed())
                        calibration.append(McaCalibration())
                        rois.append([])
                elif (tag == 'CHANNELS:'):
                    
                    nchans = int(value)
                    if n_detectors > 15:
                        progress_dialog.setMaximum(nchans)
                        progress_dialog.show()
                elif (tag == 'ROIS:'):
                    nrois = []
                    for d in range(n_detectors):
                        nrois.append(int(values[d]))
                    max_rois = max(nrois)
                    for d in range(n_detectors):
                        for r in range(nrois[d]):
                            rois[d].append(McaROI())
                elif (tag == 'REAL_TIME:'):
                    for d in range(n_detectors):
                        elapsed[d].start_time = start_time
                        elapsed[d].real_time = float(values[d])
                elif (tag == 'LIVE_TIME:'):  
                    for d in range(n_detectors):
                        elapsed[d].live_time = float(values[d])
                elif (tag == 'CAL_OFFSET:'):
                    for d in range(n_detectors):
                        calibration[d].offset = float(values[d])
                elif (tag == 'CAL_SLOPE:'):
                    for d in range(n_detectors):
                        calibration[d].slope = float(values[d])
                elif (tag == 'CAL_QUAD:'):  
                    for d in range(n_detectors):
                        calibration[d].quad = float(values[d])
                elif (tag == 'TWO_THETA:'):
                    for d in range(n_detectors):
                        calibration[d].two_theta = float(values[d])
                        calibration[d].set_dx_type('edx')
                        calibration[d].units = 'keV'
                    data_type = int
                    dx_type = 'edx'
                elif (tag == 'WAVELENGTH:'):
                    for d in range(n_detectors):
                        calibration[d].wavelength = float(values[d])
                        calibration[d].set_dx_type('adx')
                        calibration[d].units = 'degrees'
                    data_type = float
                    dx_type = 'adx'
                elif (tag == 'ENVIRONMENT:'):
                    env = McaEnvironment()
                    p1 = value.find('=')
                    env.name = value[0:p1]
                    p2 = value[p1+2:].find('"')
                    env.value = value[p1+2: p1+2+p2]
                    env.description = value[p1+2+p2+3:-1]
                    environment.append(env)
                elif (tag == 'DATA:'):
                    
                    data = []
                    for d in range(n_detectors):
                        data.append(np.zeros(nchans,  dtype=data_type))
                    for chan in range(nchans):
                        if chan % 20 == 0:
                            #update progress bar only every 5 files to save time
                            progress_dialog.setValue(chan)
                            QtWidgets.QApplication.processEvents()
                        line = fp.readline()
                        counts = line.split()
                        for d in range(n_detectors):
                            data[d][chan]=data_type(counts[d])
                    for d in range(n_detectors):    
                        if calibration[d].dx_type == '':
                            calibration[d].set_dx_type('edx')
                else:
                    for i in range(max_rois):
                        roi = 'ROI_'+str(i)+'_'
                        if (tag == roi+'LEFT:'):
                            for d in range(n_detectors):
                                if (i < nrois[d]):
                                    rois[d][i].left = int(values[d])
                                #break
                        elif (tag == roi+'RIGHT:'):
                            for d in range(n_detectors):
                                if (i < nrois[d]):
                                    rois[d][i].right = int(values[d])
                                #break
                        elif (tag == roi+'LABEL:'):
                            labels = value.split('&')
                            for d in range(n_detectors):
                                if (i < nrois[d]):
                                    rois[d][i].label = labels[d].strip()
                                #break
                        else:
                           
                            pass
            
            # Make sure DATA array is defined, else this was not a valid data file
            if (data == None): return [None, False]
            fp.close()
            # Built dictionary to return
            r = {}
            r['n_detectors'] = n_detectors
            r['calibration'] = calibration
            r['elapsed'] = elapsed
            r['rois'] = rois
            r['data'] = data
            r['environment'] = environment
            r['dx_type'] = dx_type
            
            return [r, True]
        except:
            return [None, False]

    def read_ascii_file_calibration(self, file):
        """
        Reads calibration from a disk file.  The file format is a tagged ASCII format.
        The file contains the information from the Mca object which it makes sense
        to store permanently, but does not contain all of the internal state
        information for the Mca.  This procedure reads files written with
        write_ascii_file().

        Inputs:
            file:
                The name of the disk file to read.
                
        Outputs:
            Returns a dictionary of the following type:
            'n_detectors': int,
            'calibration': [McaCalibration()]]
            
        Example:
            m = read_ascii_file('mca.001')
            

        Modification by RH Dec. 30 2021
        Version 3.1A
        Added a distionction between EDX and ADX files.
        For ADX files a WAVELENGTH field is written rather than TWO_THETA.
        For ADX data is written as float, for EDX the as int.
        """
        try:
            fp = open(file, 'r')
        except:
            return [None, False]
        line = ''

        data = None
        
        environment = []
        n_detectors = 1  # Assume single element data
        elapsed = [McaElapsed()]
        calibration = [McaCalibration()]
        rois = [[]]
        dx_type = ''
        data_type = int
        
        try:
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
                elif (tag == 'DATE:'):  
                    start_time = value
                elif (tag == 'ELEMENTS:'):
                    n_detectors  = int(value)
                    for det in range(1, n_detectors):
                        elapsed.append(McaElapsed())
                        calibration.append(McaCalibration())
                        rois.append([])
                elif (tag == 'CAL_OFFSET:'):
                    for d in range(n_detectors):
                        calibration[d].offset = float(values[d])
                elif (tag == 'CAL_SLOPE:'):
                    for d in range(n_detectors):
                        calibration[d].slope = float(values[d])
                elif (tag == 'CAL_QUAD:'):  
                    for d in range(n_detectors):
                        calibration[d].quad = float(values[d])
                        
                elif (tag == 'TWO_THETA:'):
                    for d in range(n_detectors):
                        calibration[d].two_theta = float(values[d])
                        calibration[d].set_dx_type('edx')
                        calibration[d].units = 'keV'
                    dx_type = 'edx'
                elif (tag == 'WAVELENGTH:'):
                    for d in range(n_detectors):
                        calibration[d].wavelength = float(values[d])
                        calibration[d].set_dx_type('adx')
                        calibration[d].units = 'degrees'
                    dx_type = 'adx'
                elif (tag == 'DATA:'):
                    
                    for d in range(n_detectors):    
                        if calibration[d].dx_type == '':
                            calibration[d].set_dx_type('edx')
                
                else:
                    pass

        except:
            return [None, False]

        if dx_type == '':
            dx_type = 'edx'
            for d in range(len(calibration)):
                calibration[d].set_dx_type('edx')
                calibration[d].units = 'keV'
        
        # Make sure DATA array is defined, else this was not a valid data file
        #if (data == None): return [None, False]
        fp.close()
        # Built dictionary to return
        r = {}
        r['n_detectors'] = n_detectors
        r['calibration'] = calibration
        r['elapsed'] = elapsed
        r['rois'] = rois
        r['data'] = data
        r['environment'] = environment
        r['dx_type'] = dx_type
        
        return [r, True]
        

    def compute_tth_calibration_coefficients(self, tth):
        chan = np.linspace(0,len(tth)-1,len(tth))[::50]
        tth = tth[::50]
        weights = np.ones(len(tth)) 
        coeffs = CARSMath.polyfitw(chan, tth, weights, 1)
        return coeffs
        

    def read_chi_file(self, filename, wavelength=None):  #fit2d or dioptas chi type file
        if filename.endswith('.chi') or filename.endswith('.xy'):
            '''fp = open(filename, 'r')
            first_line = fp.readline()
            second_line = fp.readline()
            unit = second_line.strip().upper()[:1]  # reserved for future functionality
            fp.close()'''
            
            skiprows = 4
            data = np.loadtxt(filename, skiprows=skiprows)
            
            x = data.T[0]
            y = data.T[1]
            basefile=os.path.basename(filename)
            #name = basefile.split('.')[:-1][0]


            coeffs = self.compute_tth_calibration_coefficients(x)
            '''if wavelength == None:

                wavelength = xyPatternParametersDialog.showDialog(basefile,'wavelength',.4)'''

            r = {}
            r['n_detectors'] = 1
            r['calibration'] = [McaCalibration(offset=coeffs[0],
                                               slope=coeffs[1],
                                               quad=0, 
                                               two_theta= None,
                                               units='degrees',
                                               wavelength=wavelength)]
            r['calibration'][0].set_dx_type('adx')
            r['elapsed'] = [McaElapsed()]
            r['rois'] = [[]]
            r['data'] = [y]
            r['environment'] = []
            r['dx_type'] = 'adx'
            return[r,True]
            
        return [None, False]
    

    ######################################################################

    def read_nsx_file(self, filename, two_theta= [5.00588, 2.99675]):
        ok = False
        r = {}
        data = read_nxs(filename, 'both')
        if len(data):
            
            r['n_detectors'] = 1
            r['calibration'] = [McaCalibration(offset=9.261257763597*1e-3,
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
            r['calibration'][0].set_dx_type('edx')
            r['calibration'][1].set_dx_type('edx')
            r['elapsed'] = [McaElapsed(),McaElapsed()]
            r['rois'] = [[],[]]
            r['data'] = [data['1'], data['2']]
            r['environment'] = []
            r['dx_type'] = 'edx'
            ok = True

        return[r,ok]
        
    #######################################################################
    def write_ascii_file(self, file, data, calibration, elapsed, presets, rois,
                        environment):
        """
        Writes Mca or Med data to a disk file.  The file 
        format is a tagged ASCII format.  The file contains the information 
        from the Mca object which it makes sense to store permanently, but 
        does not contain all of the internal state information for the Mca.  
        Files written with this routine can be read with read_ascii_file(), which
        is called by Mca.read_file() if the netcdf flag is 0.

        This procedure is typically not called directly, but is called
        by Mca.write_file if the netcdf=1 keyword is not used.

        This function can be used for writing for Mca objects, in which case
        each input parameter is an object, such as McaElapsed, etc.
        It can also be used for writing Med objects, in which case each input
        parameter is a list.

        If the rank of data is 2 then this is an Med, and the number of detectors
        is the first dimension of data

        Inputs:
            file:
                The name of the disk file to write.
                
            data:
                The data to write.  Either 1-D array or list of 1-D arrays.

            calibration:
                An object of type McaCalibration, or a list of such objects.

            elapsed:
                An object of type McaElapsed, or a list of such objects.

            presets:
                An object of type McaPresets, or a list of such objects.

            rois:
                A list of McaROI objects, or a list of lists of such objects.
            
            environment:
                A list of McaEnvironment objects, or a list of lists of such objects.
        
        Modification by RH Dec. 30 2021
        Version 3.1A
        Added a distionction between EDX and ADX files.
        For ADX files a WAVELENGTH field is written rather than TWO_THETA.
        For ADX data is written as float, for EDX the as int.
        """



        if (np.ndim(data) == 2):
            n_det = len(data)
        else:
            n_det = 1
        fformat = '%f ' * n_det
        eformat = '%e ' * n_det
        iformat = '%d ' * n_det
        sformat = '%s ' * n_det
        data_format =iformat
        
        nchans = len(data[0])
        dx_type = calibration[0].dx_type
        version = '3.1A' if dx_type == 'adx'  else '3.1'
        start_time = elapsed[0].start_time

        fp = open(file, 'w')
        fp.write('VERSION:    '+version+'\n')
        fp.write('ELEMENTS:   '+str(n_det)+'\n')
        fp.write('DATE:       '+str(start_time)+'\n')
        fp.write('CHANNELS:   '+str(nchans)+'\n')

        nrois = []
        for roi in rois:
            nrois.append(len(roi))
        fp.write('ROIS:       '+(iformat % tuple(nrois))+'\n')
        real_time=[]; live_time=[]
        for e in elapsed:
            real_time.append(e.real_time)
            live_time.append(e.live_time)
        fp.write('REAL_TIME:  '+(fformat % tuple(real_time))+'\n')
        fp.write('LIVE_TIME:  '+(fformat % tuple(live_time))+'\n')
        offset=[]; slope=[]; quad=[]; two_theta=[]; wavelength=[]
        for c in calibration:
            offset.append(c.offset)
            slope.append(c.slope)
            quad.append(c.quad)
            if c.dx_type == 'edx':
                two_theta.append(c.two_theta)
            if c.dx_type == 'adx':
                wavelength.append(c.wavelength)
        fp.write('CAL_OFFSET: '+(eformat % tuple(offset))+'\n')
        fp.write('CAL_SLOPE: '+(eformat % tuple(slope))+'\n')
        fp.write('CAL_QUAD: '+(eformat % tuple(quad))+'\n')
        if c.dx_type == 'edx':
            res = any(ele == None for ele in two_theta)
            if not res:
                fp.write('TWO_THETA: '+(fformat % tuple(two_theta))+'\n')
            data_format = iformat
        if c.dx_type == 'adx':
            fp.write('WAVELENGTH: '+(fformat % tuple(wavelength))+'\n')
            data_format = fformat

        for i in range(max(nrois)):
            num = str(i)
            left=[]; right=[]; label=[]
            for d in range(n_det):
                if (i < nrois[d]):
                    left.append(rois[d][i].left)
                    right.append(rois[d][i].right)
                    label.append(rois[d][i].label + '&')
                else:
                    left.append(0)
                    right.append(0)
                    label.append(' &')
            fp.write('ROI_'+num+'_LEFT:  '+(iformat % tuple(left))+'\n')
            fp.write('ROI_'+num+'_RIGHT:  '+(iformat % tuple(right))+'\n')
            fp.write('ROI_'+num+'_LABEL:  '+(sformat % tuple(label))+'\n')
        for e in environment:
            fp.write('ENVIRONMENT: '       + str(e.name) +
                                    '="'  + str(e.value) +
                                    '" (' + str(e.description) + ')\n')
        fp.write('DATA: \n')
        counts = np.zeros(n_det)
        for i in range(nchans):
            for d in range(n_det):
                counts[d]=data[d][i]
            fp.write((data_format % tuple(counts))+'\n')
        fp.close()


    def write_pattern(self, filename, x, y, unit, unit_, header=None):
        """
        Saves the x, y data pattern.
        :param filename: where to save
        :param header: you can specify any specific header
        """
        
        file_handle = open(filename, 'w')
        num_points = len(x)

        if filename.endswith('.chi'):
            if header is None or header == '':
                file_handle.write(filename + '\n')
                file_handle.write(unit + ' ('+ unit_+')''\n\n')
                file_handle.write("       {0}\n".format(num_points))
            else:
                file_handle.write(header)
            for ind in range(num_points):
                file_handle.write(' {0:.7E}  {1:.7E}\n'.format(x[ind], y[ind]))
        elif filename.endswith('.fxye'):
            factor = 100
            if header is not None:
                if 'CONQ' in header:
                    factor = 1
                header = header.replace('NUM_POINTS', '{0:.6g}'.format(num_points))
                header = header.replace('MIN_X_VAL', '{0:.6g}'.format(factor*x[0]))
                header = header.replace('STEP_X_VAL', '{0:.6g}'.format(factor*(x[1]-x[0])))

                file_handle.write(header)
                file_handle.write('\n')
            for ind in range(num_points):
                file_handle.write('\t{0:.6g}\t{1:.6g}\t{2:.6g}\n'.format(factor*x[ind], y[ind], sqrt(abs(y[ind]))))
        else:
            if header is not None:
                file_handle.write(header)
                file_handle.write('\n')
            for ind in range(num_points):
                file_handle.write('{0:.9E}  {1:.9E}\n'.format(x[ind], y[ind]))
        file_handle.close()

    def write_peaks(self, file, rois, background=None):
        """
        Writes a list of obejcts of type McaPeak to a disk file.
        If the background parameter is present it also writes the background
        structure to the file.
        
        Inputs:
            file:
                The name of a disk file to be written ;
            peaks:
                A list of McaPeak objects
                
        Keywords:
            background:
                An object of type McaBackground
                
        Example:
            r = read_peaks('my_peaks.pks')
            peaks = r['peaks']
            peaks[1].initial_energy = 6.4
            write_peaks('mypeaks.pks', peaks)
        """
        lines = []
        if (background != None):
            lines.append('Background_exponent,'     + str(background.exponent)+'\n')
            lines.append('Background_top_width,'    + str(background.top_width)+'\n')
            lines.append('Background_bottom_width,' + str(background.bottom_width)+'\n')
            lines.append('Background_tangent,'      + str(background.tangent)+'\n')
            lines.append('Background_compress,'     + str(background.compress)+'\n')
        lines.append('centroid,'+'fwhm,'+'energy,'+'fwhm_E,'+'q,'+'fwhm_q,'+'d_spacing,'+'fwhm_d,'+'label'+'\n')
        for r in rois:
            lines.append('{:.4e}'.format(r.centroid) + ', ' + 
                        '{:.4e}'.format(r.fwhm)    + ', ' + 
                        '{:.4e}'.format(r.energy) + ', ' + 
                        '{:.4e}'.format(r.fwhm_E)    + ', ' + 
                        '{:.4e}'.format(r.q)   + ', ' + 
                        '{:.4e}'.format(r.fwhm_q)      + ', ' + 
                        '{:.4e}'.format(r.d_spacing)   + ', ' + 
                        '{:.4e}'.format(r.fwhm_d)      + ', ' + 
                        r.label + '\n')
        fp = open(file, 'w')
        fp.writelines(lines)
        fp.close()


