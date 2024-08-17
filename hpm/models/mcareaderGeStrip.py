from .mcaComponents import McaCalibration, McaElapsed, McaROI, McaEnvironment
from PyQt6 import QtCore, QtWidgets
import numpy as np

def read_mca_header( file):
    
    file_text = open(file, "r")

    a = True
    comment_rows = 0
    first_data_line = 0
    line_n = 0
    nelem = [0,0]
    while a:
        file_line = file_text.readline().strip()
        
        if not file_line:
            #print("End Of File")
            a = False
        else:
            if file_line.startswith("#"):
                par, val = parse_mca_header_line(file_line)
                if par == 'rows':
                    nelem[0] = int(val)
                elif par == 'columns':
                    nelem[1] = int(val)
                comment_rows +=1
            else:
                a = False
    first_data_line = comment_rows
    return nelem, first_data_line

def parse_mca_header_line( line):
    tokens = line.split(':')
    par = tokens[0].strip()[1:]
    val = tokens[1].strip()
    return par, val


def test_read_mca_file_type1(file):
    nelem, first_data_line = read_mca_header(file)
    type1 = len(nelem)>0
    return type1
    
def read_mca_file_type1(path, *args, **kwargs):
    """
    Reads a single multispectral mca file from the Ge strip detector.
        The file format is a tagged ASCII format.
    The file contains the information about the number of rows and columns in the header.
    

    Inputs:
        paths:
            List, containing the name of the disk file to read. Still expects a 
            list as an imput even though it will only read the first file
            
    Outputs:
        Returns a dictionary of the following type
           
            'data'        :     mca counts in a form of a 2-D numpy array 
                                (dimension 1: spectrum index
                                    dimension 2: channel index) 
        
    Example:
        m = read_mca_ascii_file_2d(['file.mca'])  
        
    """
    file = path
    r = {}
    success = False
    correct_type = test_read_mca_file_type1(file)
    if correct_type:

        nelem, first_data_line = read_mca_header(path)

        if 'progress_dialog' in kwargs:
            progress_dialog = kwargs['progress_dialog']
        else:
            progress_dialog = QtWidgets.QProgressDialog()

        #paths = paths [:self.max_spectra]
        progress_dialog.setMaximum(nelem[0])
        data = np.zeros([nelem[0], nelem[1]])
        files_loaded = path
        times = []
        calibration = [McaCalibration()]
        elapsed = [McaElapsed()]
        rois = [[]]
        
        n_detectors= nelem[0]
        if n_detectors > 15:
            progress_dialog.setMaximum(n_detectors)
            progress_dialog.show()
        for det in range(1, n_detectors):
            elapsed.append(McaElapsed())
            calibration.append(McaCalibration())
            rois.append([])
        QtWidgets.QApplication.processEvents()
        fp = open(path, 'r')
        for h in range(first_data_line):
            line = fp.readline()

        for d in range(nelem[0]):
            if d % 5 == 0:
                #update progress bar only every 5 files to save time
                progress_dialog.setValue(d)
                QtWidgets.QApplication.processEvents()
        
            line = fp.readline()
            
            counts = line.split('  ')[:-2]
            for chan, count in enumerate(counts):
                    
                data[d][chan]=int(count)
            
            if progress_dialog.wasCanceled():
                break
        fp.close()
        progress_dialog.close()
        QtWidgets.QApplication.processEvents()
        # Built dictionary to return
        r = {}
        r['n_detectors'] = n_detectors
        r['calibration'] = calibration
        r['elapsed'] = elapsed
        r['rois'] = rois
        r['data'] = data
        r['environment'] = []
        r['dx_type'] = 'edx'

        
        success = True
    return [r, success]
