
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

"""
EPICS device-dependent support for the Mca class.  Uses the EPICS mca record.

Author:           Mark Rivers
Created:          Sept. 18, 2002
Revision history: Original version was written in IDL
Sept. 24, 2002  MLR.
    - Changed __init__ to use MCA_ENVIRONMENT environment variable when
        opening environment file.
    - Fixed bug when opening environment file
    - Fixed bug reading description from environment file
    - Fixed bug reading environment PVs
Sept. 25, 2002  MLR.
    - Fixed bug reading environment file
OCT. 30, 2018 Ross Hrubiak
    - Updated to Python 3, implemented monitor handling using QtCore signalling
    
"""

from audioop import mul
from socket import timeout
from epics import caput, caget, PV
#from epics.utils import BYTES2STR
import numpy as np
#from epics.clibs import *
import copy
import time
import utilities.hpMCAutilities as hpUtil
from PyQt6 import QtCore
from PyQt6.QtWidgets import QMessageBox
from hpm.models.mcaModel import *
from utilities.hpMCAutilities import custom_signal

class epicsMCA(MCA):
    """
    Creates a new epicsMca.

    Keywords:
        record_Name:
        The name of the EPICS MCA record for the MCA object being created, without
        a trailing period or field name.
        epics_buttons:

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
    >>> from epicsMca import *
    >>> mca = epicsMca('13IDC:mca1')
    >>> print mca.data
    """
    def __init__(self, *args,  **kwargs):
        
        super().__init__()
        
        
        record_name = kwargs['record_name']
        epics_buttons = kwargs['epics_buttons']
        file_options  = kwargs['file_options']
        environment_file  = kwargs['environment_file']
        dead_time_indicator  = kwargs['dead_time_indicator']

        multielement = False
        element_num = 1
        prefix = record_name
        if ":" in record_name:
            appendix = str.split(record_name, ':')[-1]
            prefix = ":".join(str.split(record_name, ':')[:-1])
            if len(appendix)>3:
                if str.startswith(appendix,'mca'):
                    numeric_appendix = appendix[3:]
                    if str.isnumeric(numeric_appendix):
                        element_num = int(numeric_appendix)
                        if element_num >=0 and element_num < 100:
                            multielement = True
        self.element = element_num
        self.multielement = multielement
        self.prefix = prefix

        self.name = record_name
        self.record_name = record_name

        self.files_loaded = []
        self.last_saved=''
        self.file_settings = file_options
        self.verbose = False
        
        self.mcaRead = None
        [self.btnOn, self.btnOff, self.btnErase] = epics_buttons  
        
        
        self.max_rois = 24           
        self.initOK = False    
                
        self.mcaPV = []
        
        self.n_detectors = 0
        if self.multielement:
            self.max_rois = 8
            while True:
                test_mca_pv_name = self.prefix + ':mca' + str(self.n_detectors+1)
                
                test_mca_pv= PV(test_mca_pv_name)
                test_mca_pv.wait_for_connection(timeout=0.05)
                if test_mca_pv.connected:
                    self.initOK = True
                    self.files_loaded.append(test_mca_pv_name)
                    self.n_detectors += 1
                    self.mcaPV.append(test_mca_pv)
                else:
                    break

        else:
            self.mcaPV.append(PV(self.record_name))
            test = self.mcaPV[0].get()
            if test is not None:
                self.n_detectors = 1
                self.initOK = True

        self.calibration = []
        self.rois  = []
        self.rois_from_det = []
        self.rois_from_file = []
        for det in range(self.n_detectors):
            self.calibration.append([])
            self.rois .append([])
            self.rois_from_det.append([])
            self.rois_from_file.append([])

        if self.initOK:
            self.initOK = False
            #epics related buttons:
            self.btnOn.clicked.connect(self.acqOn)
            self.btnOff.clicked.connect(self.acqOff)
            self.btnErase.clicked.connect(self.acqErase)
            self.epics_widgets = [self.btnOn,self.btnOff,self.btnErase]
            self.toggleEpicsWidgetsEnabled(False)
            if self.verbose:
                start_time = time.time()
            pvs = {'calibration': 
                            {'calo': None,
                            'cals': None,
                            'calq': None,
                            'tth' : None,
                            'egu' : None},
                        'elapsed':
                            {'ertm': None,
                            'eltm': None,
                            'act' : None,
                            'rtim': None,
                            'stim': None},
                        'acquire' : {'strt': None,
                                    'stop': None,
                                    'eras': None,
                                    'acqg': None,
                                    'proc': None,
                                    'erst': None,
                                    'read': None},
                        'data':
                            {'val':  None,
                            'nuse': None,
                            'nmax': None},
                        'presets':
                            {'prtm': None,
                            'pltm': None,
                            'pct':  None,
                            'pctl': None,
                            'pcth': None,
                            'chas': None,
                            'dwel': None,
                            'pscl': None}}
            
            self.pvs = []
            for i in range(self.n_detectors):
                self.pvs.append(copy.deepcopy(pvs))
                for group in self.pvs[i].keys():
                    for pv in self.pvs[i][group].keys():
                        name = self.record_name[:-1]+str(i+1) + '.' + pv.upper()
                        self.pvs[i][group][pv] = PV(name)

            # Construct the names of the PVs for the ROIs

            self.roi_def_pvs=[]
            self.roi_data_pvs=[]
            for det in range(self.n_detectors):
                self.roi_def_pvs.append([])
                self.roi_data_pvs.append([])
                for i in range(self.max_rois):
                    n = 'R'+str(i)
                    r = {n+'lo'  : None,
                        n+'hi'  : None,
                        n+'bg'  : None,
                        n+'nm'  : None}
                    self.roi_def_pvs[det].append(r)
                    r = {n       : None,
                        n+'n'   : None}
                    self.roi_data_pvs[det].append(r)
                for roi in range(self.max_rois):
                    for pv in self.roi_def_pvs[det][roi].keys():
                        name = self.record_name[:-1]+str(det+1) + '.' + pv.upper()
                        self.roi_def_pvs[det][roi][pv] = PV(name)
                    for pv in self.roi_data_pvs[det][roi].keys():
                        name = self.record_name[:-1]+str(det+1) + '.' + pv.upper()
                        self.roi_data_pvs[det][roi][pv] = PV(name)

            # set up the dead-time indicator pvWidget
            self.dt_pv = self.name + '.IDTIM'
            dead_time_indicator.connect(self.dt_pv)
           

            # Construct the names of the PVs for the environment
            self.env_pvs = [] 
            if environment_file != None:
                self.read_environment_file(environment_file)
                for env in self.environment:
                    if len(env):
                        self.env_pvs.append(PV(env.name))

            ## monitors for asynchronous actions
            self.read_done_monitor = epicsMonitor(self.pvs[0]['acquire']['read'], self.handle_mca_callback, autostart=True)
            self.erase_start_monitor = epicsMonitor(self.pvs[0]['acquire']['erst'], self.handle_mca_callback_erase_start, autostart=True) 
            #self.start_monitor = epicsMonitor(self.pvs['acquire']['strt'], self.handle_mca_callback_start, autostart=True) 

            self.acqg_monitor = epicsMonitor(self.pvs[0]['acquire']['acqg'], self.handle_mca_callback_acqg, autostart=True) 

            #self.stop_monitor = epicsMonitor(self.pvs['acquire']['stop'],self.handle_mca_callback_stop, autostart=False) 
            self.end_time_monitor = epicsMonitor(self.pvs[0]['elapsed']['stim'],self.handle_mca_callback_end_time, autostart=False)   

            self.erase_monitor = epicsMonitor(self.pvs[0]['acquire']['eras'], self.handle_mca_callback_erase, autostart=True)  
            '''if self.pvs['acquire']['swhy'].connected:
                self.why_stopped = epicsMonitor(self.pvs['acquire']['swhy'], self.handle_mca_callback_why_stopped, autostart=True)
            '''
            self.live_time_preset_monitor = epicsMonitor(self.pvs[0]['presets']['pltm'], self.handle_mca_callback_pltm, autostart=True)  
            self.real_time_preset_monitor = epicsMonitor(self.pvs[0]['presets']['prtm'], self.handle_mca_callback_prtm, autostart=True)  
            
            # a way to send a signal to the parent controller that new data is ready
            self.dataAcquired = custom_signal(debounce_time=0.8)  
            
            self.acq_stopped = custom_signal(debounce_time=1.2)  

            self.toggleEpicsWidgetsEnabled(True)
            if self.verbose:
                print("init --- %s seconds ---" % (time.time() - start_time))

            self.end_time = ''

            self.initOK = True

    def unload(self):
        self.read_done_monitor.unSetPVmonitor()
        self.erase_start_monitor.unSetPVmonitor()
        #self.start_monitor.unSetPVmonitor()
        #self.stop_monitor.unSetPVmonitor()
        self.erase_monitor.unSetPVmonitor()  
        #self.why_stopped.unSetPVmonitor()
        self.live_time_preset_monitor.unSetPVmonitor()
        self.real_time_preset_monitor.unSetPVmonitor()

        for i in range(self.n_detectors):
            for pv_group in self.pvs:
                p_g = self.pvs[i][pv_group]
                for pv in p_g:
                    p = p_g[pv]
                    p.disconnect()

            for p_g in self.roi_def_pvs[i]:
                for pv in p_g:
                    p = p_g[pv]
                    p.disconnect()

            for p_g in self.roi_data_pvs[i]:
                for pv in p_g:
                    p = p_g[pv]
                    p.disconnect()

    #######################################################################
    #######################################################################
    ### parent widget buttons interaction
    
    def blockSignals(self, state, widgets):
        for btn in self.epics_widgets:
            btn.blockSignals(state)

    def toggleEpicsWidgetsEnabled(self,enabled):
        self.blockSignals(True,self.epics_widgets)
        for btn in self.epics_widgets:
            btn.setEnabled(enabled)
            if not enabled:
                btn.setChecked(False)
        self.blockSignals(False,self.epics_widgets)            

    def set_epics_btns_state(self, state):
        self.blockSignals(True,self.epics_widgets)
        if state == 'on' or state == 1:
            self.btnOn.setEnabled(False)
            self.btnOn.setChecked(True)
            self.btnOff.setEnabled(True)
        if state == 'off' or state == 0:
            self.btnOn.setEnabled(True)
            self.btnOn.setChecked(False)
            self.btnOff.setEnabled(False)
        self.blockSignals(False,self.epics_widgets)

    def epicsStateUpdated(self, state):
        pass
        # this should toggle buttons if acquisitions was changed externally to this app
    

    def acqOn(self):
        self.set_epics_btns_state('on')
        self.acq_status = 'on'
        self.acq_on() 
            
    def acqOff(self):
        self.set_epics_btns_state('off')
        self.acq_status = 'off'
        self.acq_off()
        
    def acqErase(self):
        if self.file_settings is not None:
            if self.file_settings.warn_erase :
                if not self.is_saved():
                    choice = QMessageBox.question(None, 'Confirm Erase',
                                                        "Data hasn't been saved. \nDo you want to erase?",
                                                        QMessageBox.Yes | QMessageBox.No)
                    erase = choice == QMessageBox.Yes   
                else: erase = True
            else: erase = True
        if erase:
            self.acq_erase()     

    def is_saved(self):
        return self.last_saved == self.elapsed[0].start_time

    #######################################################################
    #######################################################################

    def read_environment_file(self, file):
        """
        Reads a file containing the "environment" PVs.  The values and desriptions of these
        PVs are stored in the data files written by Mca.write_file().

        Inputs:
            file:
                The name of the file containing the environment PVs. This is an ASCII file
                with each line containing a process variable name, followed by a
                space and a description field.
        """
        self.environment = []
        try:
            fp = open(file, 'r')
            lines = fp.readlines()
            fp.close()
        except:
            return
        for line in lines:
            env = McaEnvironment()
            pos = line.find(' ')
            if (pos != -1):
                env.name = line[0:pos]
                env.description = line[pos+1:].strip()
            else:
                env.name = line.strip()
                env.description = ' '
            self.environment.append(env)

    def get_environment(self, detector = 0):
        """
        Reads the current values of the environment PVs.  Returns a list of
        McaEnvironment objects with Mca.get_environment().
        """
        try:
            if (len(self.env_pvs) > 0):
                
                for i in range(len(self.environment)):
                    
                    val = self.env_pvs[i].get()
                    if type(val) == float:
                        val = round(val,12) # python rounding bug workaround
                    self.environment[i].value = val
        except:
            pass
        env = super().get_environment()
        return env

    #######################################################################
    #######################################################################

    def get_det_rois(self, energy=0):
        """     Reads the ROI information from the EPICS mca record.  Stores this information
        in the epicsMca object, and returns a list of McaROI objects with this information.
        """
        if self.verbose:
            start_time = time.time()

        for det in range(self.n_detectors):
            rois = []
            for i in range(self.max_rois):
                roi = McaROI()
                pvs = self.roi_def_pvs[det][i]
                r = 'R'+str(i)
                roi.left      = pvs[r+'lo'].get()
                roi.right     = pvs[r+'hi'].get()
                roi.label     = pvs[r+'nm'].get()
                #roi.bgd_width = pvs[r+'bg'].get()
                roi.use = 1
                if (roi.left > 0) and (roi.right > 0): rois.append(roi)
            self.auto_process_rois=True
            super().set_rois(rois, detector= det, source='detector')
        if self.verbose:
            print("get_rois --- %s seconds ---" % (time.time() - start_time))
        return self.rois_from_det




    def set_rois(self, rois, energy=0, detector = 0, source='controller'):
        """
        Writes the ROI information from a list of mcaROI objects to the EPICS mca
        record.

        Inputs:
            rois:
                A list of mcaROI objects.

        Keywords:
            energy:
                Set this flag if the .left and .right fields of the ROIs are in energy units.
                By default these fields are assumed to be in channels.
        """
        if self.verbose:
            start_time = time.time()
        nrois = len(rois)
        available_rois = len(self.roi_def_pvs[detector])
        if nrois > available_rois:
            nrois = available_rois
            rois = rois[:nrois]
        super().set_rois(rois, detector=detector, source=source)

        
        nrois = len(rois)
        
        for i in range(nrois):
            roi = rois[i]

            # shorten roi label to 12 characters 
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
            
            pvs = self.roi_def_pvs[detector][i]
            n = 'R'+str(i)
            pvs[n+'lo'].put(roi.left)
            pvs[n+'hi'].put(roi.right)
            pvs[n+'nm'].put(roi.label)
            #pvs[n+'bg'].put(roi.bgd_width)
        for i in range(nrois, self.max_rois):
            n = 'R'+str(i)
            pvs = self.roi_def_pvs[detector][i]
            pvs[n+'lo'].put(-1)
            pvs[n+'hi'].put(-1)
            pvs[n+'nm'].put(" ")
           # pvs[n+'bg'].put(0)
        if self.verbose:
            print("set_rois --- %s seconds ---" % (time.time() - start_time))

    

    def add_roi(self, roi, energy=0, detector = 0, source='controller'):
        """
        Adds a new ROI to the epicsMca.

        Inputs:
            roi:
                An McaROI object.

        Keywords:
            energy:
                Set this flag if the .left and .right fields in the mcaROI are in energy units.
                By default these fields are assumed to be in channels.
        """
        super().add_roi(roi, energy, detector)
        self.set_rois(self.rois[detector],source=source)
        
  

    def add_rois(self, rois, energy=0, detector = 0, source='controller'):
        """
        Adds multiple new ROIs to the epicsMca.

        Inputs:
            rois:
                List of McaROI objects.
        """
        super().add_rois(rois, source=source)
        self.set_rois(self.rois[detector], source=source)

    def delete_roi(self, index, detector = 0):
        """
        Deletes an ROI from the epicsMca.

        Inputs:
            index:
                The index number of the ROI to be deleted (0-31).
        """
        super().delete_roi(index, detector)
        self.set_rois(self.rois[detector])

    def clear_rois(self, source, detector):
        self.set_rois([],detector = detector, source=source)

    #######################################################################
    #######################################################################

    

    def get_calibration(self):
        
        """
        Reads the calibration information from the EPICS mca record.  Stores this information
        in the epicsMca object, and returns an McaCalibration object with this information.
        """
        if self.verbose:
            start_time = time.time()

        calibration = []
        for i in range(self.n_detectors):
            calibration.append( McaCalibration())
            pvs = self.pvs[i]['calibration']
            
            calibration[i].offset    = pvs['calo'].get()
            calibration[i].slope     = pvs['cals'].get()
            calibration[i].quad      = pvs['calq'].get()
            calibration[i].two_theta = pvs['tth'].get()
            calibration[i].units     = pvs['egu'].get()
            calibration[i].set_dx_type('edx')
            

            super().set_calibration(calibration[i], i)
        if self.verbose:
            print("get_calibration --- %s seconds ---" % (time.time() - start_time))
        return self.calibration

    #######################################################################
    def set_calibration(self, calibration):
        """
        Writes the calibration information from an McaCalibration object to the EPICS mca
        record.

        Inputs:
            calibration:
                An McaCalibration instance containing the calibration information.
        """
        if self.verbose:
            start_time = time.time()
        for i in range(self.n_detectors):
            super().set_calibration(calibration, i)
            pvs = self.pvs[i]['calibration']
            pvs['calo'].put(calibration[0].offset)
            pvs['cals'].put(calibration[0].slope)
            pvs['calq'].put(calibration[0].quad)
            pvs['tth'].put(calibration[0].two_theta)
            pvs['egu'].put(calibration[0].units)
        if self.verbose:
            print("set_calibration --- %s seconds ---" % (time.time() - start_time))

        #######################################################################
    def get_presets(self):
        """
        Reads the preset information from the EPICS mca record.  Stores this information
        in the epicsMca object, and returns an McaPresets object with this information.
        """
        presets = McaPresets()
        
        pvs = self.pvs[0]['presets']
        
        presets.real_time       = pvs['prtm'].get()
        presets.live_time       = pvs['pltm'].get()
        presets.total_counts    = pvs['pct'].get()
        presets.start_channel   = pvs['pctl'].get()
        presets.end_channel     = pvs['pcth'].get()
        presets.dwell           = pvs['dwel'].get()
        presets.channel_advance = pvs['chas'].get()
        presets.prescale        = pvs['pscl'].get()
        super().set_presets(presets)
        return presets

    #######################################################################
    def set_presets(self, presets):
        """
        Writes the presets information from an McaPresets object to the EPICS mca
        record.

        Inputs:
            presets:
                An McaPresets instance containing the presets information.
        """
        super().set_presets(presets)
        pvs = self.pvs[0]['presets']
        pvs['prtm'].put(presets.real_time)
        pvs['pltm'].put(presets.live_time)
        pvs['pct'].put(presets.total_counts)
        pvs['pctl'].put(presets.start_channel)
        pvs['pcth'].put(presets.end_channel)
        pvs['dwel'].put(presets.dwell)
        pvs['chas'].put(presets.channel_advance)
        pvs['pscl'].put(presets.prescale)
        
    def get_elapsed(self):
        """
        Reads the elapsed information from the EPICS mca record.  Stores this information
        in the epicsMca object, and returns an McaElapsed object with this information.
        """
        if self.verbose:
            start_time = time.time()
        elapsed = []
        
        for i in range(self.n_detectors):
            elapsed.append(McaElapsed())
            pvs = self.pvs[i]['elapsed']
            
            elapsed[i].real_time    = pvs['ertm'].get()
            elapsed[i].live_time    = pvs['eltm'].get()
            elapsed[i].total_counts = pvs['act'].get()
            elapsed[i].read_time    = pvs['rtim'].get()
            elapsed[i].start_time   = pvs['stim'].get().strip()

        super().set_elapsed(elapsed)
        if self.verbose:
            print("get_elapsed --- %s seconds ---" % (time.time() - start_time))
        return self.elapsed

    def get_data(self, element = 0):
        if self.verbose:
            start_time = time.time()
        data = []
        for i in range(self.n_detectors):
            data.append(self.mcaPV[i].get())

        nuse = self.pvs[0]['data']['nuse'].get()
        self.nchans = nuse
        

        if type(data[0]).__name__ == 'int':   ## if MCA is erased, for some reason the pyepics returns 0 instead of an array
      
            data[0] = np.zeros(nuse) 
        if len(data[0]) !=nuse:
            data[0] = np.zeros(nuse)
            if self.verbose:
                print ('len(data) !=nuse')
        t = type(data[0])
        if t != np.ndarray:
            if self.verbose:
                print ('not np.ndarray')
        super().set_data(data)
        if self.verbose:
            print("get_data --- %s seconds ---" % (time.time() - start_time))
        return self.data

    def get_acquire_status(self):
        return self.pvs[0]['acquire']['acqg'].get()

    def acq_on(self):
        status = self.get_acquire_status()
        if status == 0:
            self.pvs[0]['acquire']['strt'].put(1)
        self.read_done_monitor.SetPVmonitor()
        #self.stop_monitor.SetPVmonitor()
        self.end_time_monitor.SetPVmonitor()
       

    def acq_off(self):
        acq = self.get_acquire_status() 
        if acq == 1:
            self.pvs[0]['acquire']['stop'].put(1)
        self.read_done_monitor.unSetPVmonitor() 
  
        if acq == 1:
            time.sleep(0.25) #wait for late arriving data
            self.dataAcquired.emit()

    def acq_erase(self):
        self.pvs[0]['acquire']['eras'].put(1)

    def acq_erase_start(self):
        
        self.pvs[0]['acquire']['erst'].put(1)

    def is_monitor_on(self):
        return self.read_done_monitor.monitor_On

    def handle_mca_callback(self, Status):
       
        if Status == 'Done':
           
            self.dataAcquired.emit()

    def handle_mca_callback_erase_start(self, Status):
     
        if Status == 'Acquire' or Status == '1':
            if self.acq_status == 'off':
                self.acqOn()

    def handle_mca_callback_erase(self, Status):  
        
        pass


    def handle_mca_callback_start(self, Status):
        pass
        '''if Status == 'Acquire' or Status == '1':
            if self.acq_status == 'off':
                self.acqOn()'''

    def handle_mca_callback_acqg(self, Status):
        
        if Status == 'Acquire' or Status == '1':
            if self.acq_status == 'off':
                self.acqOn()

        if Status == 'Done' or Status == '0':
            if self.acq_status == 'on':
                self.acqOff()
               

    def handle_mca_callback_stop(self, Status):
        
        pass
        '''if Status == 'Done':
            #if self.acq_status == 'on':
            self.set_epics_btns_state('off')
            self.acq_status = 'off'
            
            self.stop_monitor.unSetPVmonitor()
            self.end_time_monitor.unSetPVmonitor()
            self.acq_stopped.emit()'''
            
            

    def handle_mca_callback_end_time(self, Status):
        # this is the only way I could figure out how detect stop when scanning
        
        if Status != self.end_time:
     
            self.acqOff()
            
            self.end_time_monitor.unSetPVmonitor()
            self.acq_stopped.emit()
                
                
    def handle_mca_callback_why_stopped(self, Status):
      
        pass

    def handle_mca_callback_pltm(self, Status):
       
        pass

    def handle_mca_callback_prtm(self, Status):
     
        pass
   
    #######################################################################
    #######################################################################
    def write_file(self, file, netcdf=0):
        """
        Invokes Mca.write_file to write the mca to a disk file.
        After the file is written resets the client wait flag PV, if it exists.
        This flag is typically used by the EPICS scan record to wait for a client application
        to save the file before it goes on to the next point in the scan.

        Inputs:
            file:
                The name of the file to write the mca to.
        """
        
        [file, ok] = super().write_file( file, netcdf)
            
        if ok:
            self.last_saved = copy.copy(self.elapsed[0].start_time)
            
            
        return [file, ok]
        

   ############################################################################

        

    

class epicsMonitor(QtCore.QObject):
    callback_triggered = QtCore.pyqtSignal(str)

    def __init__(self, pv, callback, debounce_time=None, autostart=False):
        super().__init__()
        self.mcaPV = pv
        self.monitor_On = False
        self.emitted_timestamp = None
        


        self.callback_triggered.connect(callback)
        if autostart:
            self.SetPVmonitor()
            

    def SetPVmonitor(self):
        self.mcaPV.clear_callbacks()
        self.mcaPV.add_callback(self.onPVChange)
        self.monitor_On = True
    def unSetPVmonitor(self):
        self.mcaPV.clear_callbacks()
        self.monitor_On = False

    def onPVChange(self, pvname=None, char_value=None, **kws):

        
        self.callback_triggered.emit(char_value)