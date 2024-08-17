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

# Principal author: R. Hrubiak (hrubiak@anl.gov)
# Copyright (C) 2018-2019 ANL, Lemont, USA

# based on old IDL code that was translated to Python 2 by Geneveve Boman


from epics import caget, caput, PV
import numpy as np
import time
import sys


from PyQt6.QtCore import QThread, pyqtSignal

class MultiangleSweep(QThread):
    callbackSignal = pyqtSignal(dict)  
    
    def __init__(self, parent):
        super().__init__(parent=parent)  
        self.verbose = True
        self.rep = 1
        self.factor = 1
        self.pvs = {'beamstop'          :'16bmb:Unidig2Bo0',
                    'tth'               :'16bmb:m5',
                    'Primary_vsize'     :'16bmb:m10',
                    'Primary_hsize'     :'16bmb:m12',
                    'Secondary_vsize'   :'16bmb:pm18',
                    'Secondary_hsize'   :'16bmb:pm20',
                    'Detector_vsize'    :'16bmb:m40',
                    'Detector_hsize'    :'16bmb:m38',
                    'mca_erase'         :'16bmb:aim_adc1Erase',
                    'mca_start'         :'16bmb:aim_adc1Start',
                    'mca_live_time'     :'16bmb:aim_adc1.ELTM',
                    'mca_stop'          :'16bmb:aim_adc1Stop'}
        
        
        self.connect_detector()

        self.exp_conditions= np.array([ ])

    def connect_detector(self):
        self.record_name='16bmb:aim_adc1'

        # Construct the names of the PVs for the ROIs
        self.roi_def_pvs=[]
        self.roi_data_pvs=[]
        self.max_rois = 24 
        for i in range(self.max_rois):
            n = 'R'+str(i)
            r = {n+'lo'  : None,
                n+'hi'  : None,
                n+'bg'  : None,
                n+'nm'  : None}
            self.roi_def_pvs.append(r)
            r = {n       : None,
                n+'n'   : None}
            self.roi_data_pvs.append(r)
        for roi in range(self.max_rois):
            for pv in self.roi_def_pvs[roi].keys():
                name = self.record_name + '.' + pv.upper()
                self.roi_def_pvs[roi][pv] = PV(name)
            for pv in self.roi_data_pvs[roi].keys():
                name = self.record_name + '.' + pv.upper()
                self.roi_data_pvs[roi][pv] = PV(name)

    def get_rois(self):
        """     Reads the ROI information from the EPICS mca record.  Stores this information
        in the epicsMca object, and returns a list of McaROI objects with this information.
        """
        if self.verbose:
            start_time = time.time()
        rois = []
        for i in range(self.max_rois):
            roi = ROI()
            pvs = self.roi_def_pvs[i]
            r = 'R'+str(i)
            roi.left      = pvs[r+'lo'].get()
            roi.right     = pvs[r+'hi'].get()
            roi.label     = pvs[r+'nm'].get()
            roi.nAvg      = pvs[r+'bg'].get()
            if (roi.left > 0) and (roi.right > 0): rois.append(roi)
        if self.verbose:
            print("get_rois --- %s seconds ---" % (time.time() - start_time))
        return rois

    def set_exp_conditions(self, config):
        self.exp_conditions= np.array(config)

    def set_factor(self, factor):
        self.factor = factor

    def set_rep(self, rep):
        self.rep = rep   

    def stop(self):
        #print('stopping')
        self.go = 0   

    def run(self):
        self.go = True
        self.multiangle_EDXD()
        self.go = False
        self.callbackSignal.emit({'message':'Done'})

    def multiangle_EDXD(self):
        rep = self.rep
        factor = self.rep
        exp_conditions = self.exp_conditions
        pvs = self.pvs
        print ("exp_cond=", exp_conditions)
        sz = exp_conditions.shape
        num_angles = sz[0]
        
        #Read current positions

        twotheta0 = caget(pvs['tth']+'.RBV')
        Primary_vsize0 = caget(pvs['Primary_vsize']+'.RBV')
        Primary_hsize0 = caget(pvs['Primary_hsize']+'.RBV')
        Secondary_vsize0 = caget(pvs['Secondary_vsize']+'.RBV')
        Secondary_hsize0 = caget(pvs['Secondary_hsize']+'.RBV')
        Detector_vsize0 = caget(pvs['Detector_vsize']+'.RBV')
        Detector_hsize0 = caget(pvs['Detector_hsize']+'.RBV')

        for n in range(0,rep):
            print ("Iteration =", n+1)
            for j in range(0,num_angles):
                self.EDXD_COLLECTING(exp_conditions[j,0],exp_conditions[j,1],exp_conditions[j,2],
                                exp_conditions[j,3],exp_conditions[j,4],exp_conditions[j,5],
                                exp_conditions[j,6],exp_conditions[j,7])
                time.sleep(1.0)
                '''
                rois = self.get_rois()
                r = rois[0]
                print('roi 0 nAvg: '+str(r.nAvg))
                '''
                now = time.localtime(time.time())
                t = time.strftime("%a_%d%b%y_%H_%M", now)
                try:
                    print ("2th =", exp_conditions[j,0], "EDXD_COLLECTING finished at", t)
                except:
                    print ("End of data collection")
                if not self.go:
                    break 
            if not self.go:
                    break 
            time.sleep(10.0)
            if not self.go:
                break 

        caput(pvs['tth']+'.VAL', twotheta0)
        caput(pvs['Primary_vsize']+'.VAL',Primary_vsize0)
        caput(pvs['Primary_hsize']+'.VAL',Primary_hsize0)
        caput(pvs['Secondary_vsize']+'.VAL', Secondary_vsize0)
        caput(pvs['Secondary_hsize']+'.VAL', Secondary_hsize0)
        caput(pvs['Detector_vsize']+'.VAL', Detector_vsize0)
        caput(pvs['Detector_hsize']+'.VAL', Detector_hsize0)
        caput(pvs['beamstop'], 1)

    def MOVE_TTH(self, tth, mcurrent_tth, mcurrent_beamstop):
        pvs = self.pvs

        if tth < mcurrent_tth and tth < 6.95 and mcurrent_beamstop == 0:
            caput(pvs['beamstop'], 1)
            time.sleep(10)
            self.START_MOVE_TTH(tth)

        if tth > mcurrent_tth and tth > 6.95 and mcurrent_beamstop ==1:
            self.START_MOVE_TTH(tth)
            caput(pvs['beamstop'], 0)
            time.sleep(10.0)

        if tth == mcurrent_tth and tth >= 6.95:
            caput(pvs['beamstop'], 0)
            time.sleep(10.0)
        self.START_MOVE_TTH(tth)

    def START_MOVE_TTH(self, tth):
        pvs = self.pvs
        temp = caget(pvs['tth']+'.RBV')
        if tth < temp:
            caput(pvs['tth']+'.VAL', tth)
            while temp > tth+0.005:
                temp = caget(pvs['tth']+'.RBV')
                time.sleep(1.0)
        if tth > temp:
            caput(pvs['tth']+'.VAL', tth)
            while temp < tth-0.005:
                temp = caget(pvs['tth']+'.RBV')
                time.sleep(1.0)

    def EDXD_COLLECTING(self, tth, pvsize, phsize, svsize, shsize, dvsize, dhsize, stime):
        pvs = self.pvs
        current_tth = caget(pvs['tth']+'.RBV')
        current_beamstop=caget(pvs['beamstop'])

        self.MOVE_TTH(tth, current_tth, current_beamstop)

        #set the slit and 2theta condition
        caput(pvs['Primary_vsize']+'.VAL', pvsize)
        caput(pvs['Primary_hsize']+'.VAL', phsize)
        caput(pvs['Secondary_vsize']+'.VAL', svsize)
        caput(pvs['Secondary_hsize']+'.VAL', shsize)
        caput(pvs['Detector_vsize']+'.VAL', dvsize)
        caput(pvs['Detector_hsize']+'.VAL', dhsize)    

        #start collecting data
        caput(pvs['mca_erase'], 1)
        time.sleep(0.2)
        caput(pvs['mca_start'], 1) 
        livetime = caget(pvs['mca_live_time'])
        while livetime < stime-0.1:
            livetime = caget(pvs['mca_live_time'])
            if not self.go:
                break 
            time.sleep(0.2)
        caput(pvs['mca_stop'], 1)
        time.sleep(0.2)


class ROI():
    """
    A simple class that defines a Region-Of-Interest (ROI)
    Fields
        .left      # Left channel or energy
        .right     # Right channel or energy
        .vAvg      # number average of counts in the ROI
        .label     # label of ROI
    """
    def __init__(self, left=0., right=0.,label='',nAvg=0):
        """
        Keywords:
            There is a keyword with the same name as each attribute that can be
            used to initialize the ROI when it is created.
        """
        self.left = left
        self.right = right
        self.label = label
        self.nAvg = nAvg
        
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