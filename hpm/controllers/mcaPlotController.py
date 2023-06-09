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
from scipy import interpolate
import copy
from numpy import pi
from PyQt5.QtCore import pyqtSignal, QObject
from hpm.controllers.RoiController import RoiController
from utilities.HelperModule import getInterpolatedCounts
from hpm.models.mcaModel import MCA
from hpm.models.UnitConversions import *
from hpm.models.mcaComponents import McaCalibration
from hpm.widgets.PltWidget import PltWidget

class plotController(QObject):

    fastCursorMovedSignal = pyqtSignal(dict)  
    staticCursorMovedSignal = pyqtSignal(dict) 
    unitUpdated = pyqtSignal(str)
    logScaleYUpdated = pyqtSignal(bool)
    selectedRoiChanged =pyqtSignal(str)
    dataPlotUpdated=pyqtSignal(dict)
    envUpdated=pyqtSignal(list)

    def __init__(self, plotWidget:PltWidget, mcaModel:MCA, mainController, horzScale='E'):
        super().__init__()
        self.mca : MCA
        self.mca = mcaModel
        self.calibration : McaCalibration
        
        self.calibration= self.mca.get_calibration()[0]
        self.pg = plotWidget
        self.mcaController = mainController
        
        #initialize roi controller
        
        self.roi_controller = RoiController(self.mca, self.pg, self, self.mcaController)
        
        self.roi_controller.roi_updated_signal.connect(self.rois_updated)
        self.roi_controller.roi_selection_changed_signal.connect(self.roi_selection_updated)
        self.unit = horzScale
        self.dataInterpolated = None
        self.cursorPosition = None
        self.fastCursorPosition = None
        self.logMode = [False, True]
        self.data = mcaModel.data[0]
        self.envs = []
        self.bottomLabel = ''
        self.LogClip = 0.5
        self.units =  {     'E':'KeV',
                            'd':f'\N{LATIN CAPITAL LETTER A WITH RING ABOVE}',
                            'q':f'\N{LATIN CAPITAL LETTER A WITH RING ABOVE}\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}',
                            'Channel':"",
                            '2 theta':f'\N{SUPERSCRIPT ZERO}'
        }
        self.makeXaxis()
        #self.horzBins = [np.linspace(0,3999,1), 'E', self.units['E']]
        self.pg.plotMouseMoveSignal.connect(self.mouseMoved)         # connect signal to mouse mothion handler 
        self.pg.getViewBox().plotMouseCursorSignal.connect(self.mouseCursor)
        
    def set_mca(self, mca, element=0):
        self.mca = mca

    ########################################################################################
    ########################################################################################

    def update_plot_data(self, element=0):
        
        m = self.mca
        baseline_state = self.mca.baseline_state
        if baseline_state:
            self.data = m.get_data()[element] -m.get_baseline()[element]
        else:

            self.data =  m.get_data()[element]
        
        calibration = m.get_calibration()[element]
        if self.calibration != calibration:
            self.calibration = calibration
            
        self.elapsed = m.get_elapsed()[element]
        self.name = m.get_name()
        
        self.envs = m.get_environment()
        '''envs = {}
        for env in self.envs:
            envs [env.description]=env.value'''
        self.envUpdated.emit(self.envs)

    def updated_baseline_state(self):
        self.update_plot_data()
        self.updateWidget()
        
    def rois_updated(self, ind, text ):
        self.roi_selection_updated(ind, text)
        self.updateWidget()

    def roi_selection_updated(self, ind, text):
        self.roi= self.roi_controller.roi
        self.selectedRoiChanged.emit(text)

    def get_data_label(self):
        dx_type = self.calibration.dx_type

        total_counts = sum(self.data)

        elapsed = self.mca.get_elapsed()[0].real_time
        cps_label = 'cps, '
        if elapsed > 0:
            cps = total_counts / elapsed 
            if cps > 10000:
                cps = cps * 0.001
                cps_label = 'kcps, '

        else:
            cps = 0

        

        if dx_type == 'edx':
            
            data_label = 'MCA, ' + str(int(cps)) + cps_label
            if hasattr(self.calibration, 'two_theta'):
                tth = self.calibration.two_theta
                if tth != None:
                    data_label += f' 2\N{GREEK SMALL LETTER THETA}='+ str(round(tth,4)) + f'\N{SUPERSCRIPT ZERO} '
            data_label += self.elapsed.start_time[:-3]
        elif dx_type == 'adx':
            
            data_label = 'ADXD, '
            if hasattr(self.calibration, 'wavelength'):
                wavelength = self.calibration.wavelength
                if wavelength != None:
                    data_label += f' \N{GREEK SMALL LETTER LAMDA}='+ str(round(wavelength,4))+f'\N{LATIN CAPITAL LETTER A WITH RING ABOVE} '
            data_label += self.name 
        else:
            data_label = 'MCA'
        return data_label

    def updateWidget(self):  
        [xAxis, xLabel, data, 
            nrois, roiHorz, roiData, self.dataInterpolated, 
            logMode] = self.formatDataForPlot(self.data, self.roi)
        # here we plot the foreground 
        pg = self.pg  
        

        data_label = self.get_data_label()
      
        pg.plotData(xAxis, data, roiHorz,roiData,xLabel,data_label)
        self.update_cursors_text()
        update = {'x_range':[min(xAxis),max(xAxis)], 'y_range':[min(data),max(data)]}
        self.dataPlotUpdated.emit(update)
        
    ########################################################################################
    ########################################################################################

    def roi_construct(self, mode, **kwargs):
        if mode == 'Add':
            width = kwargs.get('width', None)
            self.pg.lin_reg_mode(mode,width=width)
        elif mode == 'Set':
            reg = self.pg.lin_reg_mode(mode)
            maximum = max(self.horzBins[0])
            minimum = min(self.horzBins[0])
            if reg[0] > minimum and reg[0]< maximum and reg[1] > minimum and reg[1]< maximum:

                low = getInterpolatedCounts(reg[0],self.horzBins[0])
                high = getInterpolatedCounts(reg[1],self.horzBins[0])
                center = int((low+high)/2)
                half_width = int(abs((high - low))/2)
                return [center,half_width]
            else: return None

    def setLogMode(self, LogMode):
        self.logMode = LogMode
        self.updateWidget()
        self.logScaleYUpdated.emit(self.logMode[1])

    def get_log_mode_y(self):
        return self.logMode[1]

    def get_unit(self):
        return self.unit

    def set_unit(self, unit='Channel'):
        old_unit = self.unit
        element = self.mcaController.element
        inverted_x_old = old_unit == 'd'
        inverted_x_new = unit == 'd'
        #x_direction_changed = inverted_x_old != inverted_x_new
        
        #cursor_old_unit = self.pg.get_cursor_pos()
        #fast_cursor_old_unit = self.pg.get_cursorFast_pos()
        self.is_cursor_in_range()

        cursor_index = self.cursorPosition #self.calibration.scale_to_channel(cursor_old_unit, old_unit)
        fast_cursor_index = self.fastCursorPosition #self.calibration.scale_to_channel(fast_cursor_old_unit, old_unit)
        if cursor_index !=None:
             if cursor_index >= 0:
                self.cursorPosition= cursor_index
                #cursor_new_unit = self.calibration.channel_to_scale(cursor_index,unit)
        if fast_cursor_index != None:
            if fast_cursor_index >= 0:
                self.fastCursorPosition = fast_cursor_index
                #fast_cursor_new_unit = self.calibration.channel_to_scale(fast_cursor_index,unit)
        x_axis_autorange = self.pg.viewBox.state['autoRange'][0]
        if not x_axis_autorange:
            x_axis = self.pg.getAxis('bottom')
            x_range = x_axis.range
            x_min = self.calibration.scale_to_channel(x_range[0],self.unit)
            x_max = self.calibration.scale_to_channel(x_range[1],self.unit)
        self.unit = unit
       
        self.makeXaxis()
        
        self.update_plot_data(element)
        self.roi_controller.data_updated(element) 

        self.unitUpdated.emit(self.unit)
        self.update_cursors()
        
        # update widget viewbox range
        if not x_axis_autorange:
            new_x_min = self.calibration.channel_to_scale(x_min,self.unit)
            new_x_max = self.calibration.channel_to_scale(x_max,self.unit)
            self.pg.setXRange(new_x_min, new_x_max, padding=0)

    

    def getRange(self):
        xAx = self.horzBins[0]
        x_range = [ np.amin(xAx), np.amax(xAx)]
        y_range = [1, np.amax(self.data)]
        return [x_range, y_range]

    def is_cursor_in_range(self, ind=0):
        if self.cursorPosition is not None:
            p = self.calibration.channel_to_scale(self.cursorPosition,self.unit)
            [[x_min, x_max], _] =self.getRange()
            if p > x_min and p < x_max:
                return True
        return False

    def makeXaxis(self):
        self.bins = np.linspace(0,len(self.data)-1, len(self.data))    # datapoint bins
        Scale=self.calibration.channel_to_scale(self.bins,self.unit)
        unit = self.unit
        if unit == '2 theta':
            unit = u'2θ'
        self.horzBins = [Scale, unit , self.units[self.unit ]]

    def formatDataForPlot(self, data, rois):  
        #interpolated data for cursor movement:
        self.bins = np.linspace(0,len(data)-1, len(data))    # datapoint bins
        self.dataInterpolated = interpolate.interp1d(np.flipud(self.bins), np.flipud(data))
        self.makeXaxis()
        xLabel= '%s' %(self.horzBins[1])     
        if self.horzBins[2] != '':
            xLabel += ' (%s)' %(self.horzBins[2]) 
        xAxis = self.horzBins[0]
        if self.logMode[1]:
            data = np.clip(data,self.LogClip,np.Inf)
        if rois is not None:
            nrois = len(rois)
            [roiHorz, roiData, data, xAxis] = self.plotROI(rois, data, xAxis, self.logMode)
        else:
            nrois = 0
            [roiHorz, roiData] = [[],[]]
        xClip = 0
        xAxis = xAxis[xClip:]
        data = data[xClip:]
        return [xAxis, xLabel, data, nrois, roiHorz, roiData, self.dataInterpolated, self.logMode]

    def plotROI(self, rois, data, horz, logMode):
        '''
        extracts regions from data based on rois input, used for plotting 
        roi regions overlaying mca data plot
        inputs:
            rois - list of roi objects from mcaModel
            data - 1d array, (eg. counts)
            horz - 1d array, horizontal scale for data (eg. energy bins)
        outputs:
            roiHorz, roiData
                1-d arrays of extracted regions, interleaved with np.nan values
                can be used for plotting
        '''
        
        roiData = np.array([])
        roiHorz = np.array([])
        roiHorz_tight= np.array([])
        nan = np.array([np.nan])
        dataDel = data.astype(float)
        horzDel = copy.deepcopy(horz)
        for roi in rois:
            use = roi.use
            if use == True:
                left = int(roi.left)
                right = int(roi.right)
                label = roi.label
                roi_data = data[left:right+1]
                roi_horz = horz[left:right+1]
                roiData = np.concatenate((roiData, np.concatenate((roi_data, nan),axis=None)),axis=None)
                roiHorz = np.concatenate((roiHorz, np.concatenate((roi_horz, 1),axis=None)),axis=None)
                roi_horz_tight = horz[left+1:right]
                roiHorz_tight = np.concatenate((roiHorz_tight, np.concatenate((roi_horz_tight, nan),axis=None)),axis=None)
        s = np.in1d(horz,roiHorz_tight)
        horzDel[s] = np.nan
        dataDel[s] = 1
        return [roiHorz, roiData, copy.deepcopy(dataDel), copy.deepcopy(horzDel)]       
    
    def update_cursors(self):
        # update cursor positions or counts
        if self.cursorPosition is not None:
            position = self.calibration.channel_to_scale(self.cursorPosition,self.unit)
            self.mouseCursor(position)
        if self.fastCursorPosition is not None:   
            fastPosition = self.calibration.channel_to_scale(self.fastCursorPosition,self.unit)
            self.mouseMoved(fastPosition)

    def update_cursors_text(self):
        # update cursor positions or counts
        if self.cursorPosition is not None:
            position = self.calibration.channel_to_scale(self.cursorPosition,self.unit)
            self.mouseCursor_text(position)
        if self.fastCursorPosition is not None:   
            fastPosition = self.calibration.channel_to_scale(self.fastCursorPosition,self.unit)
            self.mouseMoved_text(fastPosition)


    def mouseCursor_non_signalling(self, channel):
        point = self.calibration.channel_to_scale(channel,self.unit)
        self.mouseCursor(point)
    
    def mouseMoved(self, mousePoint):
        self.mouseMoved_text(mousePoint)
        self.pg.set_cursorFast_pos(mousePoint)

    def mouseCursor(self, mousePoint):
        self.mouseCursor_text(mousePoint)
        self.pg.set_cursor_pos(mousePoint)


        
    def mouseMoved_text(self, mousePoint):
        frac, out = self.get_label_values(mousePoint)
        self.fastCursorPosition = frac
        self.fastCursorMovedSignal.emit(out)  

    def mouseCursor_text(self, mousePoint):
        frac, out = self.get_label_values(mousePoint)
        self.cursorPosition = frac
        self.staticCursorMovedSignal.emit(out)    

    def get_label_values(self, mousePoint):
        out = {}
        
        cursorPosition = None
        if self.horzBins != None and self.dataInterpolated != None:
            if mousePoint >=0 and mousePoint <= max(self.horzBins[0]):
                cursorPosition = getInterpolatedCounts(mousePoint,self.horzBins[0])
                try:
                    i = self.dataInterpolated(cursorPosition)
                except:
                    i = None
                if i != None:
                    
                    cal = self.calibration
                    
                    units = []
                    other_units = ['E','2 theta','q','d', 'Channel']
                    for unit in other_units:
                        if unit in cal.available_scales and not unit in units and unit != self.unit:
                            units.append(unit)

                    for unit in units:
                        val = cal.channel_to_scale(cursorPosition,unit)
                        out[unit]=val

                    out = {'hName':self.horzBins[1],'hValue':mousePoint,'hUnit':self.horzBins[2],'vName':self.horzBins[1], 'vValue':i, 'channel':cursorPosition}

                    #element = self.mcaController.element
                    
                    
                    for unit in units:
                        val = cal.channel_to_scale(cursorPosition,unit)
                        out[unit]=val
        return cursorPosition, out
    
    def get_cursor_position(self):
        return self.cursorPosition


    