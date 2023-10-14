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
import pyqtgraph as pg
from axd.models import aEDXD_atomic_parameters

import numpy as np
from scipy import interpolate
from PyQt5 import QtCore, QtWidgets
#from pyqtgraph.functions import pseudoScatter
import os
import time
import copy
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom

from .mcareaderGeStrip import *
from .mcaModel import McaCalibration, McaElapsed, McaROI, McaEnvironment
from utilities.CARSMath import fit_gaussian
from .eCalModel import calc_parabola_vertex, fit_energies
from .mcaComponents import McaROI
import hpm.models.Xrf as Xrf

from hpm.models.MaskModel import MaskModel
from axd.models.aEDXD_functions import I_base_calc
from axd.models.aEDXD_atomic_parameters import aEDXDAtomicParameters

from .. import resources_path

class AnalysisStep(QtCore.QObject):
    updated = QtCore.pyqtSignal(np.ndarray)
    def __init__(self, name, data_in_dims, data_out_dims, mask=False):
        super().__init__()
        self.name = name

        self.params_in = {}
        self.params_out = {}

        self.function_inputs = {}
        self.function_outputs = {}

        self.params_in['data_in_dims'] = data_in_dims
        self.params_out['data_out_dims'] = data_out_dims

        self.params_in['mask'] = mask  # this is used only by the controller to determine what kind of widget is needed

        self.analysis_function = None
        self.processed = False

    def set_param(self, param_in):
        for param in param_in:
            self.params_in[param] = param_in[param]

    def set_function(self, f , inputs, outputs):
        self.analysis_function = f
        self.function_inputs = inputs
        self.function_outputs = outputs

    def calculate (self):
        if self.analysis_function is not None:

            data = self._get_function_params()
            out = self.analysis_function(**data)
            
            for param in self.function_outputs:
                self.params_out[param] = out[param]
            self.processed = True
            self.updated.emit(self.params_out['data_out'])

    def _get_function_params(self):
        out = {}
        for param in self.function_inputs:
            out[param] = self.params_in[param]
        return out

    def set_data_in(self, data):
        self.processed = False
        self.params_in['data_in'] = data
        #self.params_in['mask_img'] = np.zeros(data.shape, dtype=bool)
        #self.params_in['weights'] = np.ones(data.shape)

    def get_data_out(self):
        return self.params_out['data_out']

    def get_unit_out(self):
        return self.params_out['unit_out']

    def get_data_out_dims(self):
        return self.params_out['data_out_dims']

    def load_mask(self, filename):
        self.processed = False
     
        if 'mask_model' in self.params_in:
            mask_model = self.params_in['mask_model']
            data = self.params_in['data_in']
            mask_model.set_dimension(data.shape)
            mask_model.load_mask(filename)

            mask = mask_model.get_mask()
            self.params_in['mask_img'] = mask

    def set_mask(self, mask):
        self.processed = False
        self.params_in['mask_img'] = mask
        if 'mask_model' in self.params_in:
            mask_model = self.params_in['mask_model']
            mask_model.set_dimension(mask.shape)
            mask_model.set_mask(mask)

    def get_mask(self):
        mask = self.params_in['mask_img']
        if 'mask_model' in self.params_in:
            mask_model = self.params_in['mask_model']
            mask = mask_model.get_mask()
        return mask


class AmorphousAnalysisModel(QtCore.QObject):  # 
    def __init__(self,  *args, **filekw):
        
        self.mca = None
        self.multi_spectra_model =  None

        self.element_opts = {'elements':['Si','O'],'choices_abc':['Si4+','O1-'],'choices_mkl':['Si','O'], 'fractions':[.333,.667]}

        self.ap = aEDXDAtomicParameters()

        self.steps = {}
        self.make_calculators()

    def set_models(self, mca, multi_spectra_model):
        
        self.mca = mca
        self.multi_spectra_model = multi_spectra_model


    def clear(self):
        self.__init__()

    def make_calculators(self):
        
        # this is a placeholder for the eventual calculation
        # List of the steps needed for the calculation
        # 1. Convert dataset to E
        # 2. apply mask in E (punch and fill any peaks)
        # 3. Flaten to 1D (np.mean, axis=0) , this is now the estimated beam profile
        # 4. Normalize the original dataset in E by the estimated beam profile
        # 5. convert to q
        # 6. apply any mask in q (punch and fill any peaks)
        # 7. calculate 2-theta-dependent scaling factors by looking at consecutive brightness of the nearby rows
        # 8. Flaten to 1D (np.mean, axis=0), this is now the estimated Iq (flattening is weighted and masked )
        # 9. convert Iq to E for each row
        #10. normalize dataset in E by Iq, remember to scale Iq for each row by 2-theta-dependent scaling factors
        #11. Flaten to 1D (np.mean, axis=0) , this is now the estimated beam profile

        # repeat steps 4. to 11. 
    
        # .. convert Iq to Sq by applying scattering factors correction


        steps = {}
        self.defs = [('dataset E',2,2),
                ('mask in E',2,2,True),
                ('blur',2,2),
                ('Flaten 1',2,1),
                ('unFlaten 1',1,2),
                ('Normalize',2,2),
                
                ('convert to q',2,2),
                ('convert to q Ieff',1,2),
                ('mask in q',2,2,True),
                ('weights q',2,2),
                ('get row scale',2,1),
                ('apply row scaling',2,2),
                ('a_I_t(q)',2,1), # I_t = I_coh + I_inc
                ('Iq to E', 1,2),
                ('normalize by Iq',2,2),
 
                ('I_eff(E)',2,1),
                ('I_base(q)',1,1),
                ('Scale factor',1,1),
                ('I_t(q)',1,1),
                ('I_t-I_base',1,1),
                ('(mean_fq)^2',1,1),
                ('S(q)',1,1),
                ]
        for i, d in enumerate(self.defs):

            steps[d[0]] = AnalysisStep(*d)
        
        steps['dataset E'].set_function(self._propagate_data, ['data_in'],  ['data_out'])
        steps['mask in E'].set_function(self._propagate_data, ['data_in'],  ['data_out'])
        steps['blur'].set_function(self._bin, ['data_in', 'sigma', 'scaling_factor'],  ['data_out'])
        steps['Flaten 1'].set_function(self._flaten_data, ['data_in', 'range','unit_in','scale_in', 'mask_img','weights'],  ['data_out'])
        steps['unFlaten 1'].set_function(self._unflaten_data, ['data_in', 'rows'],  ['data_out'])
        
        
        steps['Normalize'].set_function(self._normalize, ['data_in', 'mask_img', 'norm_function'],  ['data_out'])
        
        steps['convert to q'].set_function(self._rebin, ['data_in', 'unit_in', 'unit_out', 'mask_img'],  ['data_out', 'mask_out'])
        steps['convert to q Ieff'].set_function(self._rebin, ['data_in', 'unit_in', 'unit_out', 'mask_img'],  ['data_out', 'mask_out'])
        steps['mask in q'].set_function(self._propagate_data, ['data_in'],  ['data_out'])
        
        steps['weights q'].set_function(self._propagate_data, ['data_in'],  ['data_out'])
        steps['get row scale'].set_function(self._get_row_scale, ['data_in', 'mask_img','weights'],  ['data_out'])
        steps['apply row scaling'].set_function(self._apply_row_scale, ['data_in', 'row_scale_in'],  ['data_out'])
        steps['Iq to E'].set_function(self._q_to_channel, ['data_in'],  ['data_out'])

        steps['normalize by Iq'].set_function(self._normalize_3d, ['data_in','mask_img', 'norm_function'],  ['data_out'])
        steps['a_I_t(q)'].set_function(self._flaten_data, ['data_in',  'range','unit_in','scale_in', 'mask_img','weights'],  ['data_out'])

        steps['I_eff(E)'].set_function(self._flaten_data, ['data_in','range','unit_in','scale_in', 'mask_img','weights'],  ['data_out'])
       
        steps['I_base(q)'].set_function(self._get_I_base, ['data_in', 'opts','unit_in','scale_in'],  ['data_out'])
        steps['Scale factor'].set_function(self._get_scaling_factor, ['data_in', 'I_base', 'range','unit_in','scale_in'],  ['data_out'])

        steps['I_t(q)'].set_function(self._apply_scale_2d, ['data_in', 'scale_in'],  ['data_out'])
        steps['I_t-I_base'].set_function(self._subtract_2d, ['data_in', 'subtract_data'],  ['data_out'])
        steps['(mean_fq)^2'].set_function(self._get_mean_fq_squared, ['data_in', 'opts','unit_in','scale_in'],  ['data_out'])
        steps['S(q)'].set_function(self._get_Sq, ['data_in', 'I_base', '(mean_fq)^2','a', 'unit_in','scale_in'],  ['data_out'])

        self.steps = steps

    def calculate_1(self):

        self.multi_spectra_model.rebin_scale('E') 
        self.multi_spectra_model.rebin_scale('q') 

        self.scale_E = self.multi_spectra_model.E_scale
        self.scale_q = self.multi_spectra_model.q_scale

       
        
        self.data_in_E = copy.deepcopy(self.multi_spectra_model.E )
    
        self.data_in_q = copy.deepcopy(self.multi_spectra_model.q )
   
        #self.data_in_E[:,:600] = 0

        for step in (1,2,3,4,5,6,7,8,9,10,11,12, 13):
            self._calculate_step(step)

        for n in range(5):
            self.calculate_2()
            self.calculate_3()
        
        self.calculate_4()

    def calculate_2(self):
        for step in (14,):
            self._calculate_step(step)

    def calculate_3(self):
        for step in (5, 6,15,10,11,12, 13):
            self._calculate_step(step)

    def calculate_4(self):
        for step in (16,):
            self._calculate_step(step)

    def calculate_5(self):
        for step in (7,8,9,10,11, 12, 13):
            self._calculate_step(step)
        

    def _calculate_step(self, step):
        if step == 1:
            self.steps['dataset E'].set_data_in(self.data_in_E)
            self.steps['dataset E'].calculate()
        elif step == 2:
            data = self.steps['dataset E'].get_data_out()
            mask = np.zeros(data.shape) == 1
         
            data[:,:600] = 0
            self.steps['blur'].set_data_in(data)
            self.steps['blur'].set_param({'sigma':10, 'scaling_factor':10})
            self.steps['blur'].set_mask(mask)
            self.steps['blur'].calculate()
        
        elif step == 3:
            data = self.steps['blur'].get_data_out()
            rows = data.shape[0]

            
            self.steps['Flaten 1'].set_data_in(data)

            mask = np.zeros(data.shape) == 1
            #mask = self.steps['mask in E'].get_mask()
            weights_E = np.ones(data.shape)

            start_row = int(rows * 0.8)
            self.steps['Flaten 1'].set_param({'mask_img':mask, 'range':(start_row,rows-1)})
            self.steps['Flaten 1'].set_param({'unit_in':'E','scale_in':self.scale_E, 'weights':weights_E})
            self.steps['Flaten 1'].calculate()


            flattened = self.steps['Flaten 1'].get_data_out()[1]
            f_max = np.amax(flattened)
            flattened_norm = flattened/f_max
            #flattened_norm[:650] = 0
            indices_for_min = np.where(flattened_norm > 0.20)
            min_ind = np.amin(indices_for_min)
            indices_for_max = np.where(flattened_norm > 0.03 )
            max_ind = np.amax(indices_for_max)
          
            print(min_ind)
            print(max_ind)

            data = self.steps['dataset E'].get_data_out()
            self.steps['mask in E'].set_data_in(data)
            mask = np.zeros(data.shape) == 1
            mask[:,:min_ind] = True
            mask[:,max_ind:] = True
            self.steps['mask in E'].set_mask(mask)
            self.steps['mask in E'].calculate()

            mask = self.steps['mask in E'].get_mask()
            self.steps['Flaten 1'].set_param({'mask_img':mask, 'range':(start_row,rows-1)})
            self.steps['Flaten 1'].calculate()
            flattened = self.steps['Flaten 1'].get_data_out()[1]

            self.steps['unFlaten 1'].set_data_in(flattened)
            self.steps['unFlaten 1'].set_param({'rows':rows})
            self.steps['unFlaten 1'].calculate()
            
        elif step == 4:
            data = self.steps['mask in E'].get_data_out()
            
            self.steps['Normalize'].set_data_in(data)
            mask = self.steps['mask in E'].get_mask()
            self.steps['Normalize'].set_param({'mask_img':mask})
            norm_function = self.steps['Flaten 1'].get_data_out()[1] 
            self.steps['Normalize'].set_param({'norm_function':norm_function})
            self.steps['Normalize'].calculate()
        elif step == 5:
            self.steps['convert to q'].set_data_in(self.steps['Normalize'].get_data_out())
            mask = self.steps['mask in E'].get_mask()
            self.steps['convert to q'].set_param({'mask_img':mask})
            self.steps['convert to q'].set_param({'unit_in':'E', 'unit_out':'q'})
            self.steps['convert to q'].calculate()

            self.steps['convert to q Ieff'].set_data_in(self.steps['unFlaten 1'].get_data_out())
            mask = self.steps['mask in E'].get_mask()
            self.steps['convert to q Ieff'].set_param({'mask_img':mask})
            self.steps['convert to q Ieff'].set_param({'unit_in':'E', 'unit_out':'q'})
            self.steps['convert to q Ieff'].calculate()

        elif step == 6:
            

            data = self.steps['convert to q'].get_data_out()
            self.steps['mask in q'].set_data_in(data)
            
            mask = np.zeros(data.shape) == 1
            #self.steps['mask in q'].load_mask('/Users/ross/GitHub/hp-edxd/hpm/resources/q.mask')
            self.steps['mask in q'].set_mask(mask)
            self.steps['mask in q'].calculate()   

        elif step == 7:
           
            w_q = self.steps['convert to q Ieff'].get_data_out()
            #w_q = np.sqrt(w_q)
            w_q = w_q / np.amax(w_q)
            mask = self.steps['mask in q'].get_mask()
            w_q [w_q==0 ] = 1e-9
            
            self.steps['weights q'].set_data_in(w_q)
            self.steps['weights q'].calculate()

            #################################################

            ####### GET row scale may not be working well yet

            #################################################

            self.steps['get row scale'].set_data_in(self.steps['mask in q'].get_data_out())
            
            self.steps['get row scale'].set_param({'mask_img':mask})
            
            self.steps['get row scale'].set_param({'weights':w_q, 'unit_in':'q','scale_in':self.scale_q})
            self.steps['get row scale'].calculate()

        elif step == 8:
            self.steps['apply row scaling'].set_data_in(self.steps['mask in q'].get_data_out())
     
            row_scale_in = self.steps['get row scale'].get_data_out()
            self.steps['apply row scaling'].set_param({'row_scale_in':row_scale_in})
            self.steps['apply row scaling'].calculate()    

        elif step == 9:

            data = self.steps['apply row scaling'].get_data_out()
            rows = data.shape[0]
            self.steps['a_I_t(q)'].set_data_in(data)

            mask = self.steps['mask in q'].get_mask()
            self.steps['a_I_t(q)'].set_param({'mask_img':mask, 'range':(5,rows-1)})
            
            w_q = self.steps['weights q'].get_data_out()
            
            self.steps['a_I_t(q)'].set_param({'weights':w_q, 'unit_in':'q','scale_in':self.scale_q})
            self.steps['a_I_t(q)'].calculate()
            
        elif step == 10:
            self.steps['Iq to E'].set_data_in(self.steps['a_I_t(q)'].get_data_out())
            self.steps['Iq to E'].calculate()

        elif step == 11:
            self.steps['normalize by Iq'].set_data_in(self.steps['mask in E'].get_data_out())
            mask = self.steps['mask in E'].get_mask()
            self.steps['normalize by Iq'].set_param({'mask_img':mask})
            self.steps['normalize by Iq'].set_param({'norm_function':self.steps['Iq to E'].get_data_out()})
            self.steps['normalize by Iq'].calculate()

        elif step == 12:

            pass
            """self.steps['apply row scaling 2'].set_data_in(self.steps['normalize by Iq'].get_data_out())
            row_scale_in = self.steps['get row scale'].get_data_out()
            self.steps['apply row scaling 2'].set_param({'row_scale_in':row_scale_in})
            self.steps['apply row scaling 2'].calculate()   """ 

            '''self.steps['apply row scaling 3'].set_data_in(self.steps['dataset E'].get_data_out())
            row_scale_in = self.steps['get row scale'].get_data_out()
            self.steps['apply row scaling 3'].set_param({'row_scale_in':row_scale_in})
            self.steps['apply row scaling 3'].calculate()   ''' 

        elif step == 13:
            data = self.steps['normalize by Iq'].get_data_out()
            rows = data.shape[0]
            self.steps['I_eff(E)'].set_data_in(data)

            mask = self.steps['mask in E'].get_mask()

            
            rows = data.shape[0]
            start_row = int(rows * 0.25)
            end_row = int (rows * 0.75)

            self.steps['I_eff(E)'].set_param({'mask_img':mask, 'range':(start_row,end_row)})

            weights = np.ones(data.shape)

            self.steps['I_eff(E)'].set_param({'unit_in':'E','scale_in':self.scale_E, 'weights':weights})
            self.steps['I_eff(E)'].calculate()


        elif step == 14:

            data = self.steps['mask in E'].get_data_out()
            
            self.steps['Normalize'].set_data_in(data)
            mask = self.steps['mask in E'].get_mask()
            self.steps['Normalize'].set_param({'mask_img':mask})
            norm_function = self.steps['I_eff(E)'].get_data_out()[1] 
            self.steps['Normalize'].set_param({'norm_function':norm_function})
            self.steps['Normalize'].calculate()

       
        
        elif step == 15:
            data = self.steps['convert to q'].get_data_out()
            rows = data.shape[0]
            self.steps['a_I_t(q)'].set_data_in(data)

            mask = self.steps['mask in q'].get_mask()
            self.steps['a_I_t(q)'].set_param({'mask_img':mask, 'range':(0,rows-1)})
            
            w_q = self.steps['weights q'].get_data_out()
            
            self.steps['a_I_t(q)'].set_param({'weights':w_q, 'unit_in':'q','scale_in':self.scale_q})
            self.steps['a_I_t(q)'].calculate()

        elif step == 16:
            data = self.steps['a_I_t(q)'].get_data_out()
            q = data[0]
            self.steps['I_base(q)'].set_data_in(q)
            self.steps['I_base(q)'].set_param({'opts':self.element_opts, 'unit_in':'q','scale_in':self.scale_q})
            self.steps['I_base(q)'].calculate()

            self.steps['Scale factor'].set_data_in(data)
            I_base = self.steps['I_base(q)'].get_data_out()
            self.steps['Scale factor'].set_param({'range':[500,2200], 'I_base':I_base,'unit_in':'q','scale_in':self.scale_q})
            self.steps['Scale factor'].calculate()

            scale_in = self.steps['Scale factor'].get_data_out()[1][0]
            self.steps['I_t(q)'].set_data_in(data)
            self.steps['I_t(q)'].set_param({'scale_in':scale_in})
            self.steps['I_t(q)'].calculate()

            I_t = self.steps['I_t(q)'].get_data_out()
            self.steps['I_t-I_base'].set_data_in(I_t)    
            subtract_data = I_base
            self.steps['I_t-I_base'].set_param({'subtract_data':subtract_data})
            self.steps['I_t-I_base'].calculate()
            I_t_minus_I_base = self.steps['I_t-I_base'].get_data_out()
            
            self.steps['(mean_fq)^2'].set_data_in(q)
            self.steps['(mean_fq)^2'].set_param({'opts':self.element_opts, 'unit_in':'q','scale_in':self.scale_q})
            self.steps['(mean_fq)^2'].calculate()
            mean_fq_squared = self.steps['(mean_fq)^2'].get_data_out()

            pg.plot(mean_fq_squared[0],mean_fq_squared[1])
            
            self.steps['S(q)'].set_data_in(I_t)
            self.steps['S(q)'].set_param({'I_base':I_base,'I_t-I_base':I_t_minus_I_base , '(mean_fq)^2':mean_fq_squared, 'a':scale_in,  'unit_in':'q','scale_in':self.scale_q})
            self.steps['S(q)'].calculate()
 
    def _propagate_data(self, **args):
        # propagates data_in to data_out
        data = args['data_in']
        args['data_out'] = data
        return args

    def _mask_zeros(self, **args):
        data = args['data_in']
        mask = data == 0
        args['mask_img'] = mask
        args['data_out'] = data
        return args

    def _unflaten_data(self, **args):

        # copy a 2d data to all rows, making it 3D
        data = args['data_in']
        rows = args['rows']
        out = np.zeros((rows,len(data)))
        for row in range(rows):
            out[row,:] = data
        args['data_out'] = out
        return args
  
    def _flaten_data(self, **args):

        # Compute the average beam profile by averaging all the rows while in energy space. 
        # Then, convert that average profile to q space then you can use that to 
        # normalize all of the data rows.
        # do a weighted average because the high energy / low energy bins will be more noisy 
        
        rng = args['range']
        data = args['data_in'][rng[0]:rng[1]]
        unit = args['unit_in']
        mask = args['mask_img'][rng[0]:rng[1]]
        scale = args['scale_in']
        weights = args['weights'][rng[0]:rng[1]]
        
        not_mask = np.logical_not(mask)
        reduced_not_mask = np.logical_or.reduce(not_mask, axis=0)
        reduced_mask = np.logical_not(reduced_not_mask)

        weights = np.ma.array(weights, mask=mask)
        y = np.ma.array( np.average(np.ma.array(data, mask=mask), weights= weights, axis=0 ), mask = reduced_mask)
        y[reduced_mask] = 0
        if unit == 'E':
            scale = self.multi_spectra_model.E_scale
        elif unit == 'q':
            scale = self.multi_spectra_model.q_scale

        #l,r = self.find_non_zero_range(y)
        #y[:l]=y[l]
        
        x = np.arange(len(y)) * scale[0] + scale[1]
        args['data_out'] = np.asarray([x, y])
        return args

    def _normalize(self, **args):
        # divides each row in the data_in (2D) by the norm_function (2D)
        data = args['data_in']
        mask = args['mask_img']
        data = np.ma.array( data, mask = mask)
        
        out = np.zeros(data.shape)
        norm_function = args['norm_function']
        norm_function[norm_function==0]= np.amin(norm_function[norm_function!=0])
        for i in range(np.shape(data)[0]):
           out[i] = data[i] / norm_function
        out [mask] = 0
        args['data_out'] = out
        return args

    def _get_I_base(self, **args):
        q = args['data_in']
        
        opts = args['opts']
        elements = opts['elements']
        fractions = opts['fractions']
        choices_abc = opts['choices_abc']
        choices_mkl= opts['choices_mkl']
        unit = args['unit_in']
        
        sq_par = self.get_sq_par(elements, fractions,choices_abc, choices_mkl)
        

        mean_fqsquare,mean_fq,mean_I_inc = I_base_calc(q,q,sq_par)
        Iq_base = mean_fqsquare + mean_I_inc
        
        
        args['data_out'] = np.asarray([q, Iq_base])
        return args

    def _get_mean_fq_squared(self, **args):
        q = args['data_in']
        
        opts = args['opts']
        elements = opts['elements']
        fractions = opts['fractions']
        choices_abc = opts['choices_abc']
        choices_mkl= opts['choices_mkl']
        unit = args['unit_in']
        
        sq_par = self.get_sq_par(elements, fractions,choices_abc, choices_mkl)
        

        mean_fqsquare,mean_fq,mean_I_inc = I_base_calc(q,q,sq_par)
        # Iq_base = mean_fqsquare + mean_I_inc
        
        
        args['data_out'] = np.asarray([q, mean_fq**2 ])
        return args

    
    def _get_Sq(self, **args):
        data_in  = args['data_in']
        q = data_in[0]
        I_t = data_in[1]
        
        I_base = args['I_base'][1]
        a = args['a']
        mean_fq_squared = args['(mean_fq)^2'][1]
        mean_fq_squared_max = max(mean_fq_squared)
        print('mean_fq_squared_max '+str(mean_fq_squared_max))
        mean_fq_squared[mean_fq_squared<mean_fq_squared_max*0.005] =  mean_fq_squared_max*0.005

        unit = args['unit_in']
        
        
        Sq = (I_t  - I_base)/mean_fq_squared  +1
        
        args['data_out'] = np.asarray([q, Sq])
        return args

    def _get_scaling_factor(self, **args):
        rng = args['range']
        data = args['data_in']
        q = data[0]
        I_t = data[1]
        
        
        I_base = args['I_base'][1]
        unit = args['unit_in']
        
        
        a = np.ones(q.shape) * np.average( I_base[rng[0]:rng[1]]/I_t[rng[0]:rng[1]])
        
        
        
        args['data_out'] = np.asarray([q, a])
        return args

    def _apply_scale_2d(self, **args):
        
        data = args['data_in']
        scale_in = args['scale_in']
        
        new_data= data[1] * scale_in

        args['data_out'] = np.asarray([data[0],new_data])
        return args

    def _subtract_2d(self, **args):
        # divides each row in the data_in (3D) by the norm_function (3D)
        data = args['data_in']
        
        subtract_data = args['subtract_data'][1]
        
        out = data[1] - subtract_data 
        
        args['data_out'] = np.asarray([data[0],out])

        return args    

    def _subtract_3d(self, **args):
        # divides each row in the data_in (3D) by the norm_function (3D)
        data = args['data_in']
        
        subtract_data = args['subtract_data']
        
        out = data - subtract_data 
        out [out<=0] = 1e-7
        args['data_out'] = out
        return args    

    def _normalize_3d(self, **args):
        # divides each row in the data_in (3D) by the norm_function (3D)
        data = args['data_in']
        mask = args['mask_img']
        norm_function = args['norm_function']
        norm_function[norm_function<0.01]= 1
        
        out = data / norm_function
        out [ mask] = 0
        args['data_out'] = out
        return args

    

    def _get_row_scale(self, **args):
        # calculates the relative scaling of rows with the last row having the scale value of 1
        # uses masked and weighted scaling or adjecent rows
        # the weights are the normalized counts of the Im
        data = args['data_in']
        mask = args['mask_img']
        weights = args['weights']
        rows = np.arange(data.shape[0])
        new_data = np.ones(data.shape[0])
        last_row_index = rows[-1]
        row = data[last_row_index]
        mask_row = mask[last_row_index]
        weights_row = np.ma.array(weights[last_row_index], mask=mask_row)
        
        first_nonzero_index, last_nonzero_index = self.find_non_zero_range(row) 
        scale = 1
        for next_row_index in np.flip( rows):
            next_row = data[next_row_index]
            next_mask_row = mask[next_row_index]
            common_mask = next_mask_row | mask_row
            next_first_nonzero_index, next_last_nonzero_index = self.find_non_zero_range(next_row) 
            first_index = max(first_nonzero_index,next_first_nonzero_index)
            last_index = min(last_nonzero_index, next_last_nonzero_index)
            common_mask[:first_index] = True
            common_mask[last_index:] = True
            common_mask[row<1] = True
            common_mask[next_row<1] = True
            #row_masked = np.ma.array(row, mask=common_mask)
            #next_row_masked = np.ma.array(next_row, mask=common_mask)

            row_masked_subset = row[np.logical_not(common_mask)]
            next_row_masked_subset = next_row[np.logical_not(common_mask)]
            divided = next_row_masked_subset / row_masked_subset
            weights_row_subset = weights_row[np.logical_not(common_mask)]
            weighted_average_subset = np.sum(divided * weights_row_subset)/ np.sum(weights_row_subset)
            average_scale = weighted_average_subset
            #divided = np.ma.array(next_row_masked/row_masked, mask = common_mask)
            #average_scale = np.ma.average( divided, weights=weights_row)
            #average_scale = np.average (divided )
            scale = scale * average_scale
            new_data[next_row_index] = scale
            row = next_row
            mask_row = next_mask_row
            next_weights_row = np.ma.array(weights[next_row_index], mask=common_mask)
            
            weights_row = next_weights_row
            first_nonzero_index, last_nonzero_index = next_first_nonzero_index, next_last_nonzero_index 
        args['data_out'] = np.asarray([rows, new_data])
        return args

    def _apply_row_scale(self, **args):
        # calculates the relative scaling of rows with the last row having the scale value of 1
        data = args['data_in']
        row_scale_in = args['row_scale_in'][1]
        row_scale_in [row_scale_in==0 ] = 1e-7  # avoid division by 0
        rows = np.arange(data.shape[0])
        new_data = np.zeros(data.shape)
        for row in rows:
            new_data[row] = data[row] / row_scale_in[row]

        args['data_out'] = new_data
        return args

    def _bin(self, **args):
        # applies a gaussian blur to the 3d data, rebins the channels before applying the blur

        data = args['data_in']
        sigma = args['sigma']
        scaling_factor = args['scaling_factor']
        #mask = args['mask_img']
        array = data
        scaling_factor = 10
        channels = data.shape[1]
        binned_array = np.mean(array.reshape(-1, scaling_factor), axis=1).reshape(-1, int(channels/scaling_factor))
        blurred_array = gaussian_filter(binned_array, sigma=sigma)
        interpolated_array = zoom(blurred_array, zoom=(1, scaling_factor))
        args['data_out'] = interpolated_array
        
        #args['mask_out'] = new_mask
        return args

    def _rebin(self, **args):
        # re-bins the data_in into a different scale, 
        # for example change each row in data_in from E scale to q scale
        data = args['data_in']
        
        unit_out = args['unit_out']
        mask = args['mask_img']
        new_mask = copy.deepcopy(mask)
        new_data = np.ones(data.shape)

        self._rebin_to_scale(data, new_data, mask, new_mask, unit_out)
        
        args['data_out'] = new_data
        args['mask_out'] = new_mask
        return args

    def _q_to_channel(self, **args):

        # used to convert 2D data in (q) back to 3D data in (E, 2theta)
        data = args['data_in']
        x = data[0]
        rows = len(self.mca.get_calibration())
        y = data[1]
        data_3d = np.zeros((rows,len(x)))
        for row in range(rows):
            data_3d[row] = y
        unit = 'q'

        new_data = np.zeros((rows, len(x)))

        self._rebin_to_channel(x,data_3d, new_data, unit) # this updates the new_data array directly
        #new_data[new_data<0.01]= 1

        args['data_out']=new_data

        return args
    
    def _rebin_to_channel(self, x, data, new_data, unit):
        rows = len(data)
        bins = np.size(data[0])
        calibrations = self.mca.get_calibration()
        q_ranges = []
        channel_ranges = []
        for row in range(rows):
            calibration = calibrations[row]
            ch = calibration.scale_to_channel(x,unit)
            q_ranges.append(ch)
            channel_ranges.append(np.arange(bins))
        q_ranges = np.asarray(q_ranges)
        self.align_multialement_data(data, new_data, q_ranges, channel_ranges )

    def _rebin_to_scale(self, data, new_data, mask, new_mask, unit):
        
        rows = len(data)
        tth = np.zeros(rows)
        bins = np.size(data[0])
        x = np.arange(bins)
   
        calibrations = self.mca.get_calibration()
    
        rebinned_scales = []

   
        if unit == 'q':
            for row in range(rows):
                calibration = calibrations[row]
                q = calibration.channel_to_q(x)
                tth[row]= calibration.two_theta
                rebinned_scales.append(q)
        elif unit == 'E':
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
        if unit == 'q':
            self.q_scale = [rebinned_step, rebinned_min]
        elif unit == 'E':
            self.E_scale = [rebinned_step, rebinned_min]
        rebinned_new = [x*rebinned_step+rebinned_min]*rows
        self.align_multialement_data(data, new_data, rebinned_scales,rebinned_new )

        self.align_multialement_data(mask, new_mask , rebinned_scales,rebinned_new ,kind='nearest')


    def get_sq_par(self, elements, fractions, choices_abc, choices_mkl):
        # for testing, always returns the sq_par for Fe

        sq_pars = []
        for i in range(len(elements)):
            sq_pars.append(self._get_sq_par(elements[i],choices_abc[i],choices_mkl[i],fractions[i]))
       
        return sq_pars
        


    def s_q(self):
        """normalize individual spectrum to be S(q) in [[qi],[Sqi],[Sqi_err]] array"""

        dataarray=self.params['dataarray']
        ttharray=self.params['ttharray']
        model_func = self.params['model_func']
        p_opt = self.params['p_opt']
        Emin = self.params['Emin']
        Emax = self.params['Emax']
        polynomial_deg = self.params['polynomial_deg']
        itr_comp = self.params['itr_comp']
        sq_par = self.params['sq_par']
        model_mre = self.params['model_mre']
        sq_smoothing_factor = self.params['sq_smoothing_factor']
        q_spacing = self.params['q_spacing']

        print(sq_par)
      
        S_q = []
        #tth_used = []
        for i in range(len(dataarray)):
            
            xi = []; yi = []; y_primary = []
            xi = dataarray[i][0]
            yi = dataarray[i][1]
            
            Emin_indx = (np.abs(xi-Emin)).argmin() # find the nearest index to Emin
            Emax_indx = (np.abs(xi-Emax)).argmin() # find the nearest index to Emax
            xn = xi[Emin_indx:Emax_indx]
            yn = yi[Emin_indx:Emax_indx]
            tth = ttharray[i]
            qi = 4*np.pi/(12.3984/xn)*np.sin(np.radians(tth/2.0))
            xnc = xn-2.4263e-2*(1-np.cos(np.radians(tth/2))) # E' for Compton source
            qic = 4*np.pi/(12.3984/xnc)*np.sin(np.radians(tth/2)) # q' for the Compton source

            print('tth '+ str(tth))
            print('xn '+ str(xn[0]) + ' ... '+str(xn[-1]))
            print('xpc '+ str(xnc[0]) + ' ... '+str(xnc[-1]))
            print('qi '+ str(qi[0]) + ' ... '+str(qi[-1]))
            print('qic '+ str(qic[0]) + ' ... '+str(qic[-1]))
            mean_fqsquare,mean_fq,mean_I_inc = I_base_calc(qi,qic,sq_par)
            y_primary = model_func(xn,*p_opt)
            Iq_base = mean_fqsquare + mean_I_inc
            s = (Iq_base*y_primary).mean()/yn.mean()
            sqi = (s*yn-Iq_base*y_primary)/y_primary/mean_fq**2 + 1.0
            sqi_err = s*yn/y_primary/mean_fq**2*np.sqrt(1.0/yn+(model_mre/y_primary)**2)
            S_q.append([qi,sqi,sqi_err])
            #tth_used.append(tth)
            
        self.out_params['S_q_fragments'] = S_q = np.array(S_q)
        #self.out_params['tth']
        
        # find consequencial scale               
        for i in range(len(S_q)):
            if (len(S_q)-1-i) >= 1:
                data1 = S_q[len(S_q)-1-i]
                data2 = S_q[len(S_q)-2-i]
                
                q1 = np.abs(data1[0][0]-data2[0]).argmin()
                q2 = np.abs(data1[0]-data2[0][-1]).argmin()
                y1_mean = data1[1][0:q2].mean()
                y2_mean = data2[1][q1:].mean()
                s = y1_mean/y2_mean
                S_q[len(S_q)-2-i][1] = s*S_q[len(S_q)-2-i][1] # scale I(Q)
                S_q[len(S_q)-2-i][2] = s*S_q[len(S_q)-2-i][2] # scale I_err(Q)
        
        
        # respace and smooth the merged S(q) data using UnivariateSpine fucntion
        q_all = []; q_sort =[]
        S_q_all = []; sq_sort =[]
        S_q_err_all = []; sq_sort_err =[]
        
        # combine all data
        for i in range(len(S_q)):
            q_all += list(S_q[i][0][:])
            S_q_all += list(S_q[i][1][:])
            S_q_err_all += list(S_q[i][2][:])
        
        # sort in q
        sort_index = np.argsort(q_all)
        for j in sort_index:
            q_sort.append(q_all[j])
            sq_sort.append(S_q_all[j])
            sq_sort_err.append(S_q_err_all[j])
        
        q_sort = np.array(q_sort)
        sq_sort = np.array(sq_sort)
        sq_sort_err = np.array(sq_sort_err)
        
        # make evenly spaced [q,sq,sq_err] array using spline interpolation
        weight = sq_smoothing_factor/sq_sort_err
        spl = interpolate.UnivariateSpline(
            q_sort,sq_sort,w=weight,bbox=[None,None],k=3,s=None)
        q_even = np.arange(q_sort[0],q_sort[-1],q_spacing) # evenly spaced q
        sq_even = spl(q_even) # evenly spaced I(q)
        # estimate the root mean squre error for each spline smoothed point
        sq_even_err = []
        for i in range(len(q_even)):
            q_box_min = q_even[i]-0.5*q_spacing
            q_box_max = q_even[i]+0.5*q_spacing
            indxb = np.abs(q_box_min-q_sort).argmin()
            indxe = np.abs(q_box_max-q_sort).argmin()
            if indxe-indxb == 0:
                sq_even_err.append(sq_sort_err[indxb])
            else:
                # mean square error from the original data
                mserr = sum([err**2 for err in sq_sort_err[indxb:indxe]])/len(sq_sort_err[indxb:indxe])
                # mean square residaul for the spline fit at original data points
                msres = sum((spl(q_sort[indxb:indxe])-sq_sort[indxb:indxe])**2)/len(sq_sort[indxb:indxe])
                rmserr = np.sqrt(mserr+msres) # geometric average of the errors
                sq_even_err.append(rmserr)
        sq_even_err = np.array(sq_even_err)    
        
     

        self.out_params['q_even'] = q_even
        self.out_params['sq_even'] = sq_even
        self.out_params['sq_even_err'] = sq_even_err
        
    
  
    
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
        #x = np.arange(bins)
        for row in range(rows): 
            x = old_scales[row]
            xnew = new_scales[row]
            new_data[row] = self.shift_row(data[row],x, xnew, kind)
        

    def shift_row(self, row,x, xnew, kind='linear'):
        f = interpolate.interp1d(x, row, assume_sorted=True, bounds_error=False, fill_value=0, kind=kind)
        row = f(xnew)
        return row

    def _get_sq_par(self, atom, choice_abc, choice_mkl, fract):
        
        opt_MKL = None
        opt_abc = None
        opt_ab5 = None

        # selecting abc, abc table offers several oxidation state options 
        # for some elements, in more than one choice exists then user 
        # selects which one to use.
        options_abc=self.ap.get_abc_options(atom)
        opt_abc = options_abc[choice_abc][1:]
        
        #selecting MKL
        options_MKL=self.ap.get_MKL_options(atom)
        opt_MKL = options_MKL[choice_mkl][1:-1]
            

        # ab5 contains parameters for calculating incoherent scattering in different form than legacy MKL table
        # from Ref: H.H.M. Balyuzi, Acta Cryst. (175). A31, 600
        ab5 = self.ap.get_ab5_options(atom)[atom]
        opt_ab5=ab5[1:11]
        Z = ab5[0]
        
        part1 = int(Z), fract
        part2 = list(opt_abc)
        part3 = list(opt_MKL)
        part4 = list(opt_ab5)
        sq_par = [*part1] + part2 + part3 + part4
        return sq_par

    @staticmethod
    def find_non_zero_range(a):
        # Find the indices of the non-zero elements
        nonzero_indices = np.flatnonzero(a)
        # Find the index of the maximum value among the non-zero indices
        last_nonzero_index = np.amax(nonzero_indices)
        # Find the index of the minimum value among the non-zero indices
        first_nonzero_index = np.amin(nonzero_indices)
        return first_nonzero_index, last_nonzero_index