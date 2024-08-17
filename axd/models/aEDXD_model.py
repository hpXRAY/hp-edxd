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


###############################################################################
# Changes in v1.2 
#   1. separate class file from functions (5/16/13)
#   2. add class_version() (5/16/13)
#   3. add class MCAData(object) (5/16/13)
# Changes in v1.3
#   1. add options to choose input file format in "init" class
#   2. fix a bug that does not sum the intensities from multiple input files
#   3. work with:
#                Python 2.7.9
#                Numpy 1.9.2
#                Scipy 0.15.1
#                matplotlib 1.4.3
###############################################################################

import os
import time
import sys
import copy
import json
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from utilities.hpMCAutilities import readconfig
from axd.models.aEDXD_components import primaryBeam, structureFactor, Pdf, PdfInverse
from .. import data_path

class aEDXD_model(QObject):

    primary_beam_updated = pyqtSignal()
    structure_factor_updated = pyqtSignal()
    G_r_updated = pyqtSignal()
    Sf_filtered_updated = pyqtSignal()

    config_file_set = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        # create variables
        
        self.config_file = os.path.join(data_path, "aEDXD_defaults.cfg")
        params = ['inputdataformat',
                'inputdatadirectory',
                'outputsavedirectory',
                'bin_size',
                'mca_adc_shapingtime',
                'sq_smoothing_factor',
                'q_spacing',
                'r_spacing',
                'qmax',
                'rmax',
                'hard_core_limit',
                'itr_comp',
                'rho',
                'polynomial_deg',
                'Emax',
                'Emin',
                'sq_par',
                'mcadata',
                'E_cut']
        self.params = dict.fromkeys(params)
        self.dataarray = []
        self.ttharray = []
        self.cofigure_from_file()
        self.params['mca_adc_shapingtime'] = 4e-6
        self.params['E_cut'] = []
        self.params['mcadata'] = []
        # create sub models
        self.primary_beam = primaryBeam()
        self.structure_factor = structureFactor()
        self.pdf_object = Pdf()
        self.pdf_inverse_object = PdfInverse()

    def set_config_file(self,filename):
        self.config_file = filename
        self.config_file_set.emit(self.config_file)

    def set_params(self, params):  # used by config controller
        mp = self.params
        for p in params:
            
            par =params[p]
            typ = type(par).__name__
            if typ == 'float':
                par = round(par,15)  # avoid weird python rounding bug
            mp[p]=par

    def isCongigured(self):
        mp = self.params
        configured = True
        for p in mp:
            if mp[p] is None:

                configured = False
                break
            if p== 'sq_par':
                if len(mp[p]) == 0 :
                    configured = False
            
        return configured

    def cofigure_from_file(self):  # read config file
        config_dict = readconfig(self.config_file)
        # set the initial values from the default config dictionary
        self.configure_from_dict(config_dict)
    
    def configure_from_dict(self, config_dict):
        params = list(self.params.keys())
        for p in params:
            if p in config_dict:
                par =config_dict[p]
                typ = type(par).__name__
                if typ == 'float':
                    par = round(par,15)  # avoid weird python rounding bug
                self.params[p]=par
            else:
                self.params[p] = None
        
        if 'mcadata_use' in config_dict:
            # mcadata_use parameter is treated separately to maintain backward compatibility with older aEDXD versions
            # mcadata_use it is not a required parameter
            self.params['mcadata_use'] = config_dict['mcadata_use']
        else:
            self.params['mcadata_use'] = []

        if not 'E_cut' in self.params:
            # mcadata_use parameter is treated separately to maintain backward compatibility with older aEDXD versions
            # mcadata_use it is not a required parameter
            self.params['E_cut'] = []
        else:
            if self.params['E_cut'] is None:
                self.params['E_cut'] = []
        # flag analysis status 
        self.primary_done = False
        self.sf_normalization_done = False
        self.pdf_done = False
        self.configured = self.isCongigured()
        
    def primary(self):
        """primary beam analysis"""
        if len(self.ttharray):
            pb = self.primary_beam
            x = self.dataarray[-1][0]
            
            y = self.dataarray[-1][1]   
            tth = self.ttharray[-1]
            data = {'x':x,'y':y,'thh':tth}
            config = {**self.params, **data}
            pb.set_config(config)
            pb.set_auto_process(True)
        self.primary_beam_updated.emit()
    
    def sf_normalization(self):
        """normalize individual spectrum to be S(q) in [[qi],[Sqi],[Sqi_err]] array"""
        if len(self.ttharray):
            sf = self.structure_factor
            data = {'dataarray':self.dataarray, 'ttharray':self.ttharray}
            config = {**self.params, **data, **self.primary_beam.out_params}
            sf.set_config(config)
            sf.set_auto_process(True)
        self.structure_factor_updated.emit()
    
    def pdf(self):
        """calculate G(r) and Inverse Fourier Filtered S(q)"""
        if len(self.ttharray):
            pdf = self.pdf_object
            config = {**self.params , **self.structure_factor.out_params}
            pdf.set_config(config)
            pdf.set_auto_process(True)
        self.G_r_updated.emit()

    def pdf_inverse(self):
        """calculate G(r) and Inverse Fourier Filtered S(q)"""
        if len(self.ttharray):
            pdf_i = self.pdf_inverse_object
            config = {**self.params , **self.pdf_object.out_params}
            pdf_i.set_config(config)
            pdf_i.set_auto_process(True)
        self.Sf_filtered_updated.emit()


    

