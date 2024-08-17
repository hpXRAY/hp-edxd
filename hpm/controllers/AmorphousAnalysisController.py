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


from functools import partial
#from platform import java_ver
import pyqtgraph as pg
import copy
import numpy as np
import os
from PyQt6 import QtWidgets, QtCore
from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight
from hpm.widgets.UtilityWidgets import save_file_dialog, open_file_dialog, open_files_dialog, open_folder_dialog
from hpm.widgets.AmorphousAnalysisWidget import AmorphousAnalysisWidget
from hpm.models.AmorphousAnalysisModel import AmorphousAnalysisModel
from PyQt6.QtCore import pyqtSignal, QObject
import natsort
from hpm.controllers.DisplayPrefsController import DisplayPreferences
from hpm.models.mcaModel import MCA
from hpm.controllers.MaskController import MaskController
from hpm.models.MaskModel import MaskModel
import time


class AmorphousAnalysisController(QObject):
    file_changed_signal = pyqtSignal(str)  
    element_changed_signal = pyqtSignal(int)
    channel_changed_signal = pyqtSignal(float)  
    add_rois_signal = pyqtSignal(list)

    def __init__(self, file_save_controller, directories):
        super().__init__()
        self.file_save_controller = file_save_controller
        self.directories = directories
        
        
      
        self.model = AmorphousAnalysisModel( )
        self.widget = AmorphousAnalysisWidget(directories)

        for step in self.model.steps:
            s = self.model.steps[step]
            dims = s.get_data_out_dims()
            name = s.name
            mask = s.params_in['mask']
            
            plot = self.widget.add_scratch_plot(name, dims, mask)
            if mask:
                s.set_param({'mask_model': self.widget.mask_controllers[name].mask_model})
            
            s.updated.connect(partial( self._update_output_plot, s))
            

        self.folder = ''
        self.active = False
        self.selectedENV = 0
        self.selectedENV_persist = 0
        self.envLen = 0
        self.single_file = False

        self.scale = 'Channel'
        self.row_scale = 'Index'
        self.file = ''
        self.row = 0
       

        self.create_signals()

    def _update_output_plot(self, step):
        
        dims = step.get_data_out_dims()
        mask = step.params_in['mask']
        data_out = step.params_out['data_out']
        name = step.name

        if dims == 2 and mask == True:
            mask_controller = self.widget.mask_controllers[name]
            if len(data_out):
                
                mask_controller.mask_model._img_data = data_out
                mask_controller.update_mask_dimension()
                mask_controller.update_gui()
        else:
            plot = self.widget.scratch_plots[name]
            plot.plot_image(data_out)
        
    def create_signals(self):
       
        self.widget.step1_btn.clicked.connect(self.step1_callback)
        self.widget.step2_btn.clicked.connect(self.step2_callback)
        self.widget.step3_btn.clicked.connect(self.step3_callback)
        self.widget.step4_btn.clicked.connect(self.step4_callback)
        self.widget.step5_btn.clicked.connect(self.step5_callback)
        self.widget.widget_closed.connect(self.view_closed)


    def set_models(self, multi_spectra_model ):

        mca = multi_spectra_model.mca
        self.model.set_models(mca, multi_spectra_model)

       
        self.multispectra_loaded()


    def HorzScaleRadioToggle(self,horzScale):
        
        self.set_unit(horzScale)


    def set_row_scale(self, label='Index'):
        d = [1,0]
        row_scale = 'Spectrum index'

        if label == 'tth':
            row_scale=f'2\N{GREEK SMALL LETTER THETA}'
            d = self.model.tth_scale

        self.widget.set_image_row_scale(row_scale, d)
        self.row_scale = row_scale

    
    def multispectra_loaded(self):
    
        # do something if mca has new data
        pass
        

    def step1_callback(self):
        now = time.time()
        self.model.calculate_1()
        later = time.time()
        elapsed = later - now
        print(elapsed)

    def step2_callback(self):
        now = time.time()
        self.model.calculate_2()
        later = time.time()
        elapsed = later - now
        print(elapsed)
    
    def step3_callback(self):
        now = time.time()
        self.model.calculate_3()
        later = time.time()
        elapsed = later - now
        print(elapsed)

    def step4_callback(self):
        now = time.time()
        self.model.calculate_4()
        later = time.time()
        elapsed = later - now
        print(elapsed)

    def step5_callback(self):
        now = time.time()
        self.model.calculate_5()
        later = time.time()
        elapsed = later - now
        print(elapsed)
       
    def show_view(self):
        self.active = True
        #self.update_envs()
        self.widget.raise_widget()
        
        
    def view_closed(self):
        self.active = False
       

