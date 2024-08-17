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
import hpm.models.jcpds as jcpds
import copy
import utilities.centroid
import numpy as np
import functools, math
import os
from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight
from hpm.widgets.UtilityWidgets import save_file_dialog, open_file_dialog, open_files_dialog
from hpm.widgets.RoiWidget import RoiWidget, plotFitWindow
from hpm.widgets.LatticeRefienementWidget import LatticeRefinementWidget
from hpm.models.mcaModel import  McaROI
from hpm.models.LatticeRefinementModel import LatticeRefinementModel
from PyQt6.QtCore import pyqtSignal, QObject

import pyqtgraph as pg

class LatticeRefinementController(QObject):

    refined_pressure_updated = pyqtSignal(float)
    

    def __init__(self, mcaModel, plotWidget, plotController, mainController):
        super().__init__()

        self.phases_directory = mainController.working_directories.phase
        
        self.widget = LatticeRefinementWidget()
        self.set_mca(mcaModel)
        self.mcaController = mainController
        
        self.model = LatticeRefinementModel()
        self.P = 0

        detector = 0
        self.ddiff =[]
       
    
        self.dataLen = self.mca.nchans
        self.pattern_widget = plotWidget
        self.plotController = plotController
  
        self.nreflections = 0

        '''self.unit = 'E'
        self.unit_ = 'KeV'''

        self.selected_phase = ''

        self.active = False
        
        self.create_signals()

    def set_mca(self, mca, element=0):
        self.mca = mca
        self.dataLen = self.mca.nchans
        self.calibration = self.mca.get_calibration()[element]
        self.two_theta =  self.calibration.two_theta
        if type(self.two_theta) == type(float()):
            self.widget.two_theta.setText(str(round(self.two_theta,5))+f'\N{DEGREE SIGN}')



    def set_jcpds_directory(self, directory):
        
        self.phases_directory = directory


    def get_calibration(self):
        return self.calibration
        
    def create_signals(self):
        
        self.widget.do_fit.clicked.connect(self.menu_do_fit)
        self.widget.auto_fit.toggled.connect(self.auto_fit_callback)
        self.widget.plot_cal.clicked.connect(self.menu_plot_refinement)
        self.widget.phases_cbx.activated.connect(self.phases_cbx_callback)

    def phases_cbx_callback(self, *args):
        selected_ind = args[0]
        phase = self.widget.phases_cbx.itemText(selected_ind)
        if phase != self.selected_phase:
            self.selected_phase = phase
            reflections = self.model.refined_lattice_models[phase].get_reflections()
            self.widget.roi_tw.clear_reflections()
            self.output_clear()
            self.widget.roi_tw. set_reflections(reflections)
         
       
    def pressure(self):
        self.show_view()

    def show_view(self):
        self.active = True
        self.widget.raise_widget()

    def view_closed(self):
        self.active = False

    def auto_fit_callback(self, state):
        if state:
            self.update_phases()

    def menu_do_fit(self):
       
        self.update_phases()

    def update_phases(self):
        self.model.update_phases()
        selected_phase = self.selected_phase
        refined_lattice= self.model.refined_lattice_models[selected_phase]
        use = self.widget.roi_tw. get_use()
        refined_lattice.use = use
        if self.model.refined_lattice_models[selected_phase].phase != None:
            self.model.refine_phase(selected_phase)

            DCalc = refined_lattice.dcalc
            if len(DCalc):
                for i, dcalc in enumerate(DCalc):
                
                    ddiff = refined_lattice.ddiff[i]
                    self.widget.roi_tw.update_roi(i,round(dcalc, 4),round(ddiff,4))

                p = refined_lattice.P
                lattice_out = refined_lattice.refined_lattice
                lattice_esd_out = refined_lattice.refined_lattice_esd
                volume_out = refined_lattice.V

                self.update_output(selected_phase, refined_lattice)
        else:
            
            self.output_clear()

    def output_clear(self):
        self.widget.model_lbl.setText('')
        self.widget.parameter_widget.clear()

    def clear_reflections(self, *args, **kwargs):
        """
        Deletes all reflections from the GUI and model
        """
        self.model.clear()
        self.widget.roi_tw.clear_reflections()
      
        
        self.ddiff =[]
        self.reflection = []
        
     
    def update_reflections(self):
        pass


    def menu_plot_refinement(self):
        """ Private method """
        
        if self.selected_phase in self.model.refined_lattice_models:
            refined_lattice_p = self.model.refined_lattice_models[self.selected_phase]
            E_diff = refined_lattice_p.ddiff
            if len (E_diff):
                energy_use = refined_lattice_p.dobs
                E_diff_use = E_diff
            
                
                pltError = pg.plot(energy_use,E_diff_use, 
                        pen=(200,200,200), symbolBrush=(255,0,0),antialias=True, 
                        symbolPen='w', title= f'\N{GREEK CAPITAL LETTER DELTA} d'
                )
                pltError.setLabel('left', f'\N{GREEK CAPITAL LETTER DELTA} d '+ f'\N{LATIN CAPITAL LETTER A WITH RING ABOVE}')
                pltError.setLabel('bottom', 'd '+f'\N{LATIN CAPITAL LETTER A WITH RING ABOVE}')


    def set_reflections_and_phases(self, reflections, phases):
        self.clear_reflections()
      

        self.model.set_reflections( reflections)
        self.model.set_phases( phases)

        reflection_groups = self.model. get_phases()
        self.widget.phases_cbx.blockSignals(True)
        self.widget.phases_cbx.clear()
        self.widget.phases_cbx.addItems(reflection_groups)
        self.widget.phases_cbx.blockSignals(False)

        if not self.selected_phase in reflection_groups:
            if len(reflection_groups):
                self.selected_phase = reflection_groups[0]
            else:
                self.selected_phase = ''
        
        
        if len(self.selected_phase):
            show_reflections= self.model.refined_lattice_models[self.selected_phase].get_reflections()
            self.widget.roi_tw.set_reflections(show_reflections)

   
    
            

    def update_output(self,phase, refined_lattice):
        
        lattice = refined_lattice.refined_lattice
        lattice_esd = refined_lattice.refined_lattice_esd
        V = refined_lattice.V
        V_esd = refined_lattice.V_esd

        
        curr_phase = self.model.refined_lattice_models[phase].phase
        v0 = curr_phase.params['v0']
        v_over_v0 = V/v0
        v_over_v0_esd = V_esd/v0

        symmetry = curr_phase.params['symmetry']
        if v_over_v0 > 0.3 and v_over_v0 < 2:

            # calculate P esd using a local derivative method
            curr_phase.compute_pressure(volume = V)
            
            P = curr_phase.params['pressure']
            curr_phase.compute_pressure(volume = V-V_esd/2)
            P_min = curr_phase.params['pressure']
            curr_phase.compute_pressure(volume = V+V_esd/2)
            P_max = curr_phase.params['pressure']
            P_esd = abs(P_max - P_min)
            
        else:
            P = np.nan
            P_esd = np.nan
            

        self.P = P
        T = curr_phase.params['temperature']
        self.widget.model_lbl.setText('Symmetry: '+ symmetry)
        self.widget.parameter_widget.update_output(lattice,lattice_esd, P,V,T, v_over_v0, P_esd, V_esd, v_over_v0_esd)
        
        

        autosend = self.widget.auto_pressure_btn.isChecked()
        if autosend:
            self.refined_pressure_updated.emit(P)
       
       