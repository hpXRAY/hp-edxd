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


import os

import numpy as np
from PyQt6 import QtWidgets, QtCore
import copy

from hpm.models.hklGenModel import hklGenModel_viewModel

#from utilities.HelperModule import get_base_name
#from hpm.controllers.JcpdsEditorController import JcpdsEditorController
#from hpm.widgets.UtilityWidgets import save_file_dialog, open_file_dialog, open_files_dialog, CifConversionParametersDialog
#from hpm.models.PhaseModel import PhaseModel
from hpm.widgets.hklGenWidget import hklGenWidget
from hpm.controllers.hklCellInPatternController import hklCellInPatternController

import utilities.hpMCAutilities as mcaUtil


class hklGenController():
    """
     
    """

    def __init__(self, plotWidget, mcaModel, plotController, roiController):


        self.pattern = mcaModel
        self.model = hklGenModel_viewModel()

        self.hklGen_widget = hklGenWidget(self.model)
        self.hkl_cell_in_pattern_controller = hklCellInPatternController(plotController,self, plotWidget,self.model)
        self.active = False
        self.current_cell = None
        self.create_connections()
        self.tth = self.getTth()

    def create_connections(self):
        self.hklGen_widget.add_btn.clicked.connect(self.add_btn_callback)
        self.hklGen_widget.delete_btn.clicked.connect(self.delete_btn_callback)
        self.hklGen_widget.clear_btn.clicked.connect(self.clear_btn_callback)
        self.hklGen_widget.rois_btn.clicked.connect(self.rois_btn_callback)

        self.hklGen_widget.symmetry_edited_signal.connect(self.symmetry_edited_callback)
        self.hklGen_widget.spgrp_tb.editingFinished.connect(self.spgrp_tb_edited_callback)
        self.hklGen_widget.lattice_parameter_edited_signal.connect(self.lattice_parameter_edited_callback)

        self.hklGen_widget.cell_selection_edited_signal.connect(self.cell_selection_edited_callback)

        # 2th 
        self.hklGen_widget.tth_lbl.valueChanged.connect(self.tth_changed)
        self.hklGen_widget.tth_step.editingFinished.connect(self.update_tth_step)
        self.connect_click_function(self.hklGen_widget.get_tth_btn, self.get_2th_btn_callcack)

    def connect_click_function(self, emitter, function):
        emitter.clicked.connect(function)

    def cell_selection_edited_callback(self, ind):
        try:
            self.current_cell = self.model.get_cell_by_index(ind)
            self.update_cell_view()
        except IndexError:
            pass
    
    def lattice_parameter_edited_callback(self, param):
        
        print(param)
        if not self.current_cell is None:
            self.current_cell.set_lattice_param(param)
            self.update_cell_view()

    def spgrp_tb_edited_callback(self, *args, **kwargs):
        spgrp = self.hklGen_widget.spgrp_tb.text()
        if not self.current_cell is None:
            self.current_cell.set_spacegroup(spgrp)
            self.update_cell_view()

    def update_cell_view(self):
        enabled_params = self.current_cell.get_params_enabled()
        lattice_params = self.current_cell.get_lattice_params()
        self.hklGen_widget.set_cell(lattice_params, enabled_params)
        hkl_reflections = self.current_cell.get_hkl_reflections()
        
        '''
        ind = self.model.get_cell_index(self.current_cell)
        name = self.current_cell.get_spacegroup()
        self.model.set_cell_name(ind,name)
        '''
        
        hkl_str=''
        for hkl in hkl_reflections[0]:
            hkl_str=hkl_str+str(hkl[0])+' '+str(hkl[1])+' '+str(hkl[2])+' '+str(round(hkl[3],4)) +'<br>'
        self.hklGen_widget.hkl_widget.setText(hkl_str)
        

    def symmetry_edited_callback(self, symmetry):
        if not self.current_cell is None:
            #self.current_cell.set_symmetry(symmetry)
            current_symmetry = self.current_cell.get_symmetry()
            if symmetry.upper() != current_symmetry.upper():
                
                default_spacegroup = self.current_cell.default_spacegroup[symmetry.upper()]
                self.current_cell.set_spacegroup(default_spacegroup)
                self.hklGen_widget.spgrp_tb.setText(str(self.current_cell.get_spacegroup()))
            self.update_cell_view()
            

    def add_btn_callback(self):
        new_cell = self.model.add_cell()
        self.current_cell = new_cell
        new_cell.set_spacegroup(str(self.hklGen_widget.spgrp_tb.text()))
        self.update_cell_view()
        
        print('add')

    def delete_btn_callback(self, *args, **kwargs):
        """
        Deletes the currently selected Cell
        """
        print('delete')
        cur_ind = self.hklGen_widget.get_selected_row()
        if cur_ind >= 0:
            self.model.remove_cell_by_index(cur_ind)
        n_cells = self.model.get_num_of_cells()
        if n_cells:
            self.current_cell = self.model.cells[-1]
        else:
            self.current_cell = None
    

    def clear_btn_callback(self):
        """
        Deletes all tth from the GUI
        """
        while self.model.rowCount() > 0:
            self.model.remove_cell_by_index(self.model.rowCount()-1)
        self.current_cell = None
        print('clear')


    def rois_btn_callback(self):
        print('rois')
    
    def show_view(self):
        self.active = True
        self.hklGen_widget.raise_widget()


    # 2th
    def tth_changed(self):
        try:
            self.tth = np.clip(float(self.hklGen_widget.tth_lbl.text()),.001,179)
            self.hkl_cell_in_pattern_controller.tth_update(self.tth)
        except:
            pass

    def update_tth_step(self):
        value = self.hklGen_widget.tth_step.value()
        self.hklGen_widget.tth_lbl.setSingleStep(value) 

    def get_2th_btn_callcack(self):
        tth = self.getTth()
        self.hklGen_widget.tth_lbl.setValue(tth)

    def getTth(self):
        tth = self.pattern.get_calibration()[0].two_theta
        return tth