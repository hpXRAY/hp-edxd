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

# Based on code from Dioptas - GUI program for fast processing of 2D X-ray diffraction data

""" 


"""

import numpy as np

from PyQt6 import QtCore
from hpm.models.jcpds import jcpds
from hpm.models.cif import CifConverter
from utilities.HelperModule import calculate_color
from hpm.models.pressure.LatticeRefinement import latticeRefinement

class RefinedLattice():
    def __init__(self, phase_name):
        self.lattice_model = latticeRefinement()
        self.reflections = {}
        self.use = []
        self.phase_name = phase_name
        self.phase = None
        
        self.ddiff = []
        self.dobs = []
        self.dcalc = []
        self.P = 0
        self.V = 0
        self.V_esd = 0
        self.refined_lattice = {}
        self.refined_lattice_esd = {}

    def get_reflections(self):
        reflections = []
        for rlr in self.reflections:
            reflections.append(self.reflections [rlr])

        return reflections

class LatticeRefinementModel(QtCore.QObject):
    phase_added = QtCore.pyqtSignal()
    phase_removed = QtCore.pyqtSignal(int)  # phase ind
    phase_changed = QtCore.pyqtSignal(int)  # phase ind
    phase_reloaded = QtCore.pyqtSignal(int)  # phase ind

    reflection_added = QtCore.pyqtSignal(int)
    reflection_deleted = QtCore.pyqtSignal(int, int)  # phase index, reflection index

    num_phases = 0

    def __init__(self):
        super().__init__()
       
    
        self.active = False

        self.refined_lattice_models = {}
       
    def set_reflections(self, reflections):
        
        self.reflections = []
        for r in reflections:
            if r.fit_ok:
                self.reflections.append(r)
        self.update_phases()

    def set_phases(self, phases):
        for phase in phases:
            if not phase in self.refined_lattice_models:
                self.refined_lattice_models[phase ] = RefinedLattice(phase)
            
            self.refined_lattice_models[phase].phase = phases[phase]
        

    def update_phases(self):
        reflections = self.reflections
      
        if len(reflections)>0:
            
            phases = []
            for r in reflections:
                l = r.label.split(' ')[0]
                if len(l)>1:
                    if not l in phases:
                        phases.append(l)

            for p in phases:

                if p in self.refined_lattice_models:
                    self.refined_lattice_models[p].reflections = {}
                else:
                    self.refined_lattice_models[p] = RefinedLattice(l)
                
            for r in reflections:
                l = r.label.split(' ')[0]
                if len(l)>1:
                    reflections = self.refined_lattice_models[l].reflections
                    label = r.label
                    reflections[label] = r


    def get_phases(self):
        return list(self.refined_lattice_models.keys())
                    
    def clear(self):
        self.__init__()
            
    def refine_phase(self, p):
        lattice_out = []

        refined_lattice_p = self.refined_lattice_models[p]
        
        if p in self.refined_lattice_models and refined_lattice_p.phase != None:
            
            if p !='':
                curr_phase = refined_lattice_p.phase
          
                DHKL = []
                reflections = refined_lattice_p.get_reflections()
                use = refined_lattice_p.use
                for i, r in enumerate(reflections):
                    u = use[i] 
                    if u:
                        d = r.d_spacing
                        hkl = r.label.split(' ')[1]
                        if not len(hkl)==3 or not hkl.isdigit():
                            break
                        h = int(hkl[0])
                        k = int(hkl[1])
                        l = int(hkl[2])
                        dhkl = [d,h,k,l]
                        DHKL.append(dhkl)
                if len(DHKL)>0:
                    lattice_model = refined_lattice_p.lattice_model
                    lattice_model.set_dhkl(DHKL)
                    symmetry = curr_phase.params['symmetry']
                    lattice_model.set_symmetry(symmetry.lower())
                    lattice_model.refine()
                    volume_out, volume_esd_out = lattice_model.get_volume()
                    lattice_out = lattice_model.get_lattice()
                    lattice_esd_out = lattice_model.get_lattice_esd()
                    
                    refined_lattice_p.P = p
                    refined_lattice_p.V = volume_out
                    refined_lattice_p.V_esd = volume_esd_out
                    refined_lattice_p.refined_lattice = lattice_out
                    refined_lattice_p.refined_lattice_esd = lattice_esd_out
                    
                    DCalc = lattice_model.refinement_output['Dcalc']

                    refined_lattice_p.ddiff = []
                    refined_lattice_p.dobs = []
                    refined_lattice_p.dcalc = []

                    for i, dcalc in enumerate(DCalc):
                        d = dcalc 
                        dobs = DHKL[i][0]
                        ddiff = dobs - d
                        refined_lattice_p.ddiff.append(ddiff)
                        refined_lattice_p.dobs.append(dobs)
                        refined_lattice_p.dcalc.append(d)

                      
                        
                        
     

