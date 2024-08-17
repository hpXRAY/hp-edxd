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
from PyQt6 import QtWidgets, QtCore
import copy
import pyqtgraph as pg
from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight, HorizontalLine
from axd.models.aEDXD_atomic_parameters import aEDXDAtomicParameters
from axd.widgets.aEDXD_atom_options_widget import aEDXDAtomWidget
import numpy as np

from utilities.PeriodicTable import PeriodicTableDialog

class aEDXDAtomController(QtCore.QObject):

    
    apply_clicked_signal = QtCore.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        
        self.atom_window = aEDXDAtomWidget()
        self.ap = aEDXDAtomicParameters()

        self.available_elements = self.get_available_elements()
        self.sq_pars = []
        self.make_connections()
        
    def get_available_elements(self):
        MKL = self.ap.MKL['str_data']
        abc = self.ap.abc['str_data']
        ab5 = self.ap.ab5['str_data']
        el_abc =[]
        for row in ab5:
            el_abc.append(row)

        el_abc = self.remove_duplicates(el_abc)
        set1 = {*el_abc}  
        set2 = {*ab5}  
        
    
        # union of two sets 
        intersection = set1.intersection(set2)
        return intersection

    def remove_duplicates(self, x):
        return list(dict.fromkeys(x))

    def make_connections(self):
        self.atom_window.apply_btn.clicked.connect(self.apply)
        self.atom_window.add_btn.clicked.connect(self.add_atom_clicked)
        self.atom_window.delete_btn.clicked.connect(self.delete_atom_clicked)
        self.atom_window.clear_btn.clicked.connect(self.clear_atom_clicked)
        self.atom_window.fract_item_changed_signal.connect(self.edit_fract)

    def show_atoms(self):
        self.atom_window.raise_widget() 

    def add_atom_clicked(self):
        fract = None
        opt_MKL = None
        opt_abc = None
        opt_ab5 = None
        opt_abc_note = None
        opt_MKL_note = None
        Z = None

        atom = PeriodicTableDialog.showDialog(self.available_elements) 

        if atom == '':
            return

        # selecting abc, abc table offers several oxidation state options 
        # for some elements, in more than one choice exists then user 
        # selects which one to use.
        options_abc=self.ap.get_abc_options(atom)
        opts_abc = []
        choice = None
        for o in options_abc:
            opts_abc.append(o)
        
        if len(opts_abc)>1:
            selected, ok = QtWidgets.QInputDialog.getItem(
            self.atom_window, 'Scattering factor options', 
            'Choose option for scattering factor data: ', opts_abc, 0, False)
            if ok:
                choice = selected
            else:
                # can't continue if user cancelled this step
                return
        elif len(opts_abc)==1:
            choice = opts_abc[0]
        opt_abc = options_abc[choice][1:]
        
        #selecting MKL
        options_MKL=self.ap.get_MKL_options(atom)
        opts_MKL = []
        choice = None
        for o in options_MKL:
            opts_MKL.append(o)
        if len(opts_MKL)>1:
            selected, ok = QtWidgets.QInputDialog.getItem(
            self.atom_window, 'Incoherent scattering options', 
            'Choose option for incoherent scattering data: ', opts_MKL, 0, False)
            if ok:
                choice = selected
        elif len(opts_MKL)==1:
            choice = 0
        if opt_MKL is None:
            # user cancelled dialog, or no data exists in MKL table
            # this is okay, in this case data from ab5 table can still be used
            opt_MKL = [None]*3
        else:
            opt_MKL = options_MKL[choice][1:-1]
            
        #selecting fraction
        d, okPressed = QtWidgets.QInputDialog.getDouble(
                    self.atom_window, "Molar fraction","Enter molar fraction: (0 to 1)", 0.5, 0, 1, 4)
        if okPressed:
            fract = d
        else:
            # can't continue if user cancelled fraction choice
            return

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
        self.set_param(sq_par)
                
    def delete_atom_clicked(self):
        ind = self.atom_window.get_selected_atom_row()
        if ind > -1:
            self.atom_window.del_atom(ind)
            del self.sq_pars[ind]

    def clear_atom_clicked(self):
        self.sq_pars= []
        self.clear_widget()

    def clear_widget(self):
        while self.atom_window.atom_tw.rowCount()>0:
            self.atom_window.del_atom(0)

    def edit_fract(self,ind,fract):
        sp= self.sq_pars[ind]
        sp[1]=fract
        self.sq_pars[ind]=sp

    def apply(self):
        par = np.asarray(self.sq_pars)
        params = {}
        params['sq_par']=par
        
        self.apply_clicked_signal.emit(params)
    
    def set_params(self,params):
        sq_par = params['sq_par']
        if sq_par is not None:
            self.sq_pars = []
            self.clear_widget()
            for sp in sq_par:
                self.set_param(sp)
            
    def set_param(self, param):
        atom, atom_abc_note, atom_MKL_note, frac= \
                            self.ap.lookup_atom_by_sq_par(param)
        self.sq_pars.append(param)
        self.atom_window.add_atom(True, atom, 
                        atom_abc_note + ' / ' + atom_MKL_note, 
                        frac, len(self.sq_pars))
