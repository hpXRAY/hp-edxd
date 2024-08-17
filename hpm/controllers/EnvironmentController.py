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
import copy
import numpy as np
import os
from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight
from hpm.widgets.UtilityWidgets import save_file_dialog, open_file_dialog, open_files_dialog
from hpm.widgets.EvironmentWidget import EnvironmentWidget

from PyQt6.QtCore import pyqtSignal, QObject

class EnvironmentController(QObject):

    env_updated_signal = pyqtSignal(int, str )  
    env_selection_changed_signal = pyqtSignal(int,str)
    #envs_changed = pyqtSignal()  

    def __init__(self):
        super().__init__()
        
        self.env = []
        self.environment_widget = EnvironmentWidget()
       
        self.active = False
        self.selectedENV = 0
        self.selectedENV_persist = 0
        self.envLen = 0

        #self.phases =dict()
        self.create_signals()
        
    def create_signals(self):
       
        self.environment_widget.widget_closed.connect(self.view_closed)
        self.environment_widget.env_tw.currentCellChanged.connect(self.env_selection_changed)
    
        self.environment_widget.key_signal.connect(self.key_sig_callback)

    def connect_click_function(self, emitter, function):
        emitter.clicked.connect(function)      

  
    def key_sig_callback(self, sig):
        if sig == 'delete' :
            self.remove_btn_click_callback()

    def set_environment(self, environment):
        self.update_env(environment)

    def env_selection_changed(self, row, **kwargs):
        cur_ind = row
        self.selectedENV = row
        
        if cur_ind >-1 and cur_ind < len(self.env): 
            out = self.env[cur_ind].description
               
        else: out = ''
        
        self.env_selection_changed_signal.emit(cur_ind, out)
        

    def show_view(self):
        self.active = True
        
        self.environment_widget.raise_widget()
        
        
    def view_closed(self):
        self.active = False
       

    def update_env(self, environment):
        oldLen =  copy.copy(self.envLen)
        self.selectedENV_persist = copy.copy(self.selectedENV)
       
        self.env = environment
  
        newLen = len(self.env)
        self.nenvs = len(self.env)
        self.blockSignals(True)
        self.environment_widget.blockSignals(True)
        if newLen == oldLen:
            for i, r in enumerate(self.env):
                #index = self.env.index(r)
                
                self.update_env_by_ind(r.name, r.value, r.description, i)
        else:
            while self.environment_widget.env_tw.rowCount() > 0:
                self.environment_widget.del_env(self.environment_widget.env_tw.rowCount()-1,silent=True)
            for r in self.env:
             
                self.environment_widget.add_env(r.name, r.value, r.description, silent=True)
        self.blockSignals(False)              
        self.environment_widget.blockSignals(False)    
        self.environment_widget.env_tw.blockSignals(False)      
        if self.selectedENV_persist<len(self.env):
            sel = np.clip(self.selectedENV_persist,0,31)
        else: 
            sel = len(self.env)-1
        self.environment_widget.select_env(sel) 
        self.selectedENV=sel
    
        self.envLen = copy.copy(self.nenvs)
  
    def env_removed(self, ind):
        self.environment_widget.del_env(ind)
        

    def clear_envs(self, *args, **kwargs):
        """
        Deletes all envs from the GUI
        """
        self.blockSignals(True)
        while self.environment_widget.env_tw.rowCount() > 0:
            self.env_removed(self.environment_widget.env_tw.rowCount()-1)
        
        self.envLen = 0
        self.blockSignals(False)
       
                
    def update_env_by_ind(self, pv, value, description, ind):
        self.environment_widget.update_env(pv, value, description, ind)

    def env_name_changed(self, ind, name):
        self.environment_widget.rename_env(ind, name)


