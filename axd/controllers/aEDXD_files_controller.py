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


import os.path, sys
from PyQt5 import uic, QtWidgets,QtCore, QtGui
from  PyQt5.QtWidgets import QMainWindow, QApplication, QInputDialog, QWidget, QLabel
from PyQt5.QtCore import QObject, pyqtSignal, Qt
import numpy as np
from functools import partial
import json
import copy
from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight
from pathlib import Path
from utilities.HelperModule import calculate_color
from numpy import arange
from utilities.HelperModule import getInterpolatedCounts
from axd.widgets.aEDXD_files_widget import aEDXDFilesWidget
from axd.models.aEDXD_spectra_model import Spectra, get_tth_from_file
from utilities.hpMCAutilities import readconfig
from axd.controllers.aEDXD_peak_cut_controller import aEDXDPeakCutController
from hpm.widgets.UtilityWidgets import save_file_dialog, open_file_dialog, open_files_dialog

from utilities.hpMCAutilities import displayErrorMessage
from axd.widgets.aEDXD_file_sequence_widget import FileSequenceDialog

############################################################

class aEDXDFilesController(QObject):
    tth_selection_changed_signal = QtCore.pyqtSignal(float)
    color_changed_signal = QtCore.pyqtSignal(int,tuple)
    files_changed_signal = pyqtSignal()
    apply_clicked_signal = pyqtSignal(dict)

    def __init__(self, model, display_window, config):
        super().__init__()
        self.model = model
        self.spectra_model = Spectra()
        bsize = self.model.params['bin_size']
        self.spectra_model.bin_size = bsize
        self.display_window = display_window
        self.config = config
        self.files_window = aEDXDFilesWidget()
        self.peak_cut_controller = aEDXDPeakCutController(self.model,self.spectra_model,self.display_window,self)
        self.create_connections()
        self.colors = {}
        self.current_tth = None

    def create_connections(self):
        self.display_window.spectrum_widget.apply_btn.clicked.connect(self.apply)
        self.files_window.apply_btn.clicked.connect(self.apply)

        self.files_window.add_all_btn.clicked.connect(self.add_all_files_clicked)
        self.files_window.add_scan.clicked.connect(self.add_scan_clicked)
        self.files_window.add_btn.clicked.connect(self.add_file_clicked)
        self.files_window.add_tth_btn.clicked.connect(self.add_group_clicked)
        self.files_window.file_trw.top_level_selection_changed_signal.connect(self.file_group_selection_changed)
        self.files_window.file_trw.file_selection_changed_signal.connect(self.file_selection_changed)
        self.files_window.file_trw.color_btn_clicked.connect(self.color_btn_clicked)
        self.files_window.delete_clicked_signal.connect(self.delete_clicked)
        self.files_window.clear_btn.clicked.connect(self.clear_clicked)
        self.files_window.file_trw.cb_state_changed_signal.connect(self.file_cb_changed_callback)
        self.files_window.file_trw.drag_drop_signal.connect(self.drag_drop_signal_callback)
        self.peak_cut_controller.cut_peaks_changed_signal.connect(self.data_changed_callback)
        self.peak_cut_controller.roi_window.apply_btn.clicked.connect(self.apply)
        self.files_window.file_trw.files_dragged_in.connect(self.files_dragged_in_callback)

        self.files_window.expand_btn.clicked.connect(partial(self.trw_expand,True))
        self.files_window.collapse_btn.clicked.connect(partial(self.trw_expand,False))

    def trw_expand(self, expand):
        if expand:
            self.files_window.file_trw.expandAll()
        else:
            self.files_window.file_trw.collapseAll()

    def drag_drop_signal_callback(self, drag_drop):
        source = drag_drop['source']
        target = drag_drop['target']
        filename = source['files']
        directory = self.model.params['inputdatadirectory']
        filepath = os.path.join(directory, filename[1])
        target_tth = target

        self.delete_clicked(source)
        self.add_file(target_tth, [filepath])

    def files_dragged_in_callback(self, dragged_in):
    
        
        for key in dragged_in:
            target_tth = key
            fnames = dragged_in[key]
            if type(target_tth) == str:
                if target_tth == 'auto':
                    self.add_files_auto_tth(fnames)
            else:
                self.add_file(target_tth, fnames)

    def apply(self):
        
        self.emit_spectra()

    def emit_spectra(self):
        spectra_par=self.get_spectra()
        self.apply_clicked_signal.emit(spectra_par)

    def get_spectra(self):
        dataarray, ttharray = self.spectra_model.get_dataarray()
        mcadata = self.spectra_model.get_file_list()
        mcadata_use = self.spectra_model.get_file_use_list()
        e_cut = self.spectra_model.get_cut_peaks()
        colors = self.colors
        tth = self.spectra_model.tth
        sq_colors=[]
        i=0
        for t in tth:
            if t in ttharray:
                sq_colors.append(colors[t])
        self.sq_colors=sq_colors
        spectra_par = {'dataarray':dataarray, 'ttharray':ttharray, 'mcadata':mcadata, 'mcadata_use': mcadata_use, 'E_cut':e_cut}
        return spectra_par

    def get_file_use(self):
        mcadata_use = self.spectra_model.get_file_use_list()
        return  mcadata_use


    def add_all_files_clicked(self):
        
        tth = self.spectra_model.tth
        if len(tth):
            directory = self.model.params['inputdatadirectory']
            filenames = open_files_dialog(self.files_window, "Select all EDXD files", directory=directory) 
            mod = len(filenames)%len(self.spectra_model.tth)
            
            
            dirname = os.path.dirname(filenames[0])
            self.model.params['inputdatadirectory']=dirname
            for i, f in enumerate(filenames):
                tth_i = i%len(self.spectra_model.tth)
                tth = self.spectra_model.tth[tth_i]
                
                self.spectra_model.add_files(tth, [f])
            self.files_loaded_callback()
            self.update_files_widget()
            self.peak_cut_controller.load_peaks_from_config()
        


    def add_scan_clicked(self):
        directory = self.model.params['inputdatadirectory']
        
        filename = open_file_dialog(
            self.files_window, "Load multiangle scan settings", directory=directory)
        if filename is not '':
            if filename.endswith('.json') or filename.endswith('.scan'):
                settings = self.load_settings(filename)
                self.clear_clicked()
                
                for s in settings:
                    tth = s[0]
                    self.add_tth_to_spectra_model(tth)
                

    def ask_to_clear(self):
        row_count = len(self.spectra_model.tth)
        if row_count>0:
            qm = QtWidgets.QMessageBox()
            ret = qm.question(self.files_window,'Warning', "Clear current scan?", qm.Yes | qm.No)
            if ret == qm.Yes:
                self.clear_clicked()

  
    
    def load_settings(self, filename):
        ok = False
        try:
            with open(filename) as f:
                openned_file = json.load(f)
            ok = True
        except:
            ok = False
        obj = None
        if ok:
            key = 'multiangle_settings'
            if key in openned_file:
                obj = openned_file[key]
        
        return obj

    def add_file_clicked(self):
        tth = self.files_window.file_trw.get_selected_tth()
        if tth != None:
            directory = self.model.params['inputdatadirectory']
            filenames = open_files_dialog(self.files_window, f"Select EDXD file(s) for 2\N{GREEK SMALL LETTER THETA}="+str(tth), directory=directory) 
            if len(filenames):
                filename = filenames[0]
                dirname = os.path.dirname(filename)
                self.model.params['inputdatadirectory']=dirname
                self.add_file(tth, filenames)

    def add_files_auto_tth(self, filenames):
        for fname in filenames:
            target_tth = get_tth_from_file(fname)
            if not target_tth in self.spectra_model.tth:
                self.add_tth_to_spectra_model(target_tth)
            self.spectra_model.add_files(target_tth, [fname])
        self.files_loaded_callback()
        self.update_files_widget()
        self.peak_cut_controller.load_peaks_from_config()
                

    def add_file(self, tth, filenames):

        dirname = os.path.dirname(filenames[0])
        self.model.params['inputdatadirectory']=dirname
        self.spectra_model.add_files(tth, filenames)
        self.files_loaded_callback()
        self.update_files_widget()
        self.peak_cut_controller.load_peaks_from_config()

    def add_group_clicked(self):
        t, okPressed = QtWidgets.QInputDialog.getDouble(
                        self.files_window, "Add "+f'2\N{GREEK SMALL LETTER THETA}',
                            "Enter "+f'2\N{GREEK SMALL LETTER THETA}'+" value:", 0, 0, 180, 4)
        if okPressed:
            self.add_tth_to_spectra_model(t)

    def add_tth_to_spectra_model(self, t):
        tth = self.spectra_model.tth
        if not t in tth:
            i = len(tth)
            self.spectra_model.add_tth(t)
            color  = calculate_color(i) 
            c = (int(color[0]), int(color[1]),int(color[2]))
            self.colors[t]=c
            c_str = '#%02x%02x%02x' % c
            self.files_window.file_trw.add_file_group([],c_str,str(t),[])

    def delete_clicked(self, param):
        ind = param['ind']
        files= param['files']
        item= param['item']
        if len(ind)==2:
            tth = files[0]
            filename=files[1]
            self.spectra_model.remove_file(tth,filename)
            self.files_window.file_trw.del_file(item)
        elif len(ind)==1:
            tth = files[0]
            self.spectra_model.remove_tth(tth)
            self.files_window.file_trw.del_group(item)
        #self.files_loaded_callback()
        self.data_changed_callback()    
        

    def clear_clicked(self):
        if len(self.files_window.file_trw.top_level_items):
            self.spectra_model.clear_files()
            self.files_loaded_callback()
            self.files_window.file_trw.clear_trw()
            self.data_changed_callback()

    def file_cb_changed_callback(self, ind, files, state):
        if len(ind)==2:
            tth = files[0]
            filename=files[1]
            self.spectra_model.set_file_use(tth,filename, state)
            self.plot_one_data()
            self.disp_alldata()

    def files_loaded_callback(self):
        tth_groups = self.spectra_model.tth_groups
        tth = self.spectra_model.tth
        self.colors = {}
        i=0
        for t in tth:
            color  = calculate_color(i) 
            c = (int(color[0]), int(color[1]),int(color[2]))
            self.colors[t]=c
            i += 1
        
    def data_changed_callback(self):  #triggered by signal from peak cut module
        self.plot_one_data()
        self.disp_alldata()
        

    def plot_one_data(self):
        
        self.display_window.spectrum_widget.fig.clear()
        spectra = self.spectra_model
        if len(spectra.tth):
            if self.current_tth== None:
                    self.current_tth= spectra.tth[0]
            tth = self.current_tth
            group = spectra.tth_groups[tth]
            spectrum_raw = group.spectrum_raw
            spectrum_unscaled = group.spectrum.unscaled
            if len(spectrum_raw[0]):
                
                colors = self.colors
                color = colors[tth]
                x = spectrum_unscaled[0]
                y = spectrum_unscaled[1]
                xo = spectrum_raw[0]
                yo = spectrum_raw[1]
                self.display_window.spectrum_widget.fig.add_line_plot(xo,yo,(150,150,150),1)
                self.display_window.spectrum_widget.fig.add_line_plot(x,y,color,2)
                self.display_window.spectrum_widget.setText( 
                                    f'2\N{GREEK SMALL LETTER THETA}'+' = ' + str(tth),1)
                self.display_window.spectrum_widget.fig.set_plot_label_color(color,1)

    def update_files_widget(self):
        tth_groups = self.spectra_model.tth_groups
        tth = self.spectra_model.tth  
        colors = self.colors
        i = 0
        self.files_window.file_trw.clear_trw()
        for i, t in enumerate(tth):
            files = tth_groups[t].files.keys()

            files_base = []
            files_use = []
            for f in files:
                f_u = tth_groups[t].files[f].use
                files_use.append(f_u)
                f_b = os.path.basename(f)
                files_base.append(f_b)
            tth = str(t*1.0)
            c_str = '#%02x%02x%02x' % colors[t]
            self.files_window.file_trw.add_file_group(files_base,c_str,tth,files_use)
        



    def check_if_files_exist(self, inputdatadirectory, file_groups):
        directory = None
        for filegroup in file_groups:
            files = filegroup[:-1]
            filepahts = []
            
            for f in files:
                full = os.path.join(inputdatadirectory, f)
                ex = os.path.isfile(full)
                if ex:
                    directory = inputdatadirectory
                else:
                    filename = open_file_dialog(self.display_window, 
                                "File " + f + " not found. Select new path.",filter = f) 
                    if filename:
                        fname = os.path.basename(filename)
                        if f == fname:
                            full = filename
                            dirpath = os.path.dirname(full)
                            directory=dirpath
                            inputdatadirectory=dirpath
                        else:
                            full = None
                            directory = None
                            break
                    else:
                        full = None
                        directory = None
                        break
                if full == None:
                    directory = None
                    break
            if directory == None:
                break
            directory = inputdatadirectory
        return directory


    def set_params(self, set_params):
        self.current_tth = None
        file_groups  = set_params['mcadata']
        
        if 'mcadata_use' in set_params:
            file_use = set_params['mcadata_use']
        else:
            file_use = []
        inputdatadirectory = self.model.params['inputdatadirectory']
        
        directory = self.check_if_files_exist(inputdatadirectory, file_groups)
        
        if directory is not None:
            self.spectra_model.inputdatadirectory = directory
            self.spectra_model.load_files_from_config(file_groups, file_use)
            self.model.params['inputdatadirectory']=directory
            self.files_loaded_callback()
            self.update_files_widget()
            self.peak_cut_controller.load_peaks_from_config()
            self.emit_spectra()
        else:
            if len(file_groups):
                displayErrorMessage( 'opt_read') 



    def file_selection_changed(self, ind, files):
        p_ind = ind[0]
        c_ind = ind[1]
        i = int(p_ind)
        if self.current_tth != files[0]:
            self.set_currect_tth(files[0])
        spectra = self.spectra_model
        tth = spectra.tth[i]
        group = spectra.tth_groups[tth]
        files  = group.files
        filename = list(files.keys())[c_ind]
        f = files[filename]
        data=np.asarray([f.energy, f.intensity])
        self.plot_raw_one(data, tth, filename)

    def color_btn_clicked(self, ind, button):
        previous_color = button.palette().color(1)
        tth = self.spectra_model.tth[ind]
        new_color = QtWidgets.QColorDialog.getColor(previous_color, self.files_window)
        if new_color.isValid():
            color = new_color
        else:
            color = previous_color
        c_rgb = (color.red(), color.green(), color.blue())
        self.colors[tth]=c_rgb 
        self.plot_one_data()
        self.disp_alldata()
        self.display_window.all_spectra_widget.fig.set_plot_label_color(color,ind)
        self.change_tth_color_in_widget(ind, color)
        
    def change_tth_color_in_widget(self, ind, color):
        self.files_window.file_trw.change_tth_color(ind, color)

    def set_currect_tth(self, tth):
        self.current_tth = tth
        self.tth_selection_changed_signal.emit(tth)
        self.plot_one_data()

    def file_group_selection_changed(self, ind,tth):
        self.set_currect_tth(tth)
        i = int(ind)
        spectra = self.spectra_model
        group = spectra.tth_groups[tth]
        files  = group.files
        datas = []
        for f_name in files:
            f = files[f_name]
            datas.append(np.asarray([f.energy, f.intensity]))
        filenames = files.keys()
        self.plot_raw_mult(datas, tth, filenames)
        

    def plot_raw_one(self, data, tth, filename):
        colors = self.colors
        x = data[0]
        y = data[1]
        self.display_window.raw_widget.fig.clear()
        
        color = '#%02x%02x%02x' % colors[tth]
        self.display_window.raw_widget.fig.add_line_plot(x,y,color,1)
        self.display_window.raw_widget.setText( 
                                    os.path.basename(filename) ,0)
        self.display_window.raw_widget.fig.set_plot_label_color(color,0)

    def plot_raw_mult(self,datas, tth, filenames):
        fnames = list(filenames)
        colors = self.colors
        color = '#%02x%02x%02x' % colors[tth]
        self.display_window.raw_widget.fig.clear()
        for i, data in enumerate(datas):
            x = data[0]
            y = data[1]
            self.display_window.raw_widget.fig.add_line_plot(x,y,color,1)
            self.display_window.raw_widget.setText( 
                                    os.path.basename(fnames[i]), i)
            self.display_window.raw_widget.fig.set_plot_label_color(color,i)

    def disp_alldata(self):   
        spectra = self.spectra_model
        self.display_window.all_spectra_widget.fig.clear()
        if len(spectra.tth):
            ind = self.display_window.tabWidget.currentIndex()
            tth_groups = spectra.tth_groups
            ttharray = spectra.tth
            colors = self.colors
            for i, tth in enumerate(ttharray):
                spectrum = tth_groups[tth].spectrum
                x = spectrum.x
                y = spectrum.y_cut
                color = colors[tth]
                self.display_window.all_spectra_widget.fig.add_line_plot(x,y,color,2)
                self.display_window.all_spectra_widget.setText(str(tth),i)
                #legend_labels.append(r"$2\theta$ = %(num)3.1f" %{"num" : ttharray[i]})
        
    def show_files(self):
        self.files_window.raise_widget()

    def save_file(self, outputfile):
        self.spectra_model.save_file(outputfile)