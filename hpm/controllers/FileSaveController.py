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


import os
from posixpath import basename
from typing import TextIO
from PyQt6.QtGui import QFontMetrics
import numpy as np
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import pyqtSignal, QObject, QFileSystemWatcher, QDir
from datetime import datetime
import pyqtgraph as pg
import time
import copy
from hpm.models.PhaseModel import PhaseLoadError
from utilities.HelperModule import get_base_name, increment_filename
from hpm.widgets.UtilityWidgets import save_file_dialog, open_file_dialog, open_files_dialog, CifConversionParametersDialog, open_folder_dialog
import utilities.hpMCAutilities as mcaUtil
from hpm.widgets.SaveFileWidget import SaveFileWidget
from utilities.hpMCAutilities import custom_signal
from epics import caput, caget, PV
import natsort

from .. import epics_sync





class FileSaveController(object):
    """
    FileSaveController handles file reading and saving
    """

    def __init__(self, hpmcaController, **kwargs):
        """
        
        """
        defaults_options = kwargs['defaults_options']
        
        self.pvs_file = {}
        record_name_file = defaults_options.file_record
        self.record_name_file = record_name_file 

        self.file_types = {'xy':'Data xy','chi':'Data chi','dat':'Data dat','fxye':'GSAS','png':'png'}

        if epics_sync:
            if record_name_file != None:
                self.pvs_file ={'FullFileName_RBV': None}
                '''             ,
                                'FilePath': None,
                                'FilePath_RBV': None,
                                'FileName': None,
                                'FileName_RBV': None,
                                
                                'FileTemplate': None,
                                'FileTemplate_RBV': None,
                                'WriteMessage': None,
                                'FileNumber': None,
                                'FileNumber_RBV': None,
                                'AutoIncrement': None,
                                'AutoIncrement_RBV': None,
                                'WriteStatus': None,
                                'FilePathExists_RBV': None,
                                'AutoSave': None,
                                'AutoSave_RBV': None,
                                'WriteFile': None,
                                'WriteFile_RBV': None}'''
                for pv_file in self.pvs_file.keys():
                    name = self.record_name_file + ':' + pv_file
                    self.pvs_file[pv_file] = PV(name)
        
        
        self.mca_controller = hpmcaController
        self.widget = hpmcaController.widget
        #self.file_widget = SaveFileWidget()
        
      
        self.file_options = mcaUtil.restore_file_settings()
        self.file_naming_options = mcaUtil.restore_file_naming_settings()
        self.widget.set_file_naming_settings(self.file_naming_options)
        self.McaFileNameHolder = ''
        self.McaFileName = ''

        lbl = self.mca_controller.widget.folder_lbl
        lbl.setText(self.mca_controller.working_directories.savedata)
        lbl = self.mca_controller.widget.last_saved_lbl
        lbl.setText(self.mca_controller.working_directories.last_saved_file)
        self.folder_watcher = QFileSystemWatcher()
        self.directory_changed_connected = False
        self.proxy = pg.SignalProxy(self.folder_watcher.directoryChanged, rateLimit=2, slot=self.handle_directory_changed )
    
        self.create_signals()

    def file_dragged_in_signal(self, f):

        if os.path.isfile(f):
            self.openFile(filename=f)
        elif os.path.isdir(f):
            self.openFolder(foldername = f)

    def preferences_module(self, *args, **kwargs):
        [ok, file_options] = mcaUtil.mcaFilePreferences.showDialog(self.widget, self.file_options) 
        if ok :
            self.file_options = file_options
            if self.mca_controller.mca != None:
                self.mca_controller.mca.file_settings = file_options
            mcaUtil.save_file_settings(file_options)
        

    def show_view(self):
        pass

        
    def create_signals(self):
        ui = self.widget
       
        ui.actionSave_next.setEnabled(False)
        ui.actionSave_next.triggered.connect(lambda:self.ClickedSaveNextFile(ui.actionSave_next.text()))
        ui.actionSave_As.triggered.connect(self.ClickedSaveFile)
        ui.file_dragged_in_signal.connect(self.file_dragged_in_signal)
        ui.actionExport_pattern.triggered.connect(self.export_pattern)
        ui.actionOpen_file.triggered.connect(self.openFile)
        ui.actionOpen_folder.triggered.connect(self.openFolder)
        ui.actionPreferences.triggered.connect(self.preferences_module)
        ui.folder_browse_btn.clicked.connect(self.folder_browse_btn_callback)

        ui.file_name_ebx.           editTextChanged.     connect(self.file_naming_option_changed_callback)
        ui.increment_file_name_cbx. clicked.             connect(self.file_naming_option_changed_callback)
        ui.starting_num_int.        textChanged.         connect(self.file_naming_option_changed_callback)
        ui.min_digits_int.          currentIndexChanged. connect(self.file_naming_option_changed_callback)
        ui.add_date_cbx.            stateChanged.        connect(self.file_naming_option_changed_callback)
        ui.add_time_cbx.            stateChanged.        connect(self.file_naming_option_changed_callback)
        ui.date_format_cmb.         currentIndexChanged. connect(self.file_naming_option_changed_callback)
        ui.time_format_cmb.         currentIndexChanged. connect(self.file_naming_option_changed_callback)
        ui.prefix_rad.              toggled.             connect(self.file_naming_option_changed_callback)
        ui.suffix_rad.              toggled.             connect(self.file_naming_option_changed_callback)
        ui.xy_cb  .                 stateChanged.        connect(self.file_naming_option_changed_callback)
        ui.chi_cb .                 stateChanged.        connect(self.file_naming_option_changed_callback) 
        ui.dat_cb .                 stateChanged.        connect(self.file_naming_option_changed_callback)
        ui.fxye_cb.                 stateChanged.        connect(self.file_naming_option_changed_callback)
        ui.png_cb .                 stateChanged.        connect(self.file_naming_option_changed_callback)


        ui.save_file_btn.clicked.connect(self.save_file_btn_callback)

        
        

    def acq_stopped(self):
        
        if self.file_options.autosave:
            basename = self.get_next_filename()
            if len(basename):
                self.save_file_btn_callback()
                 
                    
        if self.file_options.autorestart:
            # Important: this will only run if the mca is of epics type, the regular mca does not have the acq_erase_method
            self.mca_controller.mca.acq_erase_start()

    def save_file_btn_callback(self, *args):
        ext = '.hpmca'
        basename = self.get_next_filename()
        
        folder = self.mca_controller.working_directories.savedata
        filepath = os.path.join(folder,basename + ext)
    
        self.saveFile(filepath)
        self.increment_file_number()

    def increment_file_number(self):
        next_n = int(self.widget.starting_num_int.text()) + 1
        self.file_naming_options.starting_number = next_n
        mcaUtil.save_file_naming_settings(self.file_naming_options)
        self.widget.starting_num_int.setText(str(next_n))

    def get_next_filename(self):
        delimiter = '_'
        s = self.file_naming_options
        base_name = self.widget.file_name_ebx.currentText()
        filename = ''
        if s.dt_append_possition == 0:
            filename = self.add_date(s,filename, delimiter)
            filename = self.add_time(s, filename, delimiter)
        if len(filename):
            filename += delimiter
        filename += base_name
        filename = self.add_number(s, filename, delimiter)
        if s.dt_append_possition == 1:
            filename = self.add_date(s,filename, delimiter)
            filename = self.add_time(s, filename, delimiter)
        return filename

    def add_number(self, s, text_in, delimiter=' '):    
        if s.increment_file_name:

            digits = s.minimum_digits +1 
            next_n = int(self.widget.starting_num_int.text())
            num_str = delimiter +str(next_n).zfill(digits)
            text_in += num_str
        return text_in
        

    def add_date(self, s, text_in, delimiter=' '):
        '''
        formats=
        ['YYYYMMDD',
        'YYYY-MM-DD',
        'YYYY-Month-DD',
        'Month-DD-YYYY']
        '''
        if s.add_date:
            d_format = s.d_format
            df = ''
            if d_format == 0:
                df = '%Y%m%d'
            if d_format == 1:
                df = '%Y-%m-%d'
            if d_format == 2:
                df = '%Y-%b-%d'
            if d_format == 3:
                df = '%b-%d-%Y'
            dstr = datetime.today().strftime(df)
            if len(text_in):
                text_in += delimiter
            text_in += dstr 
        return text_in

    def add_time(self, s, text_in, delimiter=' '):
        if s.add_time:
            t_format = s.t_format
            tf = ''
            if t_format == 0:
                tf = '%H-%M-%S'
            if t_format == 1:
                tf = '%I-%M-%S-%p'
            tstr = datetime.today().strftime(tf)
            if len(text_in):
                text_in += delimiter
            text_in += tstr 
        return text_in

    def file_naming_option_changed_callback(self, *args):

        settings = self.widget.get_file_naming_settings()
        for s in settings:
            setattr( self.file_naming_options, s, settings[s])
        mcaUtil.save_file_naming_settings(self.file_naming_options)


    def ClickedSaveFile(self):  # handles Save As...
        if self.mca_controller.mca != None:
            #self.file_widget.raise_widget()
            filename =  save_file_dialog(self.widget, "Save spectrum file.",
                                    self.mca_controller.working_directories .savedata,
                                    'Spectrum (*.hpmca)', False)
            if filename != None:
                if len(filename)>0:
                    self.saveFile(filename)

    def ClickedSaveNextFile(self, menu_text):
        if self.mca_controller.mca != None:
            filename = menu_text.split(': ')[1]
            filename = os.path.join(self.mca_controller.working_directories .savedata, filename)
            self.saveFile(filename)



    def saveFile(self, filename):
        if self.mca_controller.mca != None:
            exists = os.path.isfile(filename)
            if not exists:
                write = True
            else:
                if  os.access(filename, os.R_OK):
                    base = os.path.basename(filename)
                    if self.file_options.warn_overwrite:
                        choice = QtWidgets.QMessageBox.question(None, 'Confirm Save As',
                                                    base+" already exists. \nDo you want to replace it?",
                                                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                        write = choice == QtWidgets.QMessageBox.Yes                              
                    else: write = True
                else: write = False
            if write:
                fileout, success = self.mca_controller.mca.write_file(
                    file=filename, netcdf=0)
                self.autoexport(fileout)
                if success:
                    
                    #self.update_titlebar()
                    self.update_epics_filename(fileout)
                    
                    self.update_saveDataDir (os.path.dirname( str(fileout)), fileout) # working directory xrd files
                    
                    self.widget.last_saved_lbl.setText(fileout)
                    # new_file = increment_filename(filename) # old way of incrementing
                    '''if new_file != filename:
                        self.widget.actionSave_next.setText(
                            'Save next: ' + os.path.basename(new_file))
                        self.widget.actionSave_next.setEnabled(True)
                    else:
                        self.widget.actionSave_next.setText('Save next')
                        self.widget.actionSave_next.setEnabled(False)'''
                else:
                    mcaUtil.displayErrorMessage('fs')
            else: 
                mcaUtil.displayErrorMessage('fs')

    def folder_browse_btn_callback(self):
        
        folder = open_folder_dialog(self.widget, "Select folder for saving data.")
        if folder != '' and folder is not None:
            self.update_saveDataDir(folder)

    def update_readDataDir(self, dir):
        self.mca_controller.working_directories.readdata = os.path.abspath(dir)
        mcaUtil.save_folder_settings(self.mca_controller.working_directories )
        

    def update_saveDataDir(self, dir, file=''):
        self.mca_controller.working_directories.savedata = os.path.abspath(dir)
        if len(file):
            self.mca_controller.working_directories.last_saved_file = os.path.abspath(file)
        mcaUtil.save_folder_settings(self.mca_controller.working_directories )
        lbl = self.mca_controller.widget.folder_lbl
        
        lbl.setText(self.mca_controller.working_directories.savedata)

    def update_epics_filename(self, filename):
        if self.mca_controller.mca is not None:
            try:
                if 'FullFileName_RBV' in self.pvs_file:
                    if self.pvs_file['FullFileName_RBV'] != None:
                        self.pvs_file['FullFileName_RBV'].put(filename)
            except:
                pass

    def openFile(self, *args, **kwargs):
        filename = kwargs.get('filename', None)
        self.folder_watcher_stop_watching()
        if filename is None:
            filename = open_file_dialog(self.widget, "Open spectrum file.",
                                    self.mca_controller.working_directories .readdata)
        if filename != '' and filename is not None:
            if os.path.isfile(filename):
                if self.mca_controller.Foreground != 'file':
                    success = self.mca_controller.initMCA('file',filename) == 0
                else:
                
                    [filename, success] = self.mca_controller.mca.read_file(file=filename, netcdf=0, detector=0)
                   
                if success:
                    self.McaFileName = filename
                    self.update_readDataDir (os.path.dirname(str(filename)) )#working directory xrd files
                    
                    # best to initialize controllers only once per session
                    if not self.mca_controller.controllers_initialized:  
                        self.mca_controller.initControllers()

                    self.mca_controller.multiple_datasets_controller.set_mca(self.mca_controller.mca)
                    self.mca_controller.dataReceivedFile()
                else:
                    mcaUtil.displayErrorMessage( 'fr')
            else:
                mcaUtil.displayErrorMessage( 'fr')

    def openFolder(self, *args, **kwargs):
        foldername = kwargs.get('foldername', None)
        self.folder_watcher_stop_watching()
        if foldername is None:
            foldername = open_folder_dialog(self.widget, "Open folder.",
                                    self.mca_controller.working_directories .readdata)
        if foldername != '' and foldername is not None:
            if os.path.isdir(foldername):
                if self.mca_controller.Foreground != 'file':
                    success = self.mca_controller.initMCA('file',foldername) == 0
                    
                else:
                
                    [filenames, success] = self.mca_controller.mca.read_files(folder=foldername, netcdf=0, detector=0)
                    
                if success:
                    self.McaFileName = foldername
                    self.update_readDataDir (os.path.dirname(str(foldername)) ) #working directory xrd files
                    
                    # best to initialize controllers only once per session
                    if not self.mca_controller.controllers_initialized:  
                        self.mca_controller.initControllers()
                    self.mca_controller.multiple_datasets_controller.set_mca(self.mca_controller.mca)
                    self.mca_controller.dataReceivedFolder()

                
                    self.folder_watcher_add_directory(foldername)
                    self.folder_watcher_start_watching()
                    
                else:
                    mcaUtil.displayErrorMessage( 'fr')
            else:
                mcaUtil.displayErrorMessage( 'fr')

    #################### #################### #################### 
    #################### Folder wathcher code start
    #################### #################### #################### 

    def folder_watcher_start_watching(self):
        if not self.directory_changed_connected:

            #self.folder_watcher.directoryChanged.connect(self.handle_directory_changed)
            self.directory_changed_connected = True

    def folder_watcher_stop_watching(self):
        if self.directory_changed_connected:
            #self.folder_watcher.directoryChanged.disconnect(self.handle_directory_changed)
            self.directory_changed_connected = False
            print("Stopped watching all directories.")
        else:
            pass
            #print("Warning: 'folder_watcher_handle_directory_changed' slot was not connected to 'directoryChanged' signal.")

    '''def folder_watcher_handle_directory_changed(self):
        self.proxy.emit()'''
        

    def folder_watcher_add_directory(self, path):
        self.folder_watcher_remove_all_directories()
        self.folder_watcher.addPath(path)
        print(f"Watching directory: {path}")

    def folder_watcher_remove_all_directories(self):
        watched_directories = self.folder_watcher.directories()
        if len(watched_directories):
            self.folder_watcher.removePaths(watched_directories)
            print(f"Stopped watching all directories.")
    
    def handle_directory_changed(self):
        if self.mca_controller.Foreground == 'file':
            if self.directory_changed_connected:
                directories =  self.folder_watcher.directories()
                if len(directories):
                    path = directories[0]
                    paths = []
                    eL = QDir(path).entryList(QDir.Files)
                    for file in eL:
                        file_path = QDir.toNativeSeparators(path + '/' + file)
                        paths.append(file_path)
                    self.add_files(paths)

            

    def add_files(self, paths):
        paths = self.sort_and_filter_files(paths)
        new_paths = []
        if len(paths):
            for path in paths:
                if path not in self.mca_controller.mca.files_loaded:
                    new_paths.append(path)
        

            if len(new_paths):
                self.mca_controller.mca.read_files(paths=new_paths, replace = False)
                self.mca_controller.multiple_datasets_controller.set_mca(self.mca_controller.mca)
                self.mca_controller.dataReceivedFolder()

    def sort_and_filter_files(self, paths):
        files = natsort.natsorted(paths)
        files_filtered = []
        for f in files:
            if f.endswith('.hpmca') or f.endswith('.chi') or f.endswith('.mca') or f.endswith('.xy') or f.endswith('.nxs') or f[-3:].isnumeric() :
                
                
                files_filtered.append(f)
        return files_filtered

    #################### #################### #################### 
    #################### Folder wathcher code end
    #################### #################### #################### 

    def export_pattern(self):
        ext = self.mca_controller.working_directories.export_ext
        filters = ''
        if ext in self.file_types:
            filters = self.file_types[ext] + ' (*.' + ext + ')'
            for e in self.file_types:
                if e != ext:
                    filters += ';;'+ self.file_types[e] + ' (*.' + e + ')'
        else:
            filters = 'Data xy (*.xy);;Data chi (*.chi);;Data dat (*.dat);;GSAS (*.fxye);;png (*.png)'
            ext = 'xy'
        if self.mca_controller.mca is not None:
            img_filename, _ = os.path.splitext(os.path.basename(self.mca_controller.mca.file_name))
            if len(ext):
                extension = "." + ext
            else:
                extension = ''
            filename = save_file_dialog(
                self.widget, "Save Pattern Data.",
                os.path.join(self.mca_controller.working_directories.exportdata,
                            img_filename + extension),
                    (filters),True)
            if filename != '':
                self.export(filename)

    def autoexport(self, fileout):
        export_set = dict()
        export_set['xy'] = self.file_naming_options.export_xy
        export_set['chi'] = self.file_naming_options.export_chi
        export_set['dat'] = self.file_naming_options.export_dat
        export_set['fxye'] =  self.file_naming_options.export_fxye
        export_set['png'] =  self.file_naming_options.export_png
        
        basename = os.path.basename(fileout).split('.')[0]+'.'
        folder = os.path.dirname(fileout)
        for e in export_set:
            if export_set[e]:
                fout = os.path.join(folder,basename+e)
                self.export(fout)
            
    def export(self, filename):
        
        if filename.endswith('.png'):
            self.widget.pg.export_plot_png(filename)
        #elif filename.endswith('.svg'):
        #    self.widget.pg.export_plot_svg(filename)
        else:
            self.mca_controller.mca.export_pattern(filename, self.mca_controller.unit, self.mca_controller.plotController.units[self.mca_controller.unit])

        folder = os.path.split(filename)[0]
        self.mca_controller.working_directories.exportdata = os.path.abspath(folder)
        if '.' in filename:
            ext = str.split(filename, '.')[-1]
        else:
            ext = ''
        self.mca_controller.working_directories.export_ext = ext
        mcaUtil.save_folder_settings(self.mca_controller.working_directories )

    def epics_connections(self):

        pass

        # these may get implemented in the future to be in line with area detector file saving workflow
        '''self.pvs_file ={'FilePath': None,
                        'FilePath_RBV': None,
                        'FileName': None,
                        'FileName_RBV': None,
                        'FullFileName_RBV': None,
                        'FileTemplate': None,
                        'FileTemplate_RBV': None,
                        'WriteMessage': None,
                        'FileNumber': None,
                        'FileNumber_RBV': None,
                        'AutoIncrement': None,
                        'AutoIncrement_RBV': None,
                        'WriteStatus': None,
                        'FilePathExists_RBV': None,
                        'AutoSave': None,
                        'AutoSave_RBV': None,
                        'WriteFile': None,
                        'WriteFile_RBV': None}'''

        # filenames are written by hpMCA to this PV to be grabbed by an external data logger.
        ''' EPICS record should be created as follows:
            record(waveform,"16bmb:mca_file:FullFileName_RBV"){
                field(DESC,"FullFileName")
                field(DTYP,"Soft Channel")
                field(DESC,"file name")
                field(NELM,"256")
                field(FTVL,"CHAR")
            } 
        '''