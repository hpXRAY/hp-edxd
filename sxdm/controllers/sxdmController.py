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

from PyQt6.QtCore import QObject
import numpy as np
from functools import partial
from PyQt6 import QtWidgets, QtCore

from sxdm.widgets.sxdmWidget import sxdmWidget
from hpm.models.multipleDatasetModel import MultipleSpectraModel
from hpm.widgets.UtilityWidgets import save_file_dialog, open_file_dialog, open_files_dialog

Theme = 1   # app style 0=windows 1=dark 
from .. import style_path
from .. import home_path

############################################################

class sxdmController(QObject):
    def __init__(self, app=None):
        super().__init__()
        self.app = app
        global Theme
        self.Theme = Theme
        self.style_path = style_path
        self.setStyle(self.Theme)
        self.multi_spectra_model = MultipleSpectraModel()
        self.window = sxdmWidget()
        
        
        self.make_connections()

    def initData(self, paths, progress_dialog):
        self.width = 21
        self.height = 21
        ## Create random image

        ## Set initial view bounds
        self.window.set_view_range(0, 0, self.width, self.height)


        self.data = self.load_data(paths, progress_dialog=progress_dialog)
        

        

    def load_data(self, paths, progress_dialog):
        


        self.multi_spectra_model.read_ascii_files_2d(paths, progress_dialog=progress_dialog)
        data = self.multi_spectra_model.r['data']
        return data

    def add_btn_click_callback(self, *args, **kwargs):
        """
        Loads a counts from multiple hpmca files into 2D numpy array
        :return:
        """
        
        folder = '/Users/ross/Desktop/20191126-dac/scan'
        files = sorted([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]) 
        paths = []
        for f in files:
            if "hpmca" in f:
                file = os.path.join(folder, f) 
                
                paths.append(file)

        filenames = paths

        if filenames is None:
            filenames = open_files_dialog(None, "Load Phase(s).",
                                          self.directories.phase )

            
        if len(filenames):
            #self.directories.phase = os.path.dirname(str(filenames[0]))
            progress_dialog = QtWidgets.QProgressDialog("Loading multiple spectra.", "Abort Loading", 0, len(filenames),
                                                        None)
            progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
            progress_dialog.setWindowFlags(QtCore.Qt.FramelessWindowHint)
            progress_dialog.show()
            QtWidgets.QApplication.processEvents()
            self.initData(filenames, progress_dialog)
            progress_dialog.close()
            QtWidgets.QApplication.processEvents()
        self.window.number.setValue(550)

    def make_connections(self):
        self.window.btn1.clicked.connect(self.add_btn_click_callback)
        self.window.btn2.clicked.connect(self.btn2callback)
        self.window.btn3.clicked.connect(self.btn3callback)
        self.window.number.valueChanged.connect(self.numbercallback)

  

    def numbercallback(self, *args, **kwargs):
        num = abs(int(round(args[0])))
        data_1 = self.data[:,num]
        data = np.reshape(data_1,(21,21))
        self.window.updateView(data)

    def btn2callback(self):
        pass

    def btn3callback(self):
        pass

    def showWindow(self):
        self.window.raise_widget()

    def setStyle(self, Style):
        #print('style:  ' + str(Style))
        if Style==1:
            WStyle = 'plastique'
            file = open(os.path.join(self.style_path, "stylesheet.qss"))
            stylesheet = file.read()
            self.app.setStyleSheet(stylesheet)
            file.close()
            self.app.setStyle(WStyle)
        else:
            WStyle = "windowsvista"
            self.app.setStyleSheet(" ")
            #self.app.setPalette(self.win_palette)
            self.app.setStyle(WStyle)