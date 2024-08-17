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

"""
selfCreates a GUI window to calibrate 2-theta for an Mca.

Author:         Mark Rivers
Created:        Sept. 18, 2002
Modifications:  MLR, Sept. 20, 2002.  Numerous bug fixes.
                Ross Hrubiak, Oct. 14, 2018  
                    - re-done for Python 3
                    - computational routines changed from Numeric, MLab to numpy
                    - GUI re-written, changed from tkinter and Pmw to PyQt6
                    - plotting library changed from BltPlot to pyqtgraph
                    - changes to MCA calibration model: channel_to_energy, etc. moved to the mcaCalibration class
                    - this module now only copies the rois and the calibration objects from MCA not the entire MCA
                       
"""


from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

import copy
import numpy as Numeric
import math

import utilities.CARSMath as CARSMath
import functools
import hpm.models.jcpds as jcpds
import utilities.centroid as centroid

############################################################
class mcaCalibrate2theta_widgets(object):
   """ Private class"""
   def __init__(self, nrois):
      self.use_flag             = [None]*nrois
      self.d_spacing            = [None]*nrois
      self.fwhm                 = [None]*nrois
      self.energy               = [None]*nrois
      self.label                = [None]*nrois
      self.two_theta            = [None]*nrois
      self.two_theta_diff       = [None]*nrois
      self.two_theta_fit        = [None]*nrois

class mcaCalibrate2theta(QtWidgets.QWidget):
    def __init__(self, mca, detector=0, command=None,jcpds_directory='', data_label = ''):
        """
        Creates a new GUI window for calibrating 2-theta for an Mca object.

        Inputs:
            mca:
                An Mca instance to be calibrated.  The Mca must have at least 1
                Regions of Interest (ROI) defined.

        Keywords:
            command:
                A callback command that will be executed if the OK button on
                the GUI window is pressed.  The callback will be invoked as:
                command(exit_status)
                where exit_status is 1 if OK was pressed, and 0 if Cancel was
                pressed or the window was closed with the window manager.

        Procedure:
            The calibration is done by determining the centroid position and
            d-spacing of each ROI.

            The centroids positions are computed by fitting the
            ROI counts to a Gaussian, using CARSMath.fit_gaussian.

            The d-spacing the ROI can be entered manually in the GUI window, or it
            can be determined automatically if the label of the ROI can be
            successfully used in jcpds.lookup_jcpds_line().

            Each ROI can be selectively used or omitted when doing the calibration.

            The errors in the 2-theta calibration, can be plotted using BltPlot.
        """
        super().__init__()
        self.input_mca = mca
        self.detector = detector
        self.roi = copy.deepcopy(mca.get_rois()[detector])
        self.calibration = copy.deepcopy(mca.get_calibration()[detector])   
        #self.data = copy.deepcopy(mca.get_data()[detector])
        self.jcpds_directory = jcpds_directory
        self.data_label = data_label

        self.fname_label = 'Phase file not found. For automatic calibration, please close this \nwindow and load the corresponding phase file (.jcpds) first.'
        self.phase_name = ''
        self.phase_found = False
        if len(self.roi):
            roi = self.roi[0]
            label = roi.label
            temp = label.split()
            if len(temp) >= 2:
                file = temp[0]
                self.phase_name = file
                item = jcpds.find_fname(self.jcpds_directory,  file+'.jcpds')
                if item is not None:
                    self.fname_label = 'Using phase file: ' + item['full_file']
                    self.phase_found = True
                

        self.exit_command = command
        self.nrois = len(self.roi)
        if (self.nrois < 1):
            
            
            
            return 
        
        self.fwhm_chan   = Numeric.zeros(self.nrois, Numeric.float)
        self.two_theta   = Numeric.zeros(self.nrois, Numeric.float)
        self.widgets     = mcaCalibrate2theta_widgets(self.nrois)
        

        
        self.initUI()
        self.show()

    # start widgets

    def initUI(self):

        #### display column headings    
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self)
        self.data_lbl = QtWidgets.QLabel(self.data_label)
        self.phase_file_label = QtWidgets.QLabel(self.fname_label)
        self.verticalLayout_4.addWidget(self.data_lbl)
        self.verticalLayout_4.addWidget(self.phase_file_label)
        self.groupBox = QtWidgets.QGroupBox(self)
        self.container = self.groupBox        
        self.gridLayout = QtWidgets.QGridLayout(self.container)
        self.gridLayout.setContentsMargins(7, 15, 7, 7)
        self.gridLayout.setSpacing(5)

        
        
        header = {'ROI':0,'Use?':1,'HKL':2,
            'd-spacing':3,'Energy':4,f'2\N{GREEK SMALL LETTER THETA}':5, f'2\N{GREEK SMALL LETTER THETA}' + ' error':6
        }

        row = 0
        for key, col in header.items():
            t = QtWidgets.QLabel(self.groupBox)
            t.setText(key)
            t.setAlignment(QtCore.Qt.AlignHCenter)
            t.setMinimumSize(QtCore.QSize(60, 0))
            self.gridLayout.addWidget(t, row, col, QtCore.Qt.AlignHCenter)

        #### display rois parameters

        for i in range(self.nrois):
            row=i+1 
            
            t = QtWidgets.QLabel(self.groupBox)
            t.setText(str(i))
            t.setAlignment(QtCore.Qt.AlignHCenter)
            self.gridLayout.addWidget(t, row, 0, QtCore.Qt.AlignHCenter)  

            self.widgets.use_flag[i] = t = QtWidgets.QCheckBox(self.groupBox)
            t.setChecked(self.roi[i].use==1)
            t.toggled.connect(functools.partial(self.menu_use, i)) # lambda expression didn't work so using functools.partial instead
            self.gridLayout.addWidget(t, row, 1, QtCore.Qt.AlignHCenter)

            self.widgets.label[i] = t = QtWidgets.QLineEdit(self.groupBox)
            t.setText(self.roi[i].label)
            t.setFixedWidth(70)
            t.setAlignment(QtCore.Qt.AlignHCenter)
            self.widgets.label[i].returnPressed.connect(functools.partial(self.menu_label, i))
            self.gridLayout.addWidget(t, row, 2, QtCore.Qt.AlignHCenter)
            
            
            '''
            self.widgets.fwhm[i] = t =  QtWidgets.QLineEdit(self.groupBox)
            t.setText('%.3f' % self.roi[i].fwhm)
            t.setFixedWidth(70)
            t.setAlignment(QtCore.Qt.AlignHCenter)
            self.gridLayout.addWidget(t, row, 3, QtCore.Qt.AlignHCenter)
            '''

            # try to use the label to lookup d spacing

            d = jcpds.lookup_jcpds_line(self.roi[i].label,path=self.jcpds_directory)
            if (d != None): self.roi[i].d_spacing = d
            else: self.roi[i].d_spacing = 0

            self.widgets.d_spacing[i] = t = QtWidgets.QLineEdit(self.groupBox)
            t.setText('%.4f' % self.roi[i].d_spacing)
            t.setFixedWidth(70)
            t.setAlignment(QtCore.Qt.AlignHCenter)
            t.editingFinished.connect(functools.partial(self.menu_d_spacing, i))
            self.gridLayout.addWidget(t, row, 3, QtCore.Qt.AlignHCenter)
            
            self.widgets.energy[i] = t = QtWidgets.QLineEdit(self.groupBox)
            t.setText('%.3f' % self.roi[i].energy)
            t.setFixedWidth(70)
            t.setAlignment(QtCore.Qt.AlignHCenter)
            t.editingFinished.connect(functools.partial(self.menu_energy, i))
            self.gridLayout.addWidget(t, row, 4, QtCore.Qt.AlignHCenter)


            self.widgets.two_theta[i] = t = QtWidgets.QLineEdit(self.groupBox)
            t.setText('%.4f' % 0.0)
            t.setFixedWidth(70)
            t.setAlignment(QtCore.Qt.AlignHCenter)
            self.gridLayout.addWidget(t, row, 5, QtCore.Qt.AlignHCenter)

            self.widgets.two_theta_diff[i] = t = QtWidgets.QLineEdit(self.groupBox)
            t.setText('%.4f' % 0.0)
            t.setFixedWidth(70)
            t.setAlignment(QtCore.Qt.AlignHCenter)
            self.gridLayout.addWidget(t, row, 6, QtCore.Qt.AlignHCenter)

        self.verticalLayout_4.addWidget(self.groupBox)


        self.groupBox_2 = QtWidgets.QGroupBox(self)
        self.groupBox_2.setTitle("")

        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        
        
        # in QtDialong, setting default=False, autoDefault=False prevents button from default trigger by Enter key
        self.widgets.do_fit = t = QtWidgets.QPushButton(self.groupBox_2, default=False, autoDefault=False)
        t.setText("Compute "+f'2\N{GREEK SMALL LETTER THETA}')
        t.setFixedWidth(110)
        t.clicked.connect(self.menu_do_fit)
        self.horizontalLayout.addWidget(t)

        self.widgets.plot_cal = t = QtWidgets.QPushButton(self.groupBox_2, default=False, autoDefault=False)
        t.setText("Plot " + f'2\N{GREEK SMALL LETTER THETA}'+" error")
        t.clicked.connect(self.menu_plot_calibration)
        t.setFixedWidth(110)
        self.horizontalLayout.addWidget(t)

        self.lbltwo_theta_fit = t = QtWidgets.QLabel(self.groupBox_2)
        t.setText(f'2\N{GREEK SMALL LETTER THETA}')
        t.setFixedWidth(70)
        t.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.horizontalLayout.addWidget(t)

        self.widgets.two_theta_fit = t = QtWidgets.QLineEdit(self.groupBox_2)
        t.setText('%.5f' % self.calibration.two_theta)
        self.horizontalLayout.addWidget(t)

        self.horizontalLayout.setAlignment(QtCore.Qt.AlignHCenter)

        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.verticalLayout_4.addWidget(self.groupBox_2)

        self.groupBox_4 = QtWidgets.QGroupBox(self)
        self.groupBox_4.setTitle("")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBox_4)

        self.btnOK = QtWidgets.QPushButton(self.groupBox_4, default=False, autoDefault=False)
        self.btnOK.clicked.connect(functools.partial(self.menu_ok_cancel, 'OK'))
        self.btnOK.setText("OK")
        self.horizontalLayout_2.addWidget(self.btnOK)
        self.btnCancel = QtWidgets.QPushButton(self.groupBox_4, default=False, autoDefault=False)
        self.btnCancel.clicked.connect(functools.partial(self.menu_ok_cancel, 'Cancel'))
        self.btnCancel.setText("Cancel")
        self.horizontalLayout_2.addWidget(self.btnCancel)

        self.verticalLayout_4.addWidget(self.groupBox_4)

        self.setWindowTitle(f'2\N{GREEK SMALL LETTER THETA}' +" calibration")
        self.groupBox.setTitle("Defined regions")
        self.setFixedSize(self.verticalLayout_4.sizeHint())  

        #self.setWindowFlags(QtCore.Qt.Tool)
        #self.setAttribute(QtCore.Qt.WA_MacAlwaysShowToolWindow)     

    def raise_widget(self):
        self.show()
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
        self.activateWindow()
        self.raise_()

    def menu_use(self,  roi):
        value = self.widgets.use_flag[roi].isChecked()
        """ Private method """
        self.roi[roi].use = value 
   

    def menu_energy(self, roi):
        """ Private method """
        energy = float(self.widgets.energy[roi].text())
        self.roi[roi].energy = energy
        self.widgets.energy[roi].setText('%.3f' % energy)
    

    def menu_d_spacing(self, roi):
        """ Private method """
        d_spacing = float(self.widgets.d_spacing[roi].text())
        self.roi[roi].d_spacing = d_spacing
        self.widgets.d_spacing[roi].setText('%.3f' % d_spacing)
     

    def menu_label(self, roi):
        """ Private method """
        label = self.widgets.label[roi].text()
        d_spacing = jcpds.lookup_jcpds_line(label,path=self.jcpds_directory)
        if (d_spacing != None):
            self.roi[roi].d_spacing = d_spacing
            self.widgets.d_spacing[roi].setText('%.3f' % d_spacing)
            
        self.roi[roi].label=label

    def menu_do_fit(self):
        all_d_non_zero = True
        for i in range(self.nrois):
            d = self.roi[i].d_spacing
            u = self.roi[i].use
            if (not d>0) and u :
                all_d_non_zero = False

        if (self.phase_found and all_d_non_zero) or all_d_non_zero:
            """ Private method """
            # Compute 2-theta for each ROI
            for i in range(self.nrois):
                case = self.roi[i].energy == 0 #or (self.roi[i].d_spacing == 0)
                if case:
                    self.two_theta[i] = 0.
                if not case:
                    e = self.roi[i].energy
                    d = self.roi[i].d_spacing
                    if d > 0:
                        self.two_theta[i] = 2.0 * math.asin(12.398 / (2.0*e*d))*180./math.pi
                    
                        self.widgets.two_theta[i].setText('%.5f' % self.two_theta[i])
                    else:
                        self.widgets.two_theta[i].setText('inf')
            # Find which ROIs should be used for the calibration
            use = []
            for i in range(self.nrois):
                if (self.roi[i].use): use.append(i)
            nuse = len(use)
            if (nuse < 1):
                message='Must have at least one valid point for calibration'
                print(message)
                return
            two_theta=[]
            for u in use:
                two_theta.append(self.two_theta[u])
            self.calibration.two_theta = Numeric.mean(two_theta)
            sdev = Numeric.std(two_theta)
            self.widgets.two_theta_fit.setText(
                                        ('%.5f' % self.calibration.two_theta)
                                        + ' +/- ' + ('%.5f' % sdev))
            for i in range(self.nrois):
                two_theta_diff = self.two_theta[i] - self.calibration.two_theta
                self.widgets.two_theta_diff[i].setText('%.5f' % two_theta_diff)
        else:

            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setIcon(QtWidgets. QMessageBox.Information) 
            msg_box.setStandardButtons(QtWidgets. QMessageBox.Ok )
            msg_box.setWindowTitle("Phase not recognized")
            msg_box.setText(self.phase_name + ' phase file not found. For automatic calibration, please close this \nwindow and load the corresponding phase file (.jcpds) first. Or manually enter d-spacing values for each hkl line.')
            msg_box.show()
        
    def menu_plot_calibration(self):
        """ Private method """
        energy = []
        two_theta_diff = []
        energy_use = []
        two_theta_diff_use = []
        for i in range(self.nrois):
            energy.append(self.roi[i].energy)
            two_theta_diff.append(self.two_theta[i] - self.calibration.two_theta)
            if (self.roi[i].use):
                energy_use.append(energy[i])
                two_theta_diff_use.append(two_theta_diff[i])
        
        pltError = pg.plot(energy_use,two_theta_diff_use, 
                pen=(200,200,200), symbolBrush=(255,0,0),antialias=True, 
                symbolPen='w', title=f'2\N{GREEK SMALL LETTER THETA}'+" error"
        )
        pltError.setLabel('left', f'2\N{GREEK SMALL LETTER THETA}'+" error")
        pltError.setLabel('bottom', 'Energy')

    def menu_ok_cancel(self, button):
        """ Private method """
        if (button == 'OK') or (button == 'Apply'):
            # Copy calibration and rois to input mca object
            self.calibration.set_dx_type('adx')
            self.input_mca.set_calibration(self.calibration, self.detector)
            #self.input_mca.set_rois(self.roi)
            pass
        if (button == 'OK'):
            exit_status = 1
        elif (button == 'Apply'):
            return
        else:
            exit_status = 0
        if (self.exit_command): self.exit_command(exit_status)
        self.hide() 