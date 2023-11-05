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
Creates a GUI window to calibrate energy for an Mca.

Author:         Mark Rivers
Created:        Sept. 18, 2002
Modifications:
    Oct. 14, 2018   Ross Hrubiak
        - re-done Python 3
        - computational routines changed from np to numpy
        - GUI re-written, changed from tkinter and Pmw to pyqt5
        - plotting library changed from BltPlot to pyqtgraph
        - changes to MCA calibration model: channel_to_energy, etc. moved to the mcaCalibration class
        - this module now only copies the rois and the calibration objects from MCA not the entire MCA
            
"""

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

import copy
import numpy as np
import hpm.models.Xrf as Xrf
import utilities.CARSMath as CARSMath
import functools
from hpm.models.eCalModel import fit_energies
from hpm.models.mcaModel import MCA
from hpm.widgets.UtilityWidgets import open_file_dialog



class mcaCalibrateEnergy_widgets():

    def __init__(self, nrois):
        
        self.use_flag             = [None]*nrois
        self.centroid             = [None]*nrois
        self.fwhm                 = [None]*nrois
        self.energy               = [None]*nrois
        self.energy_diff          = [None]*nrois
        self.line                 = [None]*nrois

    def add_widgets(self, nrois):
        self.use_flag             += [None]*nrois
        self.centroid             += [None]*nrois
        self.fwhm                 += [None]*nrois
        self.energy               += [None]*nrois
        self.energy_diff          += [None]*nrois
        self.line                 += [None]*nrois

class mcaCalibrateEnergy(QtWidgets.QWidget):
    def __init__(self, mca, working_directories, detector=0, command=None, parent=None):
        """
        Creates a new GUI window for calibrating energy for an Mca object.

        Inputs:
            mca:
                An Mca instance to be calibrated.  The Mca must have at least 2
                Regions of Interest (ROIs) defined for a linear calibration and
                2 ROIs defined for a quadratic calibration.

        Keywords:
            command:
                A callback command that will be executed if the OK button on
                the GUI window is pressed.  The callback will be invoked as:
                command(exit_status)
                where exit_status is 1 if OK was pressed, and 0 if Cancel was
                pressed or the window was closed with the window manager.

        Procedure:
            The calibration is done by determining the centroid position and
            energy of each ROI.

            The centroids positions are computed by fitting the
            ROI counts to a Gaussian, using CARSMath.fit_gaussian.

            The energy the ROI can be entered manually in the GUI window, or it
            can be determined automatically if the label of the ROI can be
            successfully used in Xrf.lookup_xrf_line() or Xrf.lookup_gamma_line().

            Each ROI can be selectively used or omitted when doing the calibration.

            The errors in the energy calibration and the FWHM of each ROI as a
            function of energy, can be plotted using BltPlot.
        """
        super(mcaCalibrateEnergy, self).__init__()
        self.input_mca = mca
        self.working_directories = working_directories
        
        self.det = detector
        #self.input_mca.auto_process_rois = False
        self.roi = copy.deepcopy(mca.get_rois()[detector])
        self.calibration = copy.deepcopy(mca.get_calibration()[detector])   
        self.data = copy.deepcopy(mca.get_data()[detector])

        self.exit_command = command
        
        self.nrois = len(self.roi)
        if (self.nrois < 2):
            
            
            return
        
        self.fwhm_chan = np.zeros(self.nrois)
        self.widgets = mcaCalibrateEnergy_widgets(self.nrois)
        
        # Compute the centroid and FWHM of each ROI
        self.compute_centroids()
                                            

        self.initUI()
        self.show()

    # start widgets

    def compute_centroids(self):
        
        for i in range(self.nrois):
            left = self.roi[i].left
            right = self.roi[i].right+1
            total_counts = self.data[left:right]
            n_sel        = right - left
            sel_chans    = left + np.arange(n_sel)
            left_counts  = self.data[left]
            right_counts = self.data[right]
            bgd_counts   = (left_counts + np.arange(float(n_sel))/(n_sel-1) *
                                        (right_counts - left_counts))
            net_counts   = total_counts - bgd_counts
            net          = np.sum(net_counts)
    
            if ((net > 0.) and (n_sel >= 3)):
                amplitude, centroid, fwhm = CARSMath.fit_gaussian(sel_chans, net_counts)
                self.roi[i].centroid = centroid
                self.fwhm_chan[i] = fwhm
            else:
                self.roi[i].centroid = (left + right)/2.
                self.fwhm_chan[i] = right-left
            self.roi[i].fwhm = (self.calibration.channel_to_energy(self.roi[i].centroid + 
                                            self.fwhm_chan[i]/2.) - 
                                self.calibration.channel_to_energy(self.roi[i].centroid - 
                                            self.fwhm_chan[i]/2.))

    def add_row(self, i):
        row=i+1 

        t = QtWidgets.QLabel(self.groupBox)
        t.setText(str(i))
        t.setAlignment(QtCore.Qt.AlignHCenter)
        self.gridLayout.addWidget(t, row, 0, QtCore.Qt.AlignHCenter)  

        self.widgets.use_flag[i] = t = QtWidgets.QCheckBox(self.groupBox)
        t.setChecked(self.roi[i].use==1)
        t.toggled.connect(functools.partial(self.menu_use, i)) # lambda expression didn't work so using functools.partial instead
        self.gridLayout.addWidget(t, row, 1, QtCore.Qt.AlignHCenter)
        
        self.widgets.centroid[i] = t = QtWidgets.QLineEdit(self.groupBox)
        t.setText('%.3f' % self.roi[i].centroid)
        t.setFixedWidth(70)
        t.setAlignment(QtCore.Qt.AlignHCenter)
        t.returnPressed.connect(functools.partial(self.menu_centroid, i))
        self.gridLayout.addWidget(t, row, 2, QtCore.Qt.AlignHCenter)

        self.widgets.fwhm[i] = t =  QtWidgets.QLineEdit(self.groupBox)
        t.setText('%.3f' % self.roi[i].fwhm)
        t.setFixedWidth(70)
        t.setAlignment(QtCore.Qt.AlignHCenter)
        self.gridLayout.addWidget(t, row, 3, QtCore.Qt.AlignHCenter)

        # If the ROI energy is zero, then try to use the label to lookup an
        # XRF line energy
        self.roi[i].energy = 0
        if (self.roi[i].energy == 0.0):
            self.roi[i].energy = Xrf.lookup_xrf_line(self.roi[i].label)
            if (self.roi[i].energy == None):
                self.roi[i].energy = Xrf.lookup_gamma_line(self.roi[i].label)
            if (self.roi[i].energy == None): self.roi[i].energy=0.0

        self.widgets.energy[i] = t = QtWidgets.QLineEdit(self.groupBox)
        t.setText('%.3f' % self.roi[i].energy)
        t.setFixedWidth(70)
        t.setAlignment(QtCore.Qt.AlignHCenter)
        t.returnPressed.connect(functools.partial(self.menu_energy, i))
        self.gridLayout.addWidget(t, row, 4, QtCore.Qt.AlignHCenter)

        self.widgets.line[i] = t = QtWidgets.QLineEdit(self.groupBox)
        t.setText(self.roi[i].label)
        t.setFixedWidth(70)
        t.setAlignment(QtCore.Qt.AlignHCenter)
        self.widgets.line[i].returnPressed.connect(functools.partial(self.menu_line, i))
        self.gridLayout.addWidget(t, row, 5, QtCore.Qt.AlignHCenter)

        self.widgets.energy_diff[i] = t = QtWidgets.QLineEdit(self.groupBox)
        t.setText('%.3f' % 0.0)
        t.setFixedWidth(70)
        t.setAlignment(QtCore.Qt.AlignHCenter)
        self.gridLayout.addWidget(t, row, 6, QtCore.Qt.AlignHCenter)

    def initUI(self):
        
        
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self)

        ### 
        ### current calibration
        ###

        self.groupBox_current_calibration = QtWidgets.QGroupBox(self)
        self.groupBox_current_calibration.setTitle("")
        self.verticalLayout_current_calibration = QtWidgets.QVBoxLayout(self.groupBox_current_calibration)
        self.gridLayout_current_calibration = QtWidgets.QGridLayout()

        self.cc_label_12 = QtWidgets.QLabel(self.groupBox_current_calibration)
        self.cc_label_12.setText("Units")
        self.gridLayout_current_calibration.addWidget(self.cc_label_12, 0, 1, 1, 1)
        self.cc_label_13 = QtWidgets.QLabel(self.groupBox_current_calibration)
        self.cc_label_13.setText("Offset")
        self.gridLayout_current_calibration.addWidget(self.cc_label_13, 0, 2, 1, 1)
        self.cc_label_14 = QtWidgets.QLabel(self.groupBox_current_calibration)
        self.cc_label_14.setText("Slope")
        self.gridLayout_current_calibration.addWidget(self.cc_label_14, 0, 3, 1, 1)
        self.cc_label_15 = QtWidgets.QLabel(self.groupBox_current_calibration)
        self.cc_label_15.setText("Quadratic")
        self.gridLayout_current_calibration.addWidget(self.cc_label_15, 0, 4, 1, 1)

        self.cc_label_11 = QtWidgets.QLabel(self.groupBox_current_calibration)
        self.cc_label_11.setText("Calibration coefficients:")
        self.cc_label_11.setAlignment(QtCore.Qt.AlignRight)
        self.gridLayout_current_calibration.addWidget(self.cc_label_11, 1, 0, 1, 1)

        self.cc_cal_units = t = QtWidgets.QLabel(self.groupBox_current_calibration)
        t.setText(self.calibration.units)
        t.setFixedWidth(90)
        self.gridLayout_current_calibration.addWidget(t, 1, 1, 1, 1)

        self.cc_cal_offset = t = QtWidgets.QLabel(self.groupBox_current_calibration)
        t.setText('%.7f'%(self.calibration.offset))
        t.setFixedWidth(90)
        self.gridLayout_current_calibration.addWidget(t, 1, 2, 1, 1)

        self.cc_cal_slope = t = QtWidgets.QLabel(self.groupBox_current_calibration)
        t.setText('%.7f'%(self.calibration.slope))
        t.setFixedWidth(90)
        self.gridLayout_current_calibration.addWidget(t, 1, 3, 1, 1)

        self.cc_cal_quad = t = QtWidgets.QLabel(self.groupBox_current_calibration)
        t.setText('%.7f'%(self.calibration.quad))
        t.setFixedWidth(90)
        self.gridLayout_current_calibration.addWidget(t, 1, 4, 1, 1)

        self.verticalLayout_current_calibration.addLayout(self.gridLayout_current_calibration)

        self.verticalLayout_4.addWidget(self.groupBox_current_calibration)

        self.groupBox = QtWidgets.QGroupBox(self)
        self.container = self.groupBox        
        self.gridLayout = QtWidgets.QGridLayout(self.container)
        self.gridLayout.setContentsMargins(7, 15, 7, 7)
        self.gridLayout.setSpacing(5)
        
        header = {'ROI':0,'Use?':1,'Centroid':2,'FWHM':3,'Energy':4,'Fluor. line':5,'Energy error':6}

        row = 0
        for key, col in header.items():
            t = QtWidgets.QLabel(self.groupBox)
            t.setText(key)
            t.setAlignment(QtCore.Qt.AlignHCenter)
            #t.setMinimumSize(QtCore.QSize(60, 0))
            self.gridLayout.addWidget(t, row, col, QtCore.Qt.AlignHCenter)


        for i in range(self.nrois):
            self.add_row(i)
            
            
        self.verticalLayout_4.addWidget(self.groupBox)

        # add button to load rois from file
        self.addRois = QtWidgets.QPushButton('Load more regions...')
        self.addRois.clicked.connect(self.add_rois_from_file)
        self.verticalLayout_4.addWidget(self.addRois)

        self.groupBox_2 = QtWidgets.QGroupBox(self)
        self.groupBox_2.setTitle("")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.lblCalibrationType = QtWidgets.QLabel(self.groupBox_2)
        self.lblCalibrationType.setText("Calibration type:")
        self.lblCalibrationType.setAlignment(QtCore.Qt.AlignRight)
        self.horizontalLayout.addWidget(self.lblCalibrationType)
        self.fit_type = QtWidgets.QComboBox(self.groupBox_2)
        self.fit_type.addItem('Linear')
        self.fit_type.addItem('Quadratic')
        self.horizontalLayout.addWidget(self.fit_type)
        
        # in QtDialong, setting default=False, autoDefault=False prevents button from default trigger by Enter key
        self.do_fit = QtWidgets.QPushButton(self.groupBox_2, default=False, autoDefault=False)

        self.do_fit.setText("Compute calibration")
        self.do_fit.clicked.connect(self.menu_do_fit)
        self.horizontalLayout.addWidget(self.do_fit)
        self.plot_cal = QtWidgets.QPushButton(self.groupBox_2, default=False, autoDefault=False)
        self.plot_cal.setText("Plot calibration error")
        self.plot_cal.clicked.connect(self.menu_plot_calibration)
        self.horizontalLayout.addWidget(self.plot_cal)

        self.plot_fwhm = QtWidgets.QPushButton(self.groupBox_2, default=False, autoDefault=False)
        self.plot_fwhm.setText("Plot FWHM")
        self.plot_fwhm.clicked.connect(self.menu_plot_fwhm)
        self.horizontalLayout.addWidget(self.plot_fwhm)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.verticalLayout_4.addWidget(self.groupBox_2)


        self.groupBox_3 = QtWidgets.QGroupBox(self)
        self.groupBox_3.setTitle("")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.gridLayout_2 = QtWidgets.QGridLayout()

        self.label_12 = QtWidgets.QLabel(self.groupBox_3)
        self.label_12.setText("Units")
        self.gridLayout_2.addWidget(self.label_12, 0, 1, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.groupBox_3)
        self.label_13.setText("Offset")
        self.gridLayout_2.addWidget(self.label_13, 0, 2, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.groupBox_3)
        self.label_14.setText("Slope")
        self.gridLayout_2.addWidget(self.label_14, 0, 3, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.groupBox_3)
        self.label_15.setText("Quadratic")
        self.gridLayout_2.addWidget(self.label_15, 0, 4, 1, 1)

        self.label_11 = QtWidgets.QLabel(self.groupBox_3)
        self.label_11.setText("Calibration coefficients:")
        self.label_11.setAlignment(QtCore.Qt.AlignRight)
        self.gridLayout_2.addWidget(self.label_11, 1, 0, 1, 1)

        self.cal_units = t = QtWidgets.QLineEdit(self.groupBox_3)
        t.setText(self.calibration.units)
        t.setFixedWidth(90)
        self.gridLayout_2.addWidget(t, 1, 1, 1, 1)

        self.cal_offset = t = QtWidgets.QLineEdit(self.groupBox_3)
        t.setText('%.7f'%(self.calibration.offset))
        t.setFixedWidth(90)
        self.gridLayout_2.addWidget(t, 1, 2, 1, 1)

        self.cal_slope = t = QtWidgets.QLineEdit(self.groupBox_3)
        t.setText('%.7f'%(self.calibration.slope))
        t.setFixedWidth(90)
        self.gridLayout_2.addWidget(t, 1, 3, 1, 1)

        self.cal_quad = t = QtWidgets.QLineEdit(self.groupBox_3)
        t.setText('%.7f'%(self.calibration.quad))
        t.setFixedWidth(90)
        self.gridLayout_2.addWidget(t, 1, 4, 1, 1)

        self.verticalLayout_2.addLayout(self.gridLayout_2)

        self.verticalLayout_4.addWidget(self.groupBox_3)

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

        self.setWindowTitle("Energy Calibration")
        self.groupBox.setTitle("Defined regions")
        self.groupBox_current_calibration.setTitle("Current calibration")
        self.groupBox_3.setTitle('New calibration')

        
        #self.setFixedSize(self.verticalLayout_4.sizeHint())

        #self.setWindowFlags(QtCore.Qt.Tool)
        #self.setAttribute(QtCore.Qt.WA_MacAlwaysShowToolWindow) 


    

    def raise_widget(self):
        self.show()
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
        self.activateWindow()
        self.raise_()

    def add_rois_from_file(self,**kwargs):
        """

        """
        filename = kwargs.get('filename', None)

        if filename is None:
            filename = open_file_dialog(self, "Load hpmca file.",
                                      self.working_directories.readdata)  

        mca = MCA()
        [_,self.success]=mca.read_file(filename)
        rois = mca.rois_from_file[0]
        nrois = copy.copy(self.nrois)
        self.widgets.add_widgets(len(rois))
        for i in range(len(rois)):
            mca.compute_roi(rois[i])
            self.roi.append(rois[i])
            self.add_row(i+nrois)
        self.nrois = self.nrois + len(rois)

        self.fwhm_chan = np.zeros(self.nrois)
        self.compute_centroids()

    def menu_plot_calibration(self):
        """ Private method """
        energy = []
        energy_diff = []
        energy_use = []
        energy_diff_use = []
        for i in range(self.nrois):
            energy.append(self.roi[i].energy)
            energy_diff.append(self.roi[i].energy -
                                self.calibration.channel_to_energy(self.roi[i].centroid))
            if (self.roi[i].use):
                energy_use.append(energy[i])
                energy_diff_use.append(energy_diff[i])
        
        pltError = pg.plot(energy_use,energy_diff_use, 
            pen=(200,200,200), symbolBrush=(255,0,0),antialias=True, 
            symbolPen='w', title="MCA Calibration"
        )
        pltError.setLabel('left', 'Calibration error')
        pltError.setLabel('bottom', 'Energy')

    def menu_plot_fwhm(self):
        """ Private method """
        energy = []
        fwhm = []
        energy_use = []
        fwhm_use = []
        for i in range(self.nrois):
            energy.append(self.roi[i].energy)
            fwhm.append(self.roi[i].fwhm)
            if (self.roi[i].use):
                energy_use.append(energy[i])
                fwhm_use.append(fwhm[i])
        pltFWHM = pg.plot(energy_use,fwhm_use, 
            pen=(200,200,200), symbolBrush=(255,0,0),antialias=True, 
            symbolPen='w', title="MCA FWHM"
        )
        pltFWHM.setLabel('left', 'FWHM')
        pltFWHM.setLabel('bottom', 'Energy')     
        
    def menu_energy(self, roi):
        """ Private method """
        energy = float(self.widgets.energy[roi].text())
        self.roi[roi].energy = energy
        self.widgets.energy[roi].setText('%.3f' % energy)
        print('energy: %.3f' % energy)

    def menu_centroid(self, roi):
        """ Private method """
        centroid = float(self.widgets.centroid[roi].text())
        self.roi[roi].centroid = centroid
        self.widgets.centroid[roi].setText('%.3f' % centroid)
        print('centroid: %.3f' % centroid )

    def menu_use(self,  roi):
        value = self.widgets.use_flag[roi].isChecked()
        """ Private method """
        self.roi[roi].use = value 
        print('use: '+str(value))
        print('roi: '+str(roi))

    def menu_line(self, roi):
        """ Private method """
        line = self.widgets.line[roi].text()
        energy = Xrf.lookup_xrf_line(line)
        if (energy == None):
            energy = Xrf.lookup_gamma_line(line)
        if (energy != None): 
            self.roi[roi].energy = energy
            self.widgets.energy[roi].setText('%.3f' % energy)
        print('line: ' + line)


    def menu_do_fit(self):
        degree = self.fit_type.currentIndex() + 1
        roi = self.roi
        calibration = self.calibration
        fit_energies(roi, degree, calibration)
         
        for i in range(self.nrois):
            # Recompute FWHM
            roi[i].fwhm = (calibration.channel_to_energy(roi[i].centroid + 
                                    self.fwhm_chan[i]/2.) - 
                            calibration.channel_to_energy(roi[i].centroid -
                                    self.fwhm_chan[i]/2.))
            energy_diff = (self.roi[i].energy -
                            self.calibration.channel_to_energy(self.roi[i].centroid))
            self.widgets.energy_diff[i].setText('%.4f' % energy_diff)
            self.widgets.fwhm[i].setText('%.3f' % self.roi[i].fwhm)

        self.cal_slope.setText(str('%.7f'%(self.calibration.slope)))
        self.cal_offset.setText(str('%.7f'%(self.calibration.offset)))
        self.cal_quad.setText(str('%.7f'%(self.calibration.quad)))

    def menu_ok_cancel(self, button):
        """ Private method """
        if button == 'OK':
            self.calibration.set_dx_type('edx')
            self.input_mca.set_calibration(self.calibration, self.det)
            exit_status = 1
        else:
            exit_status = 0
        if (self.exit_command): self.exit_command(exit_status)
        self.hide()