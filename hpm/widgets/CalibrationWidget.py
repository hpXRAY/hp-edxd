# -*- coding: utf-8 -*-
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os

from PyQt6 import QtWidgets, QtGui, QtCore
from pyqtgraph import GraphicsLayoutWidget

from .CustomWidgets import NumberTextField, LabelAlignRight, CleanLooksComboBox, SpinBoxAlignRight, \
    DoubleSpinBoxAlignRight, FlatButton

from .. import icons_path





class CalibrationControlWidget(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super(CalibrationControlWidget, self).__init__(*args, **kwargs)

        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0,0,0,0)
    
        self._file_layout = QtWidgets.QHBoxLayout()
        self.load_img_btn = FlatButton("Load File", self)
        self.load_previous_img_btn = FlatButton("<", self)
        self.load_next_img_btn = FlatButton(">", self)

        self._file_layout.addWidget(self.load_img_btn)
        self._file_layout.addWidget(self.load_previous_img_btn)
        self._file_layout.addWidget(self.load_next_img_btn)

        self._layout.addLayout(self._file_layout)

        self.filename_txt = QtWidgets.QLineEdit('', self)
        self._layout.addWidget(self.filename_txt)

        self.toolbox = QtWidgets.QToolBox()
        self.calibration_parameters_widget = CalibrationParameterWidget()
        #self.pyfai_parameters_widget = PyfaiParametersWidget()
        #self.fit2d_parameters_widget = Fit2dParametersWidget()

        self.toolbox.addItem(self.calibration_parameters_widget, "Calibration Parameters")
        #self.toolbox.addItem(self.pyfai_parameters_widget, 'pyFAI Parameters')
        #self.toolbox.addItem(self.fit2d_parameters_widget, 'Fit2d Parameters')
        self._layout.addWidget(self.toolbox)

        self._bottom_layout = QtWidgets.QHBoxLayout()
        self.load_calibration_btn = FlatButton('Load Calibration')
        self.save_calibration_btn = FlatButton('Save Calibration')
        self._bottom_layout.addWidget(self.load_calibration_btn)
        self._bottom_layout.addWidget(self.save_calibration_btn)
        self._layout.addLayout(self._bottom_layout)

        self.style_widgets()

    def style_widgets(self):
        self.load_previous_img_btn.setMaximumWidth(50)
        self.load_next_img_btn.setMaximumWidth(50)
        self.setMaximumWidth(290)
        self.setMinimumWidth(290)


class CalibrationParameterWidget(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super(CalibrationParameterWidget, self).__init__(*args, **kwargs)

        self._layout = QtWidgets.QVBoxLayout(self)

        self.start_values_gb = StartValuesGroupBox(self)
        self.peak_selection_gb = PeakSelectionGroupBox()
        self.refinement_options_gb = RefinementOptionsGroupBox()

        self._layout.addWidget(self.start_values_gb)
        self._layout.addWidget(self.peak_selection_gb)
        self._layout.addWidget(self.refinement_options_gb)
        self._layout.addSpacerItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Policy.Expanding,
                                                     QtWidgets.QSizePolicy.Policy.Expanding))

        self.setLayout(self._layout)


class StartValuesGroupBox(QtWidgets.QGroupBox):
    def __init__(self, *args, **kwargs):
        super(StartValuesGroupBox, self).__init__('Start values', *args, **kwargs)
        self.setMaximumWidth(260)
        self._layout = QtWidgets.QVBoxLayout(self)

        self._grid_layout1 = QtWidgets.QGridLayout()

        self._grid_layout1.addWidget(LabelAlignRight('Distance:'), 0, 0)
        self.distance_txt = NumberTextField('1000')
        self.distance_cb = QtWidgets.QCheckBox()
        self.distance_cb.setChecked(True)
        self._grid_layout1.addWidget(self.distance_txt, 0, 1)
        self._grid_layout1.addWidget(QtWidgets.QLabel('mm'), 0, 2)
        self._grid_layout1.addWidget(self.distance_cb, 0, 3)

        self._grid_layout1.addWidget(LabelAlignRight('2 theta:'), 1, 0)
        self.two_theta_txt = NumberTextField('15')
        self.two_theta_cb = QtWidgets.QCheckBox()
        self._grid_layout1.addWidget(self.two_theta_txt, 1, 1)
        self._grid_layout1.addWidget(QtWidgets.QLabel('A'), 1, 2)
        self._grid_layout1.addWidget(self.two_theta_cb, 1, 3)

        '''self._grid_layout1.addWidget(LabelAlignRight('Polarization:'), 2, 0)
        self.polarization_txt = NumberTextField('0.99')
        self._grid_layout1.addWidget(self.polarization_txt, 2, 1)'''

        self._grid_layout1.addWidget(LabelAlignRight('Pixel width:'), 3, 0)
        self.pixel_width_txt = NumberTextField('72')
        self._grid_layout1.addWidget(self.pixel_width_txt, 3, 1)
        self._grid_layout1.addWidget(QtWidgets.QLabel('um'))

        '''self._grid_layout1.addWidget(LabelAlignRight('Pixel height:'), 4, 0)
        self.pixel_height_txt = NumberTextField('72')
        self._grid_layout1.addWidget(self.pixel_height_txt, 4, 1)
        self._grid_layout1.addWidget(QtWidgets.QLabel('um'))'''

        self._grid_layout1.addWidget(LabelAlignRight('Calibrant:'), 5, 0)
        self.calibrant_cb = CleanLooksComboBox()
        self._grid_layout1.addWidget(self.calibrant_cb, 5, 1, 1, 2)

        self._grid_layout2 = QtWidgets.QGridLayout()
        self._grid_layout2.setSpacing(6)

        '''self.rotate_p90_btn = FlatButton('Rotate +90')
        self.rotate_m90_btn = FlatButton('Rotate -90', self)
        self._grid_layout2.addWidget(self.rotate_p90_btn, 1, 0)
        self._grid_layout2.addWidget(self.rotate_m90_btn, 1, 1)'''

        '''self.flip_horizontal_btn = FlatButton('Flip horizontal', self)'''
        self.flip_vertical_btn = FlatButton('Flip vertical', self)
        '''self._grid_layout2.addWidget(self.flip_horizontal_btn, 2, 0)'''
        self._grid_layout2.addWidget(self.flip_vertical_btn, 2, 1)

        self.reset_transformations_btn = FlatButton('Reset transformations', self)
        self._grid_layout2.addWidget(self.reset_transformations_btn, 3, 0, 1, 2)

        self._layout.addLayout(self._grid_layout1)
        self._layout.addLayout(self._grid_layout2)

        self.setLayout(self._layout)


class PeakSelectionGroupBox(QtWidgets.QGroupBox):
    def __init__(self):
        super(PeakSelectionGroupBox, self).__init__('Peak Selection')
        self.setMaximumWidth(260)
        self._layout = QtWidgets.QGridLayout()

        
        self.pick_peaks_cb = QtWidgets.QCheckBox('Pick peaks')
        self.pick_peaks_cb.setChecked(True)
        self._layout.addWidget(self.pick_peaks_cb, 0, 0, 1, 3)


        self._layout.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Policy.Expanding,
                                               QtWidgets.QSizePolicy.Policy.Minimum), 1, 0)
        self._layout.addWidget(LabelAlignRight('Current Ring Number:'), 1, 0, 1, 3)
        self.peak_num_sb = SpinBoxAlignRight()
        self.peak_num_sb.setValue(1)
        self.peak_num_sb.setMinimum(1)
        self._layout.addWidget(self.peak_num_sb, 1, 3)

        self._layout.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Policy.Expanding,
                                               QtWidgets.QSizePolicy.Policy.Minimum), 2, 0, 1, 2)
        self.automatic_peak_num_inc_cb = QtWidgets.QCheckBox('automatic increase')
        self.automatic_peak_num_inc_cb.setChecked(True)
        self._layout.addWidget(self.automatic_peak_num_inc_cb, 2, 2, 1, 2)

        self.automatic_peak_search_rb = QtWidgets.QRadioButton('automatic peak search')
        self.automatic_peak_search_rb.setChecked(True)
        self.select_peak_rb = QtWidgets.QRadioButton('single peak search')
        self._layout.addWidget(self.automatic_peak_search_rb, 3, 0, 1, 4)
        self._layout.addWidget(self.select_peak_rb, 4, 0, 1, 4)

        self._layout.addWidget(LabelAlignRight('Search size:'), 5, 0, 1,3)
        self.search_size_sb = SpinBoxAlignRight()
        self.search_size_sb.setValue(10)
        self.search_size_sb.setMaximumWidth(50)
        self._layout.addWidget(self.search_size_sb, 5, 3, 1, 2)
        #self._layout.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Policy.Expanding,
        #                                       QtWidgets.QSizePolicy.Policy.Minimum), 4, 3, 1, 2)

        self.undo_peaks_btn = FlatButton("Undo")
        self.clear_peaks_btn = FlatButton("Clear All Peaks")

        self._peak_btn_layout = QtWidgets.QHBoxLayout()
        self._peak_btn_layout.addWidget(self.undo_peaks_btn)
        self._peak_btn_layout.addWidget(self.clear_peaks_btn)
        self._layout.addLayout(self._peak_btn_layout, 6, 0, 1, 4)

        self.setLayout(self._layout)


class RefinementOptionsGroupBox(QtWidgets.QGroupBox):
    def __init__(self):
        super(RefinementOptionsGroupBox, self).__init__('Refinement Options')
        self.setMaximumWidth(260)
        self._layout = QtWidgets.QGridLayout()

        self.automatic_refinement_cb = QtWidgets.QCheckBox('automatic refinement')
        self.automatic_refinement_cb.setChecked(True)
        self._layout.addWidget(self.automatic_refinement_cb, 0, 0, 1, 1)

        self.use_mask_cb = QtWidgets.QCheckBox('use mask')
        self._layout.addWidget(self.use_mask_cb, 1, 0)

        self.mask_transparent_cb = QtWidgets.QCheckBox('transparent')
        self._layout.addWidget(self.mask_transparent_cb, 1, 1)

        self._layout.addWidget(LabelAlignRight('Peak Search Algorithm:'), 2, 0)
        self.peak_search_algorithm_cb = CleanLooksComboBox()
        self.peak_search_algorithm_cb.addItems(['Massif', 'Blob'])
        self._layout.addWidget(self.peak_search_algorithm_cb, 2, 1)

        self._layout.addWidget(LabelAlignRight('Delta 2Th:'), 3, 0)
        self.delta_tth_txt = NumberTextField('0.1')
        self._layout.addWidget(self.delta_tth_txt, 3, 1)

        self._layout.addWidget(LabelAlignRight('Intensity Mean Factor:'), 4, 0)
        self.intensity_mean_factor_sb = DoubleSpinBoxAlignRight()
        self.intensity_mean_factor_sb.setValue(3.0)
        self.intensity_mean_factor_sb.setSingleStep(0.1)
        self._layout.addWidget(self.intensity_mean_factor_sb, 4, 1)

        self._layout.addWidget(LabelAlignRight('Intensity Limit:'), 5, 0)
        self.intensity_limit_txt = NumberTextField('55000')
        self._layout.addWidget(self.intensity_limit_txt, 5, 1)

        self._layout.addWidget(LabelAlignRight('Number of lines:'), 6, 0)
        self.number_of_rings_sb = SpinBoxAlignRight()
        self.number_of_rings_sb.setValue(5)
        self._layout.addWidget(self.number_of_rings_sb, 6, 1)

        self.setLayout(self._layout)





'''class PyfaiParametersWidget(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super(PyfaiParametersWidget, self).__init__(*args, **kwargs)

        self._layout = QtWidgets.QGridLayout()

        self._layout.addWidget(LabelAlignRight('Distance:'), 0, 0)
        self.distance_txt = NumberTextField()
        self.distance_cb = QtWidgets.QCheckBox()
        self.distance_cb.setChecked(True)
        self._layout.addWidget(self.distance_txt, 0, 1)
        self._layout.addWidget(QtWidgets.QLabel('mm'), 0, 2)
        self._layout.addWidget(self.distance_cb, 0, 3)

        self._layout.addWidget(LabelAlignRight('Wavelength:'), 1, 0)
        self.two_theta_txt = NumberTextField()
        self.two_theta_cb = QtWidgets.QCheckBox()
        self._layout.addWidget(self.two_theta_txt, 1, 1)
        self._layout.addWidget(QtWidgets.QLabel('A'), 1, 2)
        self._layout.addWidget(self.two_theta_cb, 1, 3)

        self._layout.addWidget(LabelAlignRight('Polarization:'), 2, 0)
        self.polarization_txt = NumberTextField()
        self._layout.addWidget(self.polarization_txt, 2, 1)

        self._layout.addWidget(LabelAlignRight('PONI:'), 3, 0)
        self.poni1_txt = NumberTextField()
        self.poni1_cb = QtWidgets.QCheckBox()
        self.poni1_cb.setChecked(True)
        self._layout.addWidget(self.poni1_txt, 3, 1)
        self._layout.addWidget(QtWidgets.QLabel('m'), 3, 2)
        self._layout.addWidget(self.poni1_cb, 3, 3)

        self.poni2_txt = NumberTextField()
        self.poni2_cb = QtWidgets.QCheckBox()
        self.poni2_cb.setChecked(True)
        self._layout.addWidget(self.poni2_txt, 4, 1)
        self._layout.addWidget(QtWidgets.QLabel('m'), 4, 2)
        self._layout.addWidget(self.poni2_cb, 4, 3)

        self._layout.addWidget(LabelAlignRight('Rotations'), 5, 0)
        self.rotation1_txt = NumberTextField()
        self.rotation2_txt = NumberTextField()
        self.rotation3_txt = NumberTextField()
        self.rotation1_cb = QtWidgets.QCheckBox()
        self.rotation2_cb = QtWidgets.QCheckBox()
        self.rotation3_cb = QtWidgets.QCheckBox()
        self.rotation1_cb.setChecked(True)
        self.rotation2_cb.setChecked(True)
        self.rotation3_cb.setChecked(True)
        self._layout.addWidget(self.rotation1_txt, 5, 1)
        self._layout.addWidget(self.rotation2_txt, 6, 1)
        self._layout.addWidget(self.rotation3_txt, 7, 1)
        self._layout.addWidget(QtWidgets.QLabel('rad'), 5, 2)
        self._layout.addWidget(QtWidgets.QLabel('rad'), 6, 2)
        self._layout.addWidget(QtWidgets.QLabel('rad'), 7, 2)
        self._layout.addWidget(self.rotation1_cb, 5, 3)
        self._layout.addWidget(self.rotation2_cb, 6, 3)
        self._layout.addWidget(self.rotation3_cb, 7, 3)

        self._layout.addWidget(LabelAlignRight('Pixel width:'), 8, 0)
        self.pixel_width_txt = NumberTextField()
        self._layout.addWidget(self.pixel_width_txt, 8, 1)
        self._layout.addWidget(QtWidgets.QLabel('um'))

        self._layout.addWidget(LabelAlignRight('Pixel height:'), 9, 0)
        self.pixel_height_txt = NumberTextField()
        self._layout.addWidget(self.pixel_height_txt, 9, 1)
        self._layout.addWidget(QtWidgets.QLabel('um'))

        self.update_btn = FlatButton('update')
        self._layout.addWidget(self.update_btn, 10, 0, 1, 4)

        self._layout.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding),
                             11, 0, 1, 4)

        self.setLayout(self._layout)'''


'''class Fit2dParametersWidget(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super(Fit2dParametersWidget, self).__init__(*args, **kwargs)

        self._layout = QtWidgets.QGridLayout()

        self._layout.addWidget(LabelAlignRight('Distance:'), 0, 0)
        self.distance_txt = NumberTextField()
        self.distance_cb = QtWidgets.QCheckBox()
        self.distance_cb.setChecked(True)
        self._layout.addWidget(self.distance_txt, 0, 1)
        self._layout.addWidget(QtWidgets.QLabel('mm'), 0, 2)
        self._layout.addWidget(self.distance_cb, 0, 3)

        self._layout.addWidget(LabelAlignRight('Wavelength:'), 1, 0)
        self.two_theta_txt = NumberTextField()
        self.two_theta_cb = QtWidgets.QCheckBox()
        self._layout.addWidget(self.two_theta_txt, 1, 1)
        self._layout.addWidget(QtWidgets.QLabel('A'), 1, 2)
        self._layout.addWidget(self.two_theta_cb, 1, 3)

        self._layout.addWidget(LabelAlignRight('Polarization:'), 2, 0)
        self.polarization_txt = NumberTextField()
        self._layout.addWidget(self.polarization_txt, 2, 1)

        self._layout.addWidget(LabelAlignRight('Center X:'), 3, 0)
        self.center_x_txt = NumberTextField()
        self._layout.addWidget(self.center_x_txt, 3, 1)
        self._layout.addWidget(QtWidgets.QLabel('px'), 3, 2)

        self._layout.addWidget(LabelAlignRight('Center Y:'), 4, 0)
        self.center_y_txt = NumberTextField()
        self._layout.addWidget(self.center_y_txt, 4, 1)
        self._layout.addWidget(QtWidgets.QLabel('px'), 4, 2)

        self._layout.addWidget(LabelAlignRight('Rotation:'), 5, 0)
        self.rotation_txt = NumberTextField()
        self._layout.addWidget(self.rotation_txt, 5, 1)
        self._layout.addWidget(QtWidgets.QLabel('deg'), 5, 2)

        self._layout.addWidget(LabelAlignRight('Tilt:'), 6, 0)
        self.tilt_txt = NumberTextField()
        self._layout.addWidget(self.tilt_txt, 6, 1)
        self._layout.addWidget(QtWidgets.QLabel('deg'), 6, 2)

        self._layout.addWidget(LabelAlignRight('Pixel width:'), 8, 0)
        self.pixel_width_txt = NumberTextField()
        self._layout.addWidget(self.pixel_width_txt, 8, 1)
        self._layout.addWidget(QtWidgets.QLabel('um'))

        self._layout.addWidget(LabelAlignRight('Pixel height:'), 9, 0)
        self.pixel_height_txt = NumberTextField()
        self._layout.addWidget(self.pixel_height_txt, 9, 1)
        self._layout.addWidget(QtWidgets.QLabel('um'))

        self.update_btn = FlatButton('update')
        self._layout.addWidget(self.update_btn, 10, 0, 1, 4)

        self._layout.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding),
                             11, 0, 1, 4)

        self.setLayout(self._layout)'''
