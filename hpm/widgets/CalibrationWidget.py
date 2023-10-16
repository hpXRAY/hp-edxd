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

from PyQt5 import QtWidgets, QtGui, QtCore
from pyqtgraph import GraphicsLayoutWidget

from .CustomWidgets import NumberTextField, LabelAlignRight, CleanLooksComboBox, SpinBoxAlignRight, \
    DoubleSpinBoxAlignRight, FlatButton

from .. import icons_path


class CalibrationWidget(QtWidgets.QWidget):
    """
    Defines the main structure of the calibration widget, which is separated into two parts.
    Calibration Display Widget - shows the image and pattern
    Calibration Control Widget - shows all the controls on the right side of the widget
    """

    def __init__(self, *args, **kwargs):
        super(CalibrationWidget, self).__init__(*args, **kwargs)

        self.calibration_control_widget = CalibrationControlWidget(self)

        self._layout = QtWidgets.QHBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self.calibration_control_widget)
        self.setLayout(self._layout)

        self.create_shortcuts()
        

    def create_shortcuts(self):
        """
        Creates shortcuts for the widgets which are directly interfacing with the controller.
        """
        self.load_img_btn = self.calibration_control_widget.load_img_btn
        self.load_next_img_btn = self.calibration_control_widget.load_next_img_btn
        self.load_previous_img_btn = self.calibration_control_widget.load_previous_img_btn
        self.filename_txt = self.calibration_control_widget.filename_txt

        self.save_calibration_btn = self.calibration_control_widget.save_calibration_btn
        self.load_calibration_btn = self.calibration_control_widget.load_calibration_btn

        self.ToolBox = self.calibration_control_widget.toolbox

        sv_gb = self.calibration_control_widget.calibration_parameters_widget.start_values_gb
        #self.rotate_m90_btn = sv_gb.rotate_m90_btn
        #self.rotate_p90_btn = sv_gb.rotate_p90_btn
        #self.invert_horizontal_btn = sv_gb.flip_horizontal_btn
        self.invert_vertical_btn = sv_gb.flip_vertical_btn
        self.reset_transformations_btn = sv_gb.reset_transformations_btn
        self.calibrant_cb = sv_gb.calibrant_cb

        self.sv_two_theta_txt = sv_gb.two_theta_txt
        self.sv_two_theta_cb = sv_gb.two_theta_cb
        self.sv_distance_txt = sv_gb.distance_txt
        self.sv_distance_cb = sv_gb.distance_cb
        #self.sv_polarisation_txt = sv_gb.polarization_txt
        self.sv_pixel_width_txt = sv_gb.pixel_width_txt
        #self.sv_pixel_height_txt = sv_gb.pixel_height_txt

        refinement_options_gb = self.calibration_control_widget.calibration_parameters_widget.refinement_options_gb
        self.use_mask_cb = refinement_options_gb.use_mask_cb
        self.mask_transparent_cb = refinement_options_gb.mask_transparent_cb
        self.options_automatic_refinement_cb = refinement_options_gb.automatic_refinement_cb
        self.options_num_rings_sb = refinement_options_gb.number_of_rings_sb
        self.options_peaksearch_algorithm_cb = refinement_options_gb.peak_search_algorithm_cb
        self.options_delta_tth_txt = refinement_options_gb.delta_tth_txt
        self.options_intensity_mean_factor_sb = refinement_options_gb.intensity_mean_factor_sb
        self.options_intensity_limit_txt = refinement_options_gb.intensity_limit_txt

        peak_selection_gb = self.calibration_control_widget.calibration_parameters_widget.peak_selection_gb
        self.peak_num_sb = peak_selection_gb.peak_num_sb
        self.automatic_peak_search_rb = peak_selection_gb.automatic_peak_search_rb
        self.select_peak_rb = peak_selection_gb.select_peak_rb
        self.search_size_sb = peak_selection_gb.search_size_sb
        self.automatic_peak_num_inc_cb = peak_selection_gb.automatic_peak_num_inc_cb
        self.clear_peaks_btn = peak_selection_gb.clear_peaks_btn
        self.undo_peaks_btn = peak_selection_gb.undo_peaks_btn

        '''self.f2_update_btn = self.calibration_control_widget.fit2d_parameters_widget.update_btn
        self.pf_update_btn = self.calibration_control_widget.pyfai_parameters_widget.update_btn

        self.f2_two_theta_cb = self.calibration_control_widget.fit2d_parameters_widget.two_theta_cb
        self.pf_two_theta_cb = self.calibration_control_widget.pyfai_parameters_widget.two_theta_cb

        self.f2_distance_cb = self.calibration_control_widget.fit2d_parameters_widget.distance_cb
        self.pf_distance_cb = self.calibration_control_widget.pyfai_parameters_widget.distance_cb

        self.pf_poni1_cb = self.calibration_control_widget.pyfai_parameters_widget.poni1_cb
        self.pf_poni2_cb = self.calibration_control_widget.pyfai_parameters_widget.poni2_cb
        self.pf_rot1_cb = self.calibration_control_widget.pyfai_parameters_widget.rotation1_cb
        self.pf_rot2_cb = self.calibration_control_widget.pyfai_parameters_widget.rotation2_cb
        self.pf_rot3_cb = self.calibration_control_widget.pyfai_parameters_widget.rotation3_cb'''
     

    def set_img_filename(self, filename):
        self.filename_txt.setText(os.path.basename(filename))

    

    def set_calibration_parameters(self, pyFAI_parameter, fit2d_parameter):
        print('TODO: set_calibration_parameters')




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
        self._layout.addSpacerItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding,
                                                     QtWidgets.QSizePolicy.Expanding))

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
        self._layout.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding,
                                               QtWidgets.QSizePolicy.Minimum), 0, 0)
        self._layout.addWidget(LabelAlignRight('Current Ring Number:'), 0, 0, 1, 3)
        self.peak_num_sb = SpinBoxAlignRight()
        self.peak_num_sb.setValue(1)
        self.peak_num_sb.setMinimum(1)
        self._layout.addWidget(self.peak_num_sb, 0, 3)

        self._layout.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding,
                                               QtWidgets.QSizePolicy.Minimum), 1, 0, 1, 2)
        self.automatic_peak_num_inc_cb = QtWidgets.QCheckBox('automatic increase')
        self.automatic_peak_num_inc_cb.setChecked(True)
        self._layout.addWidget(self.automatic_peak_num_inc_cb, 1, 2, 1, 2)

        self.automatic_peak_search_rb = QtWidgets.QRadioButton('automatic peak search')
        self.automatic_peak_search_rb.setChecked(True)
        self.select_peak_rb = QtWidgets.QRadioButton('single peak search')
        self._layout.addWidget(self.automatic_peak_search_rb, 2, 0, 1, 4)
        self._layout.addWidget(self.select_peak_rb, 3, 0, 1, 4)

        self._layout.addWidget(LabelAlignRight('Search size:'), 4, 0, 1,3)
        self.search_size_sb = SpinBoxAlignRight()
        self.search_size_sb.setValue(10)
        self.search_size_sb.setMaximumWidth(50)
        self._layout.addWidget(self.search_size_sb, 4, 3, 1, 2)
        #self._layout.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding,
        #                                       QtWidgets.QSizePolicy.Minimum), 4, 3, 1, 2)

        self.undo_peaks_btn = FlatButton("Undo")
        self.clear_peaks_btn = FlatButton("Clear All Peaks")

        self._peak_btn_layout = QtWidgets.QHBoxLayout()
        self._peak_btn_layout.addWidget(self.undo_peaks_btn)
        self._peak_btn_layout.addWidget(self.clear_peaks_btn)
        self._layout.addLayout(self._peak_btn_layout, 5, 0, 1, 4)

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

        self._layout.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding),
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

        self._layout.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding),
                             11, 0, 1, 4)

        self.setLayout(self._layout)'''
