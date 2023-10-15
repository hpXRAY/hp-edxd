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
from qtpy import QtWidgets, QtCore

import numpy as np

from ..widgets.UtilityWidgets import open_file_dialog, save_file_dialog
from .. import calibrants_path

# imports for type hinting in PyCharm -- DO NOT DELETE
from ..widgets.CalibrationWidget import CalibrationWidget
from ..widgets.UtilityWidgets import open_file_dialog
from ..model.DioptasModel import DioptasModel
from ..model.CalibrationModel import NotEnoughSpacingsInCalibrant


class CalibrationController(object):
    """
    CalibrationController handles all the interaction between the CalibrationView and the CalibrationData class
    """

    def __init__(self, widget, dioptas_model):
        """Manages the connection between the calibration GUI and data

        :param widget: Gives the Calibration Widget
        :type widget: CalibrationWidget

        :param dioptas_model: Reference to DioptasModel Object
        :type dioptas_model: DioptasModel

        """
        self.widget = widget
        self.model = dioptas_model

        self.widget.set_start_values(self.model.calibration_model.start_values)
        self._first_plot = True
        self.create_signals()
        self.plot_image()
        self.load_calibrants_list()

    def create_signals(self):
        """
        Connects the GUI signals to the appropriate Controller methods.
        """
        self.model.img_changed.connect(self.plot_image)
        self.model.configuration_selected.connect(self.update_calibration_parameter_in_view)

        self.create_transformation_signals()
        self.create_update_signals()
        self.create_mouse_signals()

        self.widget.calibrant_cb.currentIndexChanged.connect(self.load_calibrant)
        self.widget.load_img_btn.clicked.connect(self.load_img)
        self.widget.load_next_img_btn.clicked.connect(self.load_next_img)
        self.widget.load_previous_img_btn.clicked.connect(self.load_previous_img)
        self.widget.filename_txt.editingFinished.connect(self.update_filename_txt)

        self.widget.save_calibration_btn.clicked.connect(self.save_calibration)
        self.widget.load_calibration_btn.clicked.connect(self.load_calibration)
        self.widget.calibrate_btn.clicked.connect(self.calibrate)
        self.widget.refine_btn.clicked.connect(self.refine)

        self.widget.clear_peaks_btn.clicked.connect(self.clear_peaks_btn_click)
        self.widget.undo_peaks_btn.clicked.connect(self.undo_peaks_btn_clicked)

        self.widget.load_spline_btn.clicked.connect(self.load_spline_btn_click)
        self.widget.spline_reset_btn.clicked.connect(self.reset_spline_btn_click)

        self.widget.f2_wavelength_cb.stateChanged.connect(self.wavelength_cb_changed)
        self.widget.pf_wavelength_cb.stateChanged.connect(self.wavelength_cb_changed)
        self.widget.sv_wavelength_cb.stateChanged.connect(self.wavelength_cb_changed)

        self.widget.f2_distance_cb.stateChanged.connect(self.distance_cb_changed)
        self.widget.pf_distance_cb.stateChanged.connect(self.distance_cb_changed)
        self.widget.sv_distance_cb.stateChanged.connect(self.distance_cb_changed)

        self.widget.pf_poni1_cb.stateChanged.connect(self.poni1_cb_changed)
        self.widget.pf_poni2_cb.stateChanged.connect(self.poni2_cb_changed)
        self.widget.pf_rot1_cb.stateChanged.connect(self.rot1_cb_changed)
        self.widget.pf_rot2_cb.stateChanged.connect(self.rot2_cb_changed)
        self.widget.pf_rot3_cb.stateChanged.connect(self.rot3_cb_changed)

        self.widget.use_mask_cb.stateChanged.connect(self.plot_mask)
        self.widget.mask_transparent_cb.stateChanged.connect(self.mask_transparent_status_changed)

    def create_transformation_signals(self):
        """
        Connects all the rotation GUI controls.
        """
 
        self.widget.reset_transformations_btn.clicked.connect(self.reset_transformations_btn_clicked)



    def reset_transformations_btn_clicked(self):
        self.model.img_model.reset_img_transformations()
        self.clear_peaks_btn_click()

    

    def update_f2_btn_click(self):
        """
        Takes all parameters inserted into the fit2d txt-fields and updates the current calibration accordingly.
        """
        fit2d_parameter = self.widget.get_fit2d_parameter()
        self.model.calibration_model.set_fit2d(fit2d_parameter)
        self.update_all()

    def update_pyFAI_btn_click(self):
        """
        Takes all parameters inserted into the fit2d txt-fields and updates the current calibration accordingly.
        """
        pyFAI_parameter = self.widget.get_pyFAI_parameter()
        self.model.calibration_model.set_pyFAI(pyFAI_parameter)
        self.update_all()



    def load_calibrants_list(self):
        """
        Loads all calibrants from the ExampleData/calibrants directory into the calibrants combobox. And loads number 7.
        """
        self._calibrants_file_list = []
        self._calibrants_file_names_list = []
        for file in os.listdir(calibrants_path):
            if file.endswith('.D'):
                self._calibrants_file_list.append(file)
                self._calibrants_file_names_list.append(file.split('.')[:-1][0])
        self._calibrants_file_list.sort()
        self._calibrants_file_names_list.sort()
        self.widget.calibrant_cb.blockSignals(True)
        self.widget.calibrant_cb.clear()
        self.widget.calibrant_cb.addItems(self._calibrants_file_names_list)
        self.widget.calibrant_cb.blockSignals(False)
        self.widget.calibrant_cb.setCurrentIndex(self._calibrants_file_names_list.index('LaB6'))  # to LaB6
        self.load_calibrant()

    def load_calibrant(self, wavelength_from='start_values'):
        """
        Loads the selected calibrant in the calibrant combobox into the calibration data.
        :param wavelength_from: determines which wavelength to use possible values: "start_values", "pyFAI"
        """
        current_index = self.widget.calibrant_cb.currentIndex()
        filename = os.path.join(self.model.calibration_model._calibrants_working_dir,
                                self._calibrants_file_list[current_index])
        self.model.calibration_model.set_calibrant(filename)

        if wavelength_from == 'start_values':
            start_values = self.widget.get_start_values()
            wavelength = start_values['wavelength']
        elif wavelength_from == 'pyFAI':
            pyFAI_parameter, _ = self.model.calibration_model.get_calibration_parameter()
            if pyFAI_parameter['wavelength'] is not 0:
                wavelength = pyFAI_parameter['wavelength']
            else:
                start_values = self.widget.get_start_values()
                wavelength = start_values['wavelength']
        else:
            start_values = self.widget.get_start_values()
            wavelength = start_values['wavelength']

        self.model.calibration_model.calibrant.setWavelength_change2th(wavelength)
        try:
            integration_unit = self.model.current_configuration.integration_unit
        except:
            integration_unit = '2th_deg'

        calibrant_line_positions = self.convert_x_value(
            np.array(self.model.calibration_model.calibrant.get_2th()) / np.pi * 180, '2th_deg', integration_unit,
            wavelength)
        # filter them to only show the ones visible with the current pattern
        if len(self.model.pattern.x) > 0:
            pattern_min = np.min(self.model.pattern.x)
            pattern_max = np.max(self.model.pattern.x)
            calibrant_line_positions = calibrant_line_positions[calibrant_line_positions > pattern_min]
            calibrant_line_positions = calibrant_line_positions[calibrant_line_positions < pattern_max]
            self.widget.pattern_widget.plot_vertical_lines(positions=calibrant_line_positions,
                                                           name=self._calibrants_file_names_list[current_index])

    def set_calibrant(self, index):
        """
        :param index:
            index of a specific calibrant in the calibrant combobox
        """
        self.widget.calibrant_cb.setCurrentIndex(index)
        self.load_calibrant()

    def plot_image(self):
        """
        Plots the current image loaded in img_data and autoscales the intensity.
        :return:
        """
        self.widget.img_widget.plot_image(self.model.img_data, True)
        self.widget.img_widget.auto_level()
        self.widget.set_img_filename(self.model.img_model.filename)

    def search_peaks(self, x, y):
        """
        Searches peaks around a specific points (x,y) in the current image file. The algorithm for searching
        (either automatic or single peaksearch) is set in the GUI.
        :param x:
            x-Position for the search.
        :param y:
            y-Position for the search
        """
        x, y = y, x  # indeces for the img array are transposed compared to the mouse position

        # convert pixel coord into pixel index
        x, y = int(x), int(y)

        # filter events outside the image
        shape = self.model.img_model.img_data.shape
        if not (0 <= x < shape[0]):
            return
        if not (0 <= y < shape[1]):
            return

        peak_ind = self.widget.peak_num_sb.value()
        if self.widget.automatic_peak_search_rb.isChecked():
            points = self.model.calibration_model.find_peaks_automatic(x, y, peak_ind - 1)
        else:
            search_size = np.int(self.widget.search_size_sb.value())
            points = self.model.calibration_model.find_peak(x, y, search_size, peak_ind - 1)
        if len(points):
            self.plot_points(points)
            if self.widget.automatic_peak_num_inc_cb.checkState():
                self.widget.peak_num_sb.setValue(peak_ind + 1)

    def plot_points(self, points=None):
        """
        Plots points into the image view.
        :param points:
            list of points, whereby a point is a [x,y] element. If it is none it will plot the points stored in the
            calibration_data
        """
        if points is None:
            try:
                points = self.model.calibration_model.get_point_array()
            except IndexError:
                points = []
        if len(points):
            self.widget.img_widget.add_scatter_data(points[:, 0] + 0.5, points[:, 1] + 0.5)

    def clear_peaks_btn_click(self):
        """
        Deletes all points/peaks in the calibration_data and in the gui.
        :return:
        """
        self.model.calibration_model.clear_peaks()
        self.widget.img_widget.clear_scatter_plot()
        self.widget.peak_num_sb.setValue(1)

    def undo_peaks_btn_clicked(self):
        """
        undoes clicked peaks
        """
        num_points = self.model.calibration_model.remove_last_peak()
        self.widget.img_widget.remove_last_scatter_points(num_points)
        if self.widget.automatic_peak_num_inc_cb.isChecked():
            self.widget.peak_num_sb.setValue(self.widget.peak_num_sb.value() - 1)



    def calibrate(self):
        """
        Performs calibration based on the previously inputted/searched peaks and start values.
        """
        self.load_calibrant()  # load the right calibration file...
        self.model.calibration_model.set_start_values(self.widget.get_start_values())
        progress_dialog = self.create_progress_dialog('Calibrating.', '', 0, show_cancel_btn=False)
        self.model.calibration_model.calibrate()

        progress_dialog.close()

        if self.widget.options_automatic_refinement_cb.isChecked():
            self.refine()
        else:
            self.update_all()
        self.update_calibration_parameter_in_view()

    def create_progress_dialog(self, text_str, abort_str, end_value, show_cancel_btn=True):
        """ Creates a Progress Bar Dialog.
        :param text_str:  Main message string
        :param abort_str:  Text on the abort button
        :param end_value:  Number of steps for which the progressbar is being used
        :param show_cancel_btn: Whether the cancel button should be shown.
        :return: ProgressDialog reference which is already shown in the interface
        :rtype: QtWidgets.ProgressDialog
        """
        progress_dialog = QtWidgets.QProgressDialog(text_str, abort_str, 0, end_value,
                                                    self.widget)

        progress_dialog.move(int(self.widget.tab_widget.x() + self.widget.tab_widget.size().width() / 2.0 - \
                                 progress_dialog.size().width() / 2.0),
                             int(self.widget.tab_widget.y() + self.widget.tab_widget.size().height() / 2.0 -
                                 progress_dialog.size().height() / 2.0))

        progress_dialog.setWindowTitle('   ')
        progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        progress_dialog.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        if not show_cancel_btn:
            progress_dialog.setCancelButton(None)
        progress_dialog.show()
        QtWidgets.QApplication.processEvents()
        return progress_dialog

    def refine(self):
        """
        Refines the current calibration parameters by searching peaks in the approximate positions and subsequent
        refinement. Parameters for this search are set in the GUI.
        """

        # Basic Algorithm:
        # search peaks on first and second ring
        #   calibrate based on those two rings
        #   repeat until ring_ind = max_ind:
        #       search next ring
        #       calibrate based on all previous found points

        num_rings = self.widget.options_num_rings_sb.value()

        progress_dialog = self.create_progress_dialog("Refining Calibration.", 'Abort', num_rings)
        self.clear_peaks_btn_click()
        self.load_calibrant(wavelength_from='pyFAI')  # load right calibration file

        # get options
        algorithm = str(self.widget.options_peaksearch_algorithm_cb.currentText())
        delta_tth = np.float(self.widget.options_delta_tth_txt.text())
        intensity_min_factor = np.float(self.widget.options_intensity_mean_factor_sb.value())
        intensity_max = np.float(self.widget.options_intensity_limit_txt.text())

        self.model.calibration_model.setup_peak_search_algorithm(algorithm)

        if self.widget.use_mask_cb.isChecked():
            mask = self.model.mask_model.get_img()
        else:
            mask = None

        self.model.calibration_model.search_peaks_on_ring(0, delta_tth, intensity_min_factor, intensity_max, mask)
        self.widget.peak_num_sb.setValue(2)
        progress_dialog.setValue(1)
        self.model.calibration_model.search_peaks_on_ring(1, delta_tth, intensity_min_factor, intensity_max, mask)
        self.widget.peak_num_sb.setValue(3)
        if len(self.model.calibration_model.points):
            self.model.calibration_model.refine()
            self.plot_points()
        else:
            print('Did not find any Points with the specified parameters for the first two rings!')

        progress_dialog.setValue(2)

        refinement_canceled = False
        for i in range(num_rings - 2):
            try:
                points = self.model.calibration_model.search_peaks_on_ring(i + 2, delta_tth, intensity_min_factor,
                                                                           intensity_max, mask)
            except NotEnoughSpacingsInCalibrant:
                QtWidgets.QMessageBox.critical(self.widget,
                                               'Not enough d-spacings!.',
                                               'The calibrant file does not contain enough d-spacings.',
                                               QtWidgets.QMessageBox.Ok)
                break
            self.widget.peak_num_sb.setValue(i + 4)
            if len(self.model.calibration_model.points):
                self.plot_points(points)
                QtWidgets.QApplication.processEvents()
                QtWidgets.QApplication.processEvents()
                self.model.calibration_model.refine()
            else:
                print('Did not find enough points with the specified parameters!')
            progress_dialog.setLabelText("Refining Calibration. \n"
                                         "Finding peaks on Ring {0}.".format(i + 3))
            progress_dialog.setValue(i + 3)
            if progress_dialog.wasCanceled():
                refinement_canceled = True
                break
        progress_dialog.close()
        del progress_dialog

        QtWidgets.QApplication.processEvents()
        if not refinement_canceled:
            self.update_all()

    def load_calibration(self):
        """
        Loads a '*.poni' file and updates the calibration data class
        """
        filename = open_file_dialog(self.widget, caption="Load calibration...",
                                    directory=self.model.working_directories['calibration'],
                                    filter='*.poni')
        if filename is not '':
            self.model.working_directories['calibration'] = os.path.dirname(filename)
            self.model.calibration_model.load(filename)
            if self.model.img_model.filename != '':
                self.update_all()

    def plot_mask(self):
        """
        Plots the mask
        """
        state = self.widget.use_mask_cb.isChecked()
        if state:
            self.widget.img_widget.plot_mask(self.model.mask_model.get_img())
        else:
            self.widget.img_widget.plot_mask(np.zeros(self.model.mask_model.get_img().shape))

    def mask_transparent_status_changed(self, state):
        """
        :param state: Boolean value whether the mask is being transparent
        :type state: bool
        """
        if state:
            self.widget.img_widget.update_pen([255, 0, 0, 100])
        else:
            self.widget.img_widget.update_pen([255, 0, 0, 255])

    def update_all(self, integrate=True):
        """
        Performs 1d and 2d integration based on the current calibration parameter set. Updates the GUI interface
        accordingly with the new diffraction pattern and cake image.
        """
        if integrate:
            progress_dialog = self.create_progress_dialog('Integrating to cake.', '',
                                                          0, show_cancel_btn=False)
            QtWidgets.QApplication.processEvents()
            self.model.current_configuration.integrate_image_1d()
            progress_dialog.setLabelText('Integrating to pattern.')
            QtWidgets.QApplication.processEvents()
            QtWidgets.QApplication.processEvents()
            self.model.current_configuration.integrate_image_2d()
            progress_dialog.close()
        self.widget.cake_widget.plot_image(self.model.cake_data, False)
        self.widget.cake_widget.auto_level()

        self.widget.pattern_widget.plot_data(*self.model.pattern.data)
        self.widget.pattern_widget.plot_vertical_lines(self.convert_x_value(np.array(
            self.model.calibration_model.calibrant.get_2th()) / np.pi * 180, '2th_deg',
                                                                            self.model.current_configuration.integration_unit,
                                                                            None))

        if self.model.current_configuration.integration_unit == '2th_deg':
            self.widget.pattern_widget.pattern_plot.setLabel('bottom', u'2θ', '°')
        elif self.model.current_configuration.integration_unit == 'q_A^-1':
            self.widget.pattern_widget.pattern_plot.setLabel('bottom', 'Q', 'A<sup>-1</sup>')
        elif self.model.current_configuration.integration_unit == 'd_A':
            self.widget.pattern_widget.pattern_plot.setLabel('bottom', 'd', 'A')

        self.widget.pattern_widget.view_box.autoRange()
        if self.widget.tab_widget.currentIndex() == 0:
            self.widget.tab_widget.setCurrentIndex(1)

        if self.widget.ToolBox.currentIndex() is not 2 or \
                self.widget.ToolBox.currentIndex() is not 3:
            self.widget.ToolBox.setCurrentIndex(2)
        self.update_calibration_parameter_in_view()
        self.load_calibrant('pyFAI')

    def update_calibration_parameter_in_view(self):
        """
        Reads the calibration parameter from the calibration_data object and displays them in the GUI.
        :return:
        """
        pyFAI_parameter, fit2d_parameter = self.model.calibration_model.get_calibration_parameter()
        self.widget.set_calibration_parameters(pyFAI_parameter, fit2d_parameter)

        if self.model.calibration_model.distortion_spline_filename:
            self.widget.spline_filename_txt.setText(
                os.path.basename(self.model.calibration_model.distortion_spline_filename))
        else:
            self.widget.spline_filename_txt.setText('None')

    def save_calibration(self):
        """
        Saves the current calibration in a file.
        :return:
        """

        filename = save_file_dialog(self.widget, "Save calibration...",
                                    self.model.working_directories['calibration'], '*.poni')
        if filename is not '':
            self.model.working_directories['calibration'] = os.path.dirname(filename)
            if not filename.rsplit('.', 1)[-1] == 'poni':
                filename = filename + '.poni'
            self.model.calibration_model.save(filename)

