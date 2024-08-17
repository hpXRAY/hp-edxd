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
from signal import signal
from PyQt6 import QtWidgets, QtCore

import numpy as np

from ..widgets.UtilityWidgets import open_file_dialog, save_file_dialog
from .. import calibrants_path
from utilities.HelperModule import get_partial_index

from ..widgets.GSDCalibrationWidget import GSDCalibrationWidget
from ..widgets.UtilityWidgets import open_file_dialog
from ..models.GSDCalibrationModel import GSD2thetaCalibrationModel, NotEnoughSpacingsInCalibrant


class GSDCalibrationController(QtCore.QObject):
    """
    CalibrationController handles all the interaction between the CalibrationView and the CalibrationData class
    """
    scale_correction_signal = QtCore.pyqtSignal(list)

    def __init__(self):
        """Manages the connection between the calibration GUI and data

        :param widget: Gives the Calibration Widget
        :type widget: CalibrationWidget

        :param dioptas_model: Reference to DioptasModel Object
        :type dioptas_model: DioptasModel

        """
        super().__init__()
        self.widget = GSDCalibrationWidget()
        self.model = GSD2thetaCalibrationModel()

        self.widget.set_start_values(self.model.start_values)
        self._first_plot = True
        self.create_signals()
   
        self.load_calibrants_list()

    def create_signals(self):
        """
        Connects the GUI signals to the appropriate Controller methods.
        """
        
        self.create_transformation_signals()
    
        self.widget.calibrant_cb.currentIndexChanged.connect(self.load_calibrant)


        self.widget.save_calibration_btn.clicked.connect(self.save_calibration)
        self.widget.load_calibration_btn.clicked.connect(self.load_calibration)
        #self.widget.calibrate_btn.clicked.connect(self.calibrate)
        #self.widget.refine_btn.clicked.connect(self.refine)

        self.widget.clear_peaks_btn.clicked.connect(self.clear_peaks_btn_click)
        self.widget.undo_peaks_btn.clicked.connect(self.undo_peaks_btn_clicked)

        self.widget.use_mask_cb.stateChanged.connect(self.plot_mask)
        self.widget.mask_transparent_cb.stateChanged.connect(self.mask_transparent_status_changed)

        self.widget.cal_gsd_add_pt_btn.clicked.connect(self.cal_gsd_add_pt_btn_callback)
        #self.widget.cal_gsd_calc_btn.clicked.connect(self.cal_gsd_calc_btn_callback)
        self.widget.calibrate_btn.clicked.connect(self.calibrate_btn_callback)
        self.widget.refine_btn.clicked.connect(self.refine_btn_callback)
        self.widget.refine_e_btn.clicked.connect(self.refine_energy)

        self.widget.plotMouseCursorSignal.connect(self.cursor_moved_callback)

        xrf_point = self.model.fixed_xrf_points[0]
        self.widget.plot_flat.win.set_cursorFast_pos(xrf_point)
        self.widget.xrf_line.setPos(xrf_point)

    def set_2D_data(self, E_scale, data):
        self.model.set_data(data)
        self.model.set_e_scale(E_scale)
        self.widget.set_image_scale('E',E_scale)
        display_data = self.model.data_raw
        self.widget.set_spectral_data(display_data)

        
        x = self.model.E
        flat_E = self.model.flat_E
        self.widget.plot_flat.win.plotData(x, flat_E)


        
    def cursor_moved_callback(self, cursor):
        pick_peaks = self.widget.calibration_control_widget.calibration_parameters_widget.peak_selection_gb.pick_peaks_cb.isChecked()
        if pick_peaks:
            self.search_peaks(cursor)
        else:
            x, y = cursor[0] ,cursor[1] 
            self.widget.plot_flat.set_cursor(y)
            self.widget.set_cursor_pos(x, y)

    def search_peaks(self, cursor):
        """
        Searches peaks around a specific points (x,y) in the current image file. The algorithm for searching
        (either automatic or single peaksearch) is set in the GUI.
        :param x:
            x-Position for the search.
        :param y:
            y-Position for the search
        """
        
        x, y = cursor[0] ,cursor[1] 
        x, y = y, x  # indeces for the img array are transposed compared to the mouse position

        x = self .model.convert_point_E_to_channel(x) // self.model.bin

        # convert pixel coord into pixel index
        x, y = int(x), int(y)

        # filter events outside the image
        shape = self.model.data.shape
        x_len = shape[1]
        y_len = shape[0]

        if not (0 <= x < x_len):
            return
        if not (0 <= y < y_len):
            return

        peak_ind = self.widget.peak_num_sb.value()
        if self.widget.automatic_peak_search_rb.isChecked():
            points = self.model.find_peaks_automatic(y, x, peak_ind - 1)
        else:
            search_size = int(self.widget.search_size_sb.value())
            points = self.model.find_peak(y, x, search_size, peak_ind - 1)
        if len(points):
            
            self.plot_points()
            
            if self.widget.automatic_peak_num_inc_cb.checkState():
                self.widget.peak_num_sb.setValue(peak_ind + 1)

    def calibrate_btn_callback(self):
        # calibrate based on user picked points
        self.model.do_2theta_calibration()  
        tth = self.model.tth_calibrated 
        segments_x, segments_y = self.model.get_simulated_lines(tth)
        self.widget.plot_lines(segments_x, segments_y)

    def refine_btn_callback(self):
        # calibrate based on automatically found points after the initial calibration
        self.model.refine_2theta_calibration()  
        tth = self.model.tth_calibrated 
        segments_x, segments_y = self.model.get_simulated_lines(tth)
        self.widget.plot_lines(segments_x, segments_y)

    def refine_energy(self):
        # TODO implement refining energy calibration based on XRD peaks
        self.model.refine_e_simple()
        
        

        #self.model. set_e_scale(e_corrected)

        x = self.model.E
        flat_E = self.model.flat_E
        self.widget.plot_flat.win.plotData(x, flat_E)

        self.widget.set_image_scale('E',self.model.E_scale)
        self.widget.set_spectral_data(self.model.data_raw)
        self.widget.lines.clear()
        self.clear_peaks_btn_click()

        '''#self.widget.set_spectral_data(self.model.data_raw)
        tth = self.model.tth_calibrated
        segments_x, segments_y = self.model.get_simulated_lines(tth)
        self.widget.plot_lines(segments_x, segments_y)'''
        

        self.scale_correction_signal.emit(self.model.E_scale)

    def cal_gsd_add_pt_btn_callback(self): 
   
        cursor_pt = self.widget.cursorPoints[0]
        x = cursor_pt[0]
        y = cursor_pt[1]
        x_range, y_range = self.model.add_point(x,y)
        
        x_range = (x_range[::self.model.bin] )
        y_range= y_range[::self.model.bin]+0.5
        self.widget.p_scatter.setData(x_range,y_range)
        

    def create_transformation_signals(self):
        """
        Connects all the rotation GUI controls.
        """
        self.widget.invert_vertical_btn.clicked.connect(self.invert_vertical_btn_clicked)



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
        self.widget.calibrant_cb.setCurrentIndex(self._calibrants_file_names_list.index('Au'))  # to Au
        self.load_calibrant()

    def load_calibrant(self):
        """
        Loads the selected calibrant in the calibrant combobox into the calibration data.
        :param two_theta: determines which two_theta to use possible values: "start_values"
        """
        current_index = self.widget.calibrant_cb.currentIndex()
        filename = os.path.join(calibrants_path,
                                self._calibrants_file_list[current_index])
        self.model.set_calibrant(filename)

        
        start_values = self.widget.get_start_values()
        two_theta = start_values['two_theta']
        self.model.calibrant.set2thetachangeE(two_theta)
    
        calibrant_line_positions = np.array(self.model.calibrant.get_E()) 
        
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


    def plot_points(self, points=None):
        """
        Plots points into the image view.
        :param points:
            list of points, whereby a point is a [x,y] element. If it is none it will plot the points stored in the
            calibration_data
        """
        if points is None:
            try:
                points = self.model.get_point_array()
            except IndexError:
                points = []
        if len(points):
            y_data, x_data, ind = zip(*points)
            y_data = np.array(y_data)+0.5
            x_data = np.array(x_data)* self.model.bin + 0.5
            x_data = self.model.convert_point_channel_to_E(x_data)

            self.widget.p_scatter.setData(x_data, y_data)
            

      

    def clear_peaks_btn_click(self):
        """
        Deletes all points/peaks in the calibration_data and in the gui.
        :return:
        """
        self.model.clear_peaks()
        self.widget.p_scatter.setData([],[])
        self.widget.peak_num_sb.setValue(1)

    def undo_peaks_btn_clicked(self):
        """
        undoes clicked peaks
        """
        num_points = self.model.remove_last_peak()
        self.widget.remove_last_scatter_points(num_points)
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
        delta_tth = float(self.widget.options_delta_tth_txt.text())
        intensity_min_factor = float(self.widget.options_intensity_mean_factor_sb.value())
        intensity_max = float(self.widget.options_intensity_limit_txt.text())

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
        if filename != '':
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

        if self.widget.ToolBox.currentIndex() != 2 or \
                self.widget.ToolBox.currentIndex() !=3:
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
        if filename != '':
            self.model.working_directories['calibration'] = os.path.dirname(filename)
            if not filename.rsplit('.', 1)[-1] == 'poni':
                filename = filename + '.poni'
            self.model.save(filename)

    def convert_x_value(self, value, previous_unit, new_unit, wavelength):
        if wavelength is None:
            wavelength = self.model.wavelength
        if previous_unit == '2th_deg':
            tth = value
        elif previous_unit == 'q_A^-1':
            tth = np.arcsin(
                value * 1e10 * wavelength / (4 * np.pi)) * 360 / np.pi
        elif previous_unit == 'd_A':
            tth = 2 * np.arcsin(wavelength / (2 * value * 1e-10)) * 180 / np.pi
        else:
            tth = 0

        if new_unit == '2th_deg':
            res = tth
        elif new_unit == 'q_A^-1':
            res = 4 * np.pi * \
                  np.sin(tth / 360 * np.pi) / \
                  wavelength / 1e10
        elif new_unit == 'd_A':
            res = wavelength / (2 * np.sin(tth / 360 * np.pi)) * 1e10
        else:
            res = 0
        return res

    def invert_vertical_btn_clicked(self):
        self.model.flip_img_vertically()
        self.clear_peaks_btn_click()