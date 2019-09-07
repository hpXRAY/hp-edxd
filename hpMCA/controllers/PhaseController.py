# -*- coding: utf8 -*-
# Dioptas - GUI program for fast processing of 2D X-ray data
# Copyright (C) 2017  Clemens Prescher (clemens.prescher@gmail.com)
# Institute for Geology and Mineralogy, University of Cologne
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

import numpy as np
from PyQt5 import QtWidgets, QtCore
import copy

from hpMCA.models.PhaseModel import PhaseLoadError
from utilities.HelperModule import get_base_name
from hpMCA.controllers.JcpdsEditorController import JcpdsEditorController
from hpMCA.widgets.UtilityWidgets import save_file_dialog, open_file_dialog, open_files_dialog, CifConversionParametersDialog
from hpMCA.models.PhaseModel import PhaseModel
from hpMCA.widgets.PhaseWidget import PhaseWidget
from hpMCA.widgets.PhaseWidget2 import PhaseWidget as  PhaseWidget2
import utilities.hpMCAutilities as mcaUtil

# imports for type hinting in PyCharm -- DO NOT DELETE
##from ...model.DioptasModel import DioptasModel
#from ...widgets.integration import IntegrationWidget




class PhaseController(object):
    """
    IntegrationPhaseController handles all the interaction between the phase controls in the IntegrationView and the
    PhaseData object. It needs the PatternData object to properly handle the rescaling of the phase intensities in
    the pattern plot and it needs the calibration data to have access to the currently used wavelength.
    """

    def __init__(self, plotWidget, mcaModel, plotController, roiController, directories):
        """
        :param integration_widget: Reference to an IntegrationWidget
        :param dioptas_model: reference to DioptasModel object

        :type integration_widget: IntegrationWidget
        :type dioptas_model: DioptasModel
        """

        self.pattern = mcaModel
        self.mca = mcaModel
        self.directories = directories
        
        
        self.roi_controller = roiController
        self.plotController = plotController
        
        self.pattern_widget = plotWidget
        self.phase_widget = PhaseWidget()
        # testing dioptas 0.5 widgets
        #self.phase_widget2 = PhaseWidget2()
        #self.phase_widget2.show()
        self.wavelength = 0.406626
        self.cif_conversion_dialog = CifConversionParametersDialog()
        self.phase_model = PhaseModel()
        self.jcpds_editor_controller = JcpdsEditorController(self.phase_widget, phase_model=self.phase_model)
        self.phase_lw_items = []
        self.create_signals()
        self.update_temperature_step()
        self.update_pressure_step()
        self.unit = ''
        self.phases = []
        self.tth = self.getTth()
        #self.phase_widget.tth_lbl.setValue(self.tth)

    def set_mca(self, mca):
        self.pattern = mca

    def show_view(self):
        self.active = True

        self.phase_widget.raise_widget()

    def create_signals(self):
        
        self.plotController.dataPlotUpdated.connect(self.update_plot)

        self.connect_click_function(self.phase_widget.add_btn, self.add_btn_click_callback)
        self.connect_click_function(self.phase_widget.delete_btn, self.remove_btn_click_callback)
        self.connect_click_function(self.phase_widget.clear_btn, self.clear_phases)
        self.connect_click_function(self.phase_widget.edit_btn, self.edit_btn_click_callback)
        self.connect_click_function(self.phase_widget.rois_btn, self.rois_btn_click_callback)
        self.connect_click_function(self.phase_widget.save_list_btn, self.save_btn_clicked_callback)
        self.connect_click_function(self.phase_widget.get_tth_btn, self.getTth)
        
        self.connect_click_function(self.phase_widget.load_list_btn, self.load_btn_clicked_callback)

        self.phase_widget.pressure_step_msb.editingFinished.connect(self.update_pressure_step)
        self.phase_widget.temperature_step_msb.editingFinished.connect(self.update_temperature_step)

        self.phase_widget.pressure_sb.valueChanged.connect(self.pressure_sb_changed)
        self.phase_widget.temperature_sb.valueChanged.connect(self.temperature_sb_changed)

        self.phase_widget.show_in_pattern_cb.stateChanged.connect(self.update_phase_legend)

        self.phase_widget.phase_tw.currentCellChanged.connect(self.phase_selection_changed)
        self.phase_widget.color_btn_clicked.connect(self.color_btn_clicked)
        self.phase_widget.show_cb_state_changed.connect(self.show_cb_state_changed)

        self.phase_widget.tth_lbl.valueChanged.connect(self.tth_changed)
        self.phase_widget.tth_step.editingFinished.connect(self.update_tth_step)

        self.phase_widget.file_dragged_in.connect(self.file_dragged_in)
        
        

        self.jcpds_editor_controller.canceled_editor.connect(self.jcpds_editor_reload_phase)

        self.jcpds_editor_controller.lattice_param_changed.connect(self.update_cur_phase_parameters)
        self.jcpds_editor_controller.eos_param_changed.connect(self.update_cur_phase_parameters)

        self.jcpds_editor_controller.reflection_line_added.connect(self.jcpds_editor_reflection_added)
        self.jcpds_editor_controller.reflection_line_removed.connect(self.jcpds_editor_reflection_removed)
        self.jcpds_editor_controller.reflection_line_edited.connect(self.update_cur_phase_parameters)
        self.jcpds_editor_controller.reflection_line_cleared.connect(self.jcpds_editor_reflection_cleared)

        self.jcpds_editor_controller.phase_modified.connect(self.update_phase_legend)

        # Signals from phase model
        self.phase_model.phase_added.connect(self.phase_added)
        self.phase_model.phase_removed.connect(self.phase_removed)
        self.phase_model.phase_changed.connect(self.pattern_widget.update_phase_line_visibilities)
        self.phase_model.phase_changed.connect(self.update_all_phase_intensities)
        self.phase_model.phase_changed.connect(self.update_phase_legend)
        


        


    def connect(self): 
        self.model.phase_model.phase_added.connect(self.add_phase_plot)
        self.model.phase_model.phase_removed.connect(self.pattern_widget.del_phase)

        self.model.phase_model.phase_changed.connect(self.update_phase_lines)
        self.model.phase_model.phase_changed.connect(self.update_phase_legend)
        self.model.phase_model.phase_changed.connect(self.update_phase_color)
        self.model.phase_model.phase_changed.connect(self.update_phase_visible)

        self.model.phase_model.reflection_added.connect(self.reflection_added)
        self.model.phase_model.reflection_deleted.connect(self.reflection_deleted)

        # pattern signals
        self.pattern_widget.view_box.sigRangeChangedManually.connect(self.update_all_phase_lines)
        self.pattern_widget.pattern_plot.autoBtn.clicked.connect(self.update_all_phase_lines)
        self.model.pattern_changed.connect(self.pattern_data_changed)


    def file_dragged_in(self,files):
        
        self.add_btn_click_callback(filenames=files)

    def update_plot(self):
        self.unit = self.plotController.get_unit()

        self.update_all_phase_intensities() 
        self.pattern_widget.update_phase_line_visibilities()

    def JCPDS_roi_btn_clicked(self, index):
        # add rois based on selected JCPDS phase
        phases = self.phase_model.phases
        files = self.phase_model.phase_files
        tth = self.phase_widget.tth_lbl.value()
        d_to_channel = self.mca.get_calibration()[0].d_to_channel
        phase = phases[index]
        filename = files[index]
        name = phase.name
        reflections = phase.get_reflections()
        rois = []
        for reflection in reflections:
            channel = d_to_channel(reflection.d,tth = tth)
            
            lbl = str(name + " " + reflection.get_hkl())
            rois.append({'channel':channel,'halfwidth':10, 'label':lbl, \
                           'name':name, 'hkl':reflection.get_hkl_list()})
        
        self.roi_controller.addJCPDSReflections(rois, phase)

    
    def connect_click_function(self, emitter, function):
        emitter.clicked.connect(function)

    def add_btn_click_callback(self, *args, **kwargs):
        """
        Loads a new phase from jcpds file.
        :return:
        """
        

        filenames = kwargs.get('filenames', None)

        if filenames is None:
            filenames = open_files_dialog(self.phase_widget, "Load Phase(s).",
                                          self.directories.phase )

            
        if len(filenames):
            self.directories.phase = os.path.dirname(str(filenames[0])) #working directory for jcpds files
            
            mcaUtil.save_folder_settings(self.directories)
            progress_dialog = QtWidgets.QProgressDialog("Loading multiple phases.", "Abort Loading", 0, len(filenames),
                                                        self.phase_widget)
            progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
            progress_dialog.setWindowFlags(QtCore.Qt.FramelessWindowHint)
            progress_dialog.show()
            QtWidgets.QApplication.processEvents()
            for ind, filename in enumerate(filenames):
                filename = str(filename)
                progress_dialog.setValue(ind)
                progress_dialog.setLabelText("Loading: " + os.path.basename(filename))
                QtWidgets.QApplication.processEvents()

                self._add_phase(filename)

                if progress_dialog.wasCanceled():
                    break
            progress_dialog.close()
            QtWidgets.QApplication.processEvents()
            self.update_temperature_control_visibility()

    def _add_phase(self, filename):
        #try:
        if filename.endswith("jcpds"):
            self.phase_model.add_jcpds(filename)
        elif filename.endswith(".cif"):
            self.cif_conversion_dialog.exec_()
            
            self.phase_model.add_cif(filename,
                                        self.cif_conversion_dialog.int_cutoff,
                                        self.cif_conversion_dialog.min_d_spacing)
        
            
            
        else:
            
            return
        if self.phase_widget.apply_to_all_cb.isChecked():
            pressure = self.phase_widget.pressure_sb.value()
            temperature = self.phase_widget.temperature_sb.value()
            self.phase_model.phases[-1].compute_d(pressure=pressure,
                                                        temperature=temperature)
        else:
            pressure = 0
            temperature = 298

        self.phase_model.get_lines_d(-1)
        color = self.add_phase_plot()
        self.phase_widget.add_phase(get_base_name(filename), '#%02x%02x%02x' % (int(color[0]), int(color[1]),
                                                                                int(color[2])))

        self.phase_widget.set_phase_pressure(len(self.phase_model.phases) - 1, pressure)
        self.update_phase_legend()
        self.update_temperature(len(self.phase_model.phases) - 1, temperature)
        if self.jcpds_editor_controller.active:
            self.jcpds_editor_controller.show_phase(self.phase_model.phases[-1])
        #except:
            #self.integration_widget.show_error_msg(
            #    'Could not load:\n\n{}.\n\nPlease check if the format of the input file is correct.'. \
            #        format(e.filename))
        #    mcaUtil.displayErrorMessage('phase')

      

    def phase_added(self):
        self.phase_model.get_lines_d(-1)
        color = self.add_phase_plot()
        #print(self.phase_model.phases[-1].params['modified'])
        self.phase_widget.add_phase(self.phase_model.phases[-1].name, '#%02x%02x%02x' %
                                    (int(color[0]), int(color[1]), int(color[2])))

        self.phase_widget.set_phase_pressure(len(self.phase_model.phases) - 1,
                                             self.phase_model.phases[-1].params['pressure'])
        self.update_temperature(len(self.phase_model.phases) - 1,
                                self.phase_model.phases[-1].params['temperature'])
        self.update_phase_legend()
        if self.jcpds_editor_controller.active:
            self.jcpds_editor_controller.show_phase(self.phase_model.phases[-1])

    def add_phase_plot(self):
        """
        Adds a phase to the Pattern view.
        :return:
        """
        axis_range = self.plotController.getRange()
        
        x_range = axis_range[0]
        y_range = axis_range[1]
        self.phases = [positions, intensities, baseline] = \
            self.phase_model.get_rescaled_reflections(
                -1, 'pattern_placeholder_var',
                x_range, y_range,
                self.wavelength,
                self.get_unit(),tth=self.tth)
                
        color = self.pattern_widget.add_phase(self.phase_model.phases[-1].name,
                                              positions,
                                              intensities,
                                              baseline)
        #print(color)
        return color
        

    def edit_btn_click_callback(self):
        cur_ind = self.phase_widget.get_selected_phase_row()
        if cur_ind >= 0:

            self.jcpds_editor_controller.show_phase(self.phase_model.phases[cur_ind])
            self.jcpds_editor_controller.show_view()

    def rois_btn_click_callback(self):
        cur_ind = self.phase_widget.get_selected_phase_row()
        if cur_ind >=0:
            self.JCPDS_roi_btn_clicked(cur_ind)

    def remove_btn_click_callback(self):
        """
        Deletes the currently selected Phase
        """
        cur_ind = self.phase_widget.get_selected_phase_row()
        if cur_ind >= 0:
            self.phase_model.del_phase(cur_ind)

    def phase_removed(self, ind):
        self.phase_widget.del_phase(ind)
        self.pattern_widget.del_phase(ind)
        self.update_temperature_control_visibility()
        if self.jcpds_editor_controller.active:
            ind = self.phase_widget.get_selected_phase_row()
            if ind >= 0:
                self.jcpds_editor_controller.show_phase(self.phase_model.phases[ind])
            else:
                self.jcpds_editor_controller.jcpds_widget.close()

    def load_btn_clicked_callback(self):
        filename = open_file_dialog(self.phase_widget, caption="Load Phase List",
                                    directory='',
                                    filter="*.txt")

        if filename == '':
            return
        #try:

        with open(filename, 'r') as phase_file:
            if phase_file == '':
                return
            for line in phase_file.readlines():
                line = line.replace('\n', '')
                phase, use_flag, color, name, pressure, temperature = line.split(',')
                self.add_btn_click_callback(filenames=phase)
                row = self.phase_widget.phase_tw.rowCount() - 1
                self.phase_widget.phase_show_cbs[row].setChecked(bool(use_flag))
                self.phase_widget.phase_color_btns[row].setStyleSheet('background-color:' + color)
                self.pattern_widget.set_phase_color(row, color)
                self.phase_widget.phase_tw.item(row, 2).setText(name)
                self.phase_widget.set_phase_pressure(row, pressure.replace(' GPa', ''))
                self.phase_model.set_pressure(row, float(pressure.replace(' GPa', '')))
                temperature = temperature.replace(' K', '').replace('-', '')

                if temperature is not '':
                    self.phase_widget.set_phase_temperature(row, temperature)
                    self.phase_model.set_temperature(row, float(temperature))
                    self.update_phase_intensities(row)
                self.update_phase_legend()
        #except:
            
        #    pass


    def save_btn_clicked_callback(self):
        if len(self.phase_model.phase_files) < 1:
            return
        filename = save_file_dialog(self.phase_widget, "Save Phase List.",
                                    '',
                                    'Text (*.txt)')

        if filename == '':
            return

        with open(filename, 'w') as phase_file:
            for file_name, phase_cb, color_btn, row in zip(self.phase_model.phase_files,
                                                           self.phase_widget.phase_show_cbs,
                                                           self.phase_widget.phase_color_btns,
                                                           range(self.phase_widget.phase_tw.rowCount())):
                phase_file.write(file_name + ',' + str(phase_cb.isChecked()) + ',' +
                                 color_btn.styleSheet().replace('background-color:', '').replace(' ', '') + ',' +
                                 self.phase_widget.phase_tw.item(row, 2).text() + ',' +
                                 self.phase_widget.phase_tw.item(row, 3).text() + ',' +
                                 self.phase_widget.phase_tw.item(row, 4).text() + '\n')

    def clear_phases(self):
        """
        Deletes all phases from the GUI and phase data
        """
        while self.phase_widget.phase_tw.rowCount() > 0:
            self.remove_btn_click_callback()
            self.jcpds_editor_controller.close_view()

    def update_pressure_step(self):
        value = self.phase_widget.pressure_step_msb.value()
        self.phase_widget.pressure_sb.setSingleStep(value)

    def update_tth_step(self):
        value = self.phase_widget.tth_step.value()
        self.phase_widget.tth_lbl.setSingleStep(value)    

    def update_temperature_step(self):
        value = self.phase_widget.temperature_step_msb.value()
        self.phase_widget.temperature_sb.setSingleStep(value)

    def pressure_sb_changed(self, val):
        """
        Called when pressure spinbox emits a new value. Calculates the appropriate EOS values and updates line
        positions and intensities.
        """
        if self.phase_widget.apply_to_all_cb.isChecked():
            for ind in range(len(self.phase_model.phases)):
                self.phase_model.set_pressure(ind, np.float(val))
                self.phase_widget.set_phase_pressure(ind, val)
            self.update_all_phase_intensities()

        else:
            cur_ind = self.phase_widget.get_selected_phase_row()
            self.phase_model.set_pressure(cur_ind, np.float(val))
            self.phase_widget.set_phase_pressure(cur_ind, val)
            self.update_phase_intensities(cur_ind)

        self.update_phase_legend()
        self.update_jcpds_editor()

    def temperature_sb_changed(self, val):
        """
        Called when temperature spinbox emits a new value. Calculates the appropriate EOS values and updates line
        positions and intensities.
        """
        if self.phase_widget.apply_to_all_cb.isChecked():
            for ind in range(len(self.phase_model.phases)):
                self.update_temperature(ind, val)
            self.update_all_phase_intensities()

        else:
            cur_ind = self.phase_widget.get_selected_phase_row()
            self.update_temperature(cur_ind, val)
            self.update_phase_intensities(cur_ind)
        self.update_phase_legend()
        self.update_jcpds_editor()

    def update_temperature(self, ind, val):
        if self.phase_model.phases[ind].has_thermal_expansion():
            self.phase_model.set_temperature(ind, np.float(val))
            self.phase_widget.set_phase_temperature(ind, val)
        else:
            self.phase_model.set_temperature(ind, 298)
            self.phase_widget.set_phase_temperature(ind, '-')
        self.update_phase_legend()

    def update_phase_legend(self):
        value = self.phase_widget.show_in_pattern_cb.isChecked()
        self.phase_widget.show_parameter_in_pattern = value
        for ind in range(len(self.phase_model.phases)):
            name = self.phase_model.phases[ind].name
            if self.phase_widget.show_in_pattern_cb.isChecked():
                parameter_str = ''
                pressure = self.phase_model.phases[ind].params['pressure']
                temperature = self.phase_model.phases[ind].params['temperature']
                if pressure != 0:
                    parameter_str += '{:0.2f} GPa '.format(pressure)
                if temperature != 0 and temperature != 298 and temperature is not None:
                    parameter_str += '{:0.2f} K '.format(temperature)
                self.pattern_widget.rename_phase(ind, parameter_str + name)
            else:
                self.pattern_widget.rename_phase(ind, name)
                

    def phase_selection_changed(self, row, col, prev_row, prev_col):
        cur_ind = row
        pressure = self.phase_model.phases[cur_ind].params['pressure']
        temperature = self.phase_model.phases[cur_ind].params['temperature']

        self.phase_widget.pressure_sb.blockSignals(True)
        self.phase_widget.pressure_sb.setValue(pressure)
        self.phase_widget.pressure_sb.blockSignals(False)

        self.phase_widget.temperature_sb.blockSignals(True)
        self.phase_widget.temperature_sb.setValue(temperature)
        self.phase_widget.temperature_sb.blockSignals(False)
        self.update_temperature_control_visibility(row)

        if self.jcpds_editor_controller.active:
            self.jcpds_editor_controller.show_phase(self.phase_model.phases[cur_ind])

    def update_temperature_control_visibility(self, row_ind=None):
        if row_ind is None:
            row_ind = self.phase_widget.get_selected_phase_row()

        if row_ind == -1:
            return

        if self.phase_model.phases[row_ind-1].has_thermal_expansion():
            self.phase_widget.temperature_sb.setEnabled(True)
            self.phase_widget.temperature_step_msb.setEnabled(True)
        else:
            self.phase_widget.temperature_sb.setDisabled(True)
            self.phase_widget.temperature_step_msb.setDisabled(True)

    def color_btn_clicked(self, ind, button):
        previous_color = button.palette().color(1)
        new_color = QtWidgets.QColorDialog.getColor(previous_color, self.phase_widget)
        if new_color.isValid():
            color = str(new_color.name())
        else:
            color = str(previous_color.name())
        self.pattern_widget.set_phase_color(ind, color)
        #print(color)
        button.setStyleSheet('background-color:' + color)

    def show_cb_state_changed(self, ind, state):
        if state:
            self.pattern_widget.show_phase(ind)
        else:
            self.pattern_widget.hide_phase(ind)
        pass
    
        
    def get_unit(self):
        """
        returns the unit currently selected in the GUI
                possible values: 'tth', 'q', 'd', 'e'
        """
        #if self.integration_widget.pattern_tth_btn.isChecked():
        #    return 'tth'
        #elif self.integration_widget.pattern_q_btn.isChecked():
        #    return 'q'
        #elif self.integration_widget.pattern_d_btn.isChecked():
        #    return 'd'
        return self.unit

    def pattern_data_changed(self):
        """
        Function is called after the pattern data has changed.
        """
        # QtWidgets.QApplication.processEvents()
        #self.update_phase_lines_slot()
        self.pattern_widget.update_phase_line_visibilities()


    def tth_changed(self):
        try:
            self.tth = np.clip(float(self.phase_widget.tth_lbl.text()),1,179)
            self.update_all_phase_intensities() 
            self.pattern_widget.update_phase_line_visibilities()
               
            
        except:
            pass
        

    def update_all_phase_intensities(self):
        """
        Updates all intensities of all phases in the pattern view. Also checks if phase lines are still visible.
        (within range of pattern and/or overlays
        """
        axis_range = self.plotController.getRange()
        x_range = axis_range[0]
        y_range = axis_range[1]
        self.phases = []
        for ind in range(len(self.phase_model.phases)):
            self.phases.append(self.update_phase_intensities(ind, axis_range))
        

    def update_phase_intensities(self, ind, axis_range=None):
        """
        Updates the intensities of a specific phase with index ind.
        :param ind: Index of the phase
        :param axis_range: list/tuple of visible x_range and y_range -- ((x_min, x_max), (y_min, y_max))
        """
        if axis_range is None:
            axis_range = self.plotController.getRange()

        x_range = axis_range[0]
        y_range = axis_range[1]
        
        positions, intensities, baseline = self.phase_model.get_rescaled_reflections(
            ind, 'pattern_placeholder_var',
            x_range, y_range,
            self.wavelength,
            self.get_unit(), tth=self.tth
        )
        

        self.pattern_widget.update_phase_intensities(
            ind, positions, intensities, y_range[0])

    def getTth(self):
        tth = self.pattern.get_calibration()[0].two_theta
        self.phase_widget.tth_lbl.setValue(tth)
        
        return self.pattern.get_calibration()[0].two_theta


    ###JCPDS editor callbacks:
    def update_jcpds_editor(self, cur_ind=None):
        if cur_ind is None:
            cur_ind = self.phase_widget.get_selected_phase_row()
        if self.jcpds_editor_controller.jcpds_widget.isVisible():
            self.jcpds_editor_controller.update_phase_view(self.phase_model.phases[cur_ind])

    def jcpds_editor_reload_phase(self, jcpds):
        cur_ind = self.phase_widget.get_selected_phase_row()
        self.phase_model.phases[cur_ind] = jcpds
        self.pattern_widget.phases[cur_ind].clear_lines()
        for dummy_line_ind in self.phase_model.phases[cur_ind].reflections:
            self.pattern_widget.phases[cur_ind].add_line()
           
        self.update_cur_phase_parameters()

    def update_cur_phase_parameters(self):
        cur_ind = self.phase_widget.get_selected_phase_row()
        self.phase_model.get_lines_d(cur_ind)
        self.update_phase_intensities(cur_ind)
        self.update_temperature_control_visibility(cur_ind)
        self.pattern_widget.update_phase_line_visibility(cur_ind)

    def jcpds_editor_reflection_removed(self, reflection_ind):
        cur_phase_ind = self.phase_widget.get_selected_phase_row()
        self.pattern_widget.phases[cur_phase_ind].remove_line(reflection_ind)
        self.phase_model.get_lines_d(cur_phase_ind)
        self.update_phase_intensities(cur_phase_ind)

    def jcpds_editor_reflection_added(self):
        cur_ind = self.phase_widget.get_selected_phase_row()
        self.pattern_widget.phases[cur_ind].add_line()
        self.phase_model.get_lines_d(cur_ind)

    def jcpds_editor_reflection_cleared(self):
        cur_phase_ind = self.phase_widget.get_selected_phase_row()
        self.pattern_widget.phases[cur_phase_ind].clear_lines()