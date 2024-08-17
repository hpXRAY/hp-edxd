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


from PyQt6 import QtCore, QtGui, QtWidgets, Qt

from  PyQt6.QtWidgets import QMainWindow, QApplication, QInputDialog, QWidget, QLabel
from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight
from hpm.widgets.PltWidget import plotWindow
from functools import partial

import os
from .. import style_path, icons_path

class aEDXDWidget(QMainWindow):
    
    def __init__(self,app):
        super().__init__()
        
        self.initUi()
        self.app = app

        raw_fig_parameters = 'Spectrum', 'Counts','Energy (keV)' 
        self.raw_widget = aEDXDRawWidget(raw_fig_parameters)
        self.vertical_layout_tab_raw.addWidget(self.raw_widget)

        spectrum_fig_params = 'Spectrum', 'Intensity (arb.)','Energy (keV)'  
        self.spectrum_widget = aEDXDSpectrumWidget(spectrum_fig_params )
        self.vertical_layout_tab_spectrum.addWidget(self.spectrum_widget)

        primary_beam_fig_params = 'MA-EDXD Primary Beam Analysis', 'Intensity (arb.)','Energy (keV)'
        self.primary_beam_widget = aEDXDPrimaryBeamWidget(primary_beam_fig_params)
        self.vertical_layout_tab_primary_beam.addWidget(self.primary_beam_widget)
        
        all_spectra_fig_params = 'MA-EDXD All Spectra', 'Intensity (arb.)','Energy (keV)'
        self.all_spectra_widget = aEDXDAllSpectraWidget(all_spectra_fig_params)
        self.vertical_layout_tab_all_spectra.addWidget(self.all_spectra_widget)
        
        sq_fig_parameters = 'S(q) Merged and Smoothened', 'S(q)', u'q (\u212B<sup>-1</sup>)'
        self.sq_widget = aEDXDSqWidget(sq_fig_parameters)
        self.vertical_layout_tab_sq.addWidget(self.sq_widget )

        pdf_fig_parameters = 'd(r)', 'd(r)', u'r (\u212B)' # Angstrom symbol in unicode
        self.pdf_widget = aEDXDPDFWidget(pdf_fig_parameters)
        self.vertical_layout_tab_pdf.addWidget(self.pdf_widget)

        inverse_fig_parameters = 'S(q) filtered', 'S(q)', u'q (\u212B<sup>-1</sup>)' 
        self.inverse_widget = aEDXDInverseWidget(inverse_fig_parameters)
        self.vertical_layout_tab_inverse.addWidget(self.inverse_widget)

        '''rdf_fig_parameters = 't(r)', 't(r)', u'r (\u212B)' 
        self.rdf_widget = aEDXDRDFWidget(rdf_fig_parameters)
        self.vertical_layout_tab_rdf.addWidget(self.rdf_widget)'''

   
    def raise_widget(self):
        self.show()
        #self.setWindowState(self.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
        #self.activateWindow()
        #self.raise_()    
    


    def initUi(self):
        
        self.resize(793, 574)
        self.centralwidget = QtWidgets.QWidget(self)

        self._menu_layout = QtWidgets.QVBoxLayout()
        self._menu_layout.setContentsMargins(5, 0, 3, 0)
        self._menu_layout.setSpacing(5)

        self.save_as_btn = FlatButton()
        self.save_as_btn.setEnabled(False)
        self.save_btn = FlatButton()
        self.save_btn.setEnabled(False)
        self.load_btn = FlatButton()
        self.undo_btn = FlatButton()
        self.reset_btn = FlatButton()
        self.export_btn = FlatButton()

        self.angle_btn = FlatButton()
        self.spectra_btn = FlatButton()
        self.atoms_btn = FlatButton()
        self.sq_btn = FlatButton('Sq')
        self.pdf_btn = FlatButton('Gr')
        self.peaks_btn = FlatButton()


        self._menu_layout.addSpacerItem(
            QtWidgets.QSpacerItem(10, 25, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))
        self._menu_layout.addWidget(self.load_btn)
        self._menu_layout.addWidget(self.save_btn)
        self._menu_layout.addWidget(self.save_as_btn)
        self._menu_layout.addSpacerItem(
            QtWidgets.QSpacerItem(10, 30, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))
        self._menu_layout.addWidget(self.undo_btn)
        self._menu_layout.addWidget(self.reset_btn)
        self._menu_layout.addSpacerItem(
            QtWidgets.QSpacerItem(10, 30, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))
        self._menu_layout.addWidget(self.angle_btn)
        self._menu_layout.addWidget(self.spectra_btn)
        self._menu_layout.addWidget(self.atoms_btn)
        self._menu_layout.addWidget(self.sq_btn)
        self._menu_layout.addWidget(self.pdf_btn)
        self._menu_layout.addSpacerItem(
            QtWidgets.QSpacerItem(10, 30, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))
        
        self._menu_layout.addWidget(self.peaks_btn)

        self._menu_layout.addSpacerItem(VerticalSpacerItem())

        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setSpacing(2)
        self.horizontalLayout.setContentsMargins(0,15,5,5)

        self.horizontalLayout.addLayout(self._menu_layout)
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)

        self.tab_1 = QtWidgets.QWidget()
        self.vertical_layout_tab_raw = QtWidgets.QVBoxLayout(self.tab_1)
        self.vertical_layout_tab_raw.setContentsMargins(15,5,15,0)
        self.tabWidget.addTab(self.tab_1, "")

        self.tab = QtWidgets.QWidget()
        self.vertical_layout_tab_spectrum = QtWidgets.QVBoxLayout(self.tab)
        self.vertical_layout_tab_spectrum.setContentsMargins(15,5,15,10)
        self.tabWidget.addTab(self.tab, "")

        self.tab_2 = QtWidgets.QWidget()
          
        self.vertical_layout_tab_all_spectra = QtWidgets.QVBoxLayout(self.tab_2)
        self.vertical_layout_tab_all_spectra.setContentsMargins(15,10,15,0)
        self.tabWidget.addTab(self.tab_2, "")

        self.tab_3 = QtWidgets.QWidget()
       
        self.vertical_layout_tab_primary_beam = QtWidgets.QVBoxLayout(self.tab_3)
        self.vertical_layout_tab_primary_beam.setContentsMargins(15,5,15,0)
        self.tabWidget.addTab(self.tab_3, "")

        
        

        self.tab_4 = QtWidgets.QWidget()
        self.vertical_layout_tab_sq = QtWidgets.QVBoxLayout(self.tab_4)
        self.vertical_layout_tab_sq.setContentsMargins(15,10,15,0)
        self.tabWidget.addTab(self.tab_4, "")

        self.tab_5 = QtWidgets.QWidget()
        self.vertical_layout_tab_pdf = QtWidgets.QVBoxLayout(self.tab_5)
        self.vertical_layout_tab_pdf.setContentsMargins(15,10,15,0)
        self.tabWidget.addTab(self.tab_5, "")

        self.tab_6 = QtWidgets.QWidget()
        self.vertical_layout_tab_inverse = QtWidgets.QVBoxLayout(self.tab_6)
        self.vertical_layout_tab_inverse.setContentsMargins(15,10,15,0)
        self.tabWidget.addTab(self.tab_6, "")

        '''self.tab_7 = QtWidgets.QWidget()
        self.vertical_layout_tab_rdf = QtWidgets.QVBoxLayout(self.tab_7)
        self.vertical_layout_tab_rdf.setContentsMargins(15,10,15,0)
        self.tabWidget.addTab(self.tab_7, "")'''

        self.horizontalLayout.addWidget(self.tabWidget)
        self.retranslateUi()
        self.centralwidget.setLayout(self.horizontalLayout)

        self.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 793, 22))

        self.file_menu = self.menubar.addMenu('&File')
        self.file_op_act = QtWidgets.QAction('&Open project', self)        
        self.file_menu.addAction(self.file_op_act)
        self.file_save_act = QtWidgets.QAction('&Save project', self)  
        self.file_save_act.setEnabled(False)      
        self.file_menu.addAction(self.file_save_act)
        self.file_save_as_act = QtWidgets.QAction('Save project &as...', self)  
        self.file_save_as_act.setEnabled(False)
        self.file_menu.addAction(self.file_save_as_act)
        self.file_save_hdf5_act = QtWidgets.QAction('Save to HDF5', self)        
        #self.file_menu.addAction(self.file_save_hdf5_act)

        self.file_exp_menu = QtWidgets.QMenu('&Export', self)
        self.file_exp_data_act = QtWidgets.QAction('&I(E)', self) 
        self.file_exp_menu.addAction(self.file_exp_data_act)
        self.file_exp_sf_act = QtWidgets.QAction('&S(q)', self) 
        self.file_exp_menu.addAction(self.file_exp_sf_act)

        self.file_exp_sf_plot_act = QtWidgets.QAction('S(q) &plot', self) 
        self.file_exp_menu.addAction(self.file_exp_sf_plot_act)

        self.file_exp_pdf_act = QtWidgets.QAction('&G(r)', self) 
        self.file_exp_menu.addAction(self.file_exp_pdf_act)
        self.file_exp_sf_inv_act = QtWidgets.QAction('Inverse &Fourier-filtered S(q)', self) 
        self.file_exp_menu.addAction(self.file_exp_sf_inv_act)
        self.file_menu.addMenu(self.file_exp_menu)
        
        self.opts_menu = self.menubar.addMenu('Options')
        self.tools_files_act = QtWidgets.QAction(f'2\N{GREEK SMALL LETTER THETA} Files Input', self)    
        self.opts_menu.addAction(self.tools_files_act)

        self.opts_proc_act = QtWidgets.QAction('Spectra', self)        
        self.opts_menu.addAction(self.opts_proc_act)
        self.tools_atoms_act = QtWidgets.QAction('Atoms', self)        
        self.opts_menu.addAction(self.tools_atoms_act)
        self.opts_sq_act = QtWidgets.QAction('Scattering factor', self)        
        self.opts_menu.addAction(self.opts_sq_act)
        self.opts_gr_act = QtWidgets.QAction('PDF', self)        
        self.opts_menu.addAction(self.opts_gr_act)

        self.tools_menu = self.menubar.addMenu('Tools')
        self.tools_peaks_act = QtWidgets.QAction('Peak cutting', self)        
        self.tools_menu.addAction(self.tools_peaks_act)
        
        self.setMenuBar(self.menubar)

        sb = self.statusBar()
        self.progress_bar = QtWidgets.QProgressBar()
        #self.progress_bar.setFixedHeight(15)
        
        #self.pb_widget = QWidget()
        #self._pb_widget_layout = QtWidgets.QHBoxLayout()

        #self._pb_widget_layout.addWidget(QtWidgets.QLabel('Progress: '))
        #self._pb_widget_layout.addWidget(self.progress_bar)
        #self._pb_widget_layout.setContentsMargins(10,0,0,3)
        #self.pb_widget.setLayout(self._pb_widget_layout)
        #sb.addWidget(self.pb_widget)
        
        '''
        self.setStyleSheet("""
            QTabBar::tab {
                background: lightgray;
                color: black;
                border: 0;
                min-width: 90px; 
                max-width: 250px;
                width: 90px; 
                height: 20px;
                padding: 5px;
            }

            QTabBar::tab:selected {
                background: gray;
                color: white;
            }
        """)
        '''

        self.style_widgets()
        self.add_tooltips()
       


    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("aEDXDWidget", "aEDXD"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("aEDXDWidget", "I (E)"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_1), _translate("aEDXDWidget", "EDXD files"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("aEDXDWidget", "All I (E)"))
        
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("aEDXDWidget", "I p,eff (E)"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("aEDXDWidget", "S (q)"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("aEDXDWidget", "PDF"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_6), _translate("aEDXDWidget", "S (q) filtered"))
        #self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_7), _translate("aEDXDWidget", "t(r)"))

    def style_widgets(self):

            
        button_height = 32
        button_width = 32

        icon_size = QtCore.QSize(22, 22)
        self.save_btn.setIcon(QtGui.QIcon(os.path.join(icons_path, 'save.ico')))
        self.save_btn.setIconSize(icon_size)
        self.save_btn.setMinimumHeight(button_height)
        self.save_btn.setMaximumHeight(button_height)
        self.save_btn.setMinimumWidth(button_width)
        self.save_btn.setMaximumWidth(button_width)

        self.save_as_btn.setIcon(QtGui.QIcon(os.path.join(icons_path, 'save_as.ico')))
        self.save_as_btn.setIconSize(icon_size)
        self.save_as_btn.setMinimumHeight(button_height)
        self.save_as_btn.setMaximumHeight(button_height)
        self.save_as_btn.setMinimumWidth(button_width)
        self.save_as_btn.setMaximumWidth(button_width)

        self.load_btn.setIcon(QtGui.QIcon(os.path.join(icons_path, 'open.ico')))
        self.load_btn.setIconSize(icon_size)
        self.load_btn.setMinimumHeight(button_height)
        self.load_btn.setMaximumHeight(button_height)
        self.load_btn.setMinimumWidth(button_width)
        self.load_btn.setMaximumWidth(button_width)

        self.undo_btn.setIcon(QtGui.QIcon(os.path.join(icons_path, 'undo.ico')))
        self.undo_btn.setIconSize(icon_size)
        self.undo_btn.setMinimumHeight(button_height)
        self.undo_btn.setMaximumHeight(button_height)
        self.undo_btn.setMinimumWidth(button_width)
        self.undo_btn.setMaximumWidth(button_width)

        self.reset_btn.setIcon(QtGui.QIcon(os.path.join(icons_path, 'restore.ico')))
        self.reset_btn.setIconSize(icon_size)
        self.reset_btn.setMinimumHeight(button_height)
        self.reset_btn.setMaximumHeight(button_height)
        self.reset_btn.setMinimumWidth(button_width)
        self.reset_btn.setMaximumWidth(button_width)

        self.export_btn.setIcon(QtGui.QIcon(os.path.join(icons_path, 'export.ico')))
        self.export_btn.setIconSize(icon_size)
        self.export_btn.setMinimumHeight(button_height)
        self.export_btn.setMaximumHeight(button_height)
        self.export_btn.setMinimumWidth(button_width)
        self.export_btn.setMaximumWidth(button_width)

        self.angle_btn.setIcon(QtGui.QIcon(os.path.join(icons_path, 'import.ico')))
        self.angle_btn.setIconSize(icon_size)
        self.angle_btn.setMinimumHeight(button_height)
        self.angle_btn.setMaximumHeight(button_height)
        self.angle_btn.setMinimumWidth(button_width)
        self.angle_btn.setMaximumWidth(button_width)

        self.spectra_btn.setIcon(QtGui.QIcon(os.path.join(icons_path, 'spectra.ico')))
        self.spectra_btn.setIconSize(icon_size)
        self.spectra_btn.setMinimumHeight(button_height)
        self.spectra_btn.setMaximumHeight(button_height)
        self.spectra_btn.setMinimumWidth(button_width)
        self.spectra_btn.setMaximumWidth(button_width)

        self.atoms_btn.setIcon(QtGui.QIcon(os.path.join(icons_path, 'atoms1.ico')))
        self.atoms_btn.setIconSize(icon_size)
        self.atoms_btn.setMinimumHeight(button_height)
        self.atoms_btn.setMaximumHeight(button_height)
        self.atoms_btn.setMinimumWidth(button_width)
        self.atoms_btn.setMaximumWidth(button_width)

        #self.sq_btn.setIcon(QtGui.QIcon(os.path.join(icons_path, 'sq1.ico')))
        self.sq_btn.setIconSize(icon_size)
        self.sq_btn.setMinimumHeight(button_height)
        self.sq_btn.setMaximumHeight(button_height)
        self.sq_btn.setMinimumWidth(button_width)
        self.sq_btn.setMaximumWidth(button_width)

        #self.pdf_btn.setIcon(QtGui.QIcon(os.path.join(icons_path, 'pdf1.ico')))
        self.pdf_btn.setIconSize(icon_size)
        self.pdf_btn.setMinimumHeight(button_height)
        self.pdf_btn.setMaximumHeight(button_height)
        self.pdf_btn.setMinimumWidth(button_width)
        self.pdf_btn.setMaximumWidth(button_width)
    
        self.peaks_btn.setIcon(QtGui.QIcon(os.path.join(icons_path, 'cut.ico')))
        self.peaks_btn.setIconSize(icon_size)
        self.peaks_btn.setMinimumHeight(button_height)
        self.peaks_btn.setMaximumHeight(button_height)
        self.peaks_btn.setMinimumWidth(button_width)
        self.peaks_btn.setMaximumWidth(button_width)

        

    def add_tooltips(self):
        self.load_btn.setToolTip('Open Project')
        self.save_btn.setToolTip('Save Project')
        self.save_as_btn.setToolTip('Save Project As...')
        self.undo_btn.setToolTip('Undo')
        self.reset_btn.setToolTip('Reset to last saved configuration')
        self.angle_btn.setToolTip('EDXD Files Input')
        self.spectra_btn.setToolTip('Spectra options')
        self.atoms_btn.setToolTip('Atoms options')
        self.sq_btn.setToolTip('Scattering Factor options')
        self.pdf_btn.setToolTip('PDF options')
        self.peaks_btn.setToolTip('Peak cutting')


class customWidget(QtWidgets.QWidget):
    widget_closed = QtCore.pyqtSignal()
    def __init__(self, fig_params):
        super().__init__()
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0,0,0,0)
        
        self.top_button_widget = QtWidgets.QWidget()
        self._top_button_widget_layout = QtWidgets.QHBoxLayout()
        self._top_button_widget_layout.setContentsMargins(0,0,0,0)
        self.top_button_widget.setLayout(self._top_button_widget_layout)

        self._layout.addWidget(self.top_button_widget)

        self.fig = plotWindow(*fig_params)
        self._layout.addWidget(self.fig)
        self.cursor_fast = QLabel()
        self.cursor_fast.setFixedWidth(70)
        self.cursor = QLabel()
        self.cursor.setFixedWidth(70)
        self.cursor_widget = QWidget()
        self._cursor_widget_layout = QtWidgets.QHBoxLayout()
        self._cursor_widget_layout.setContentsMargins(0,0,0,0)
        self._cursor_widget_layout.addSpacerItem(HorizontalSpacerItem())
        self._cursor_widget_layout.addWidget(self.cursor_fast)
        self._cursor_widget_layout.addWidget(self.cursor)
        self._cursor_widget_layout.addSpacerItem(HorizontalSpacerItem())
        self.cursor_widget.setLayout(self._cursor_widget_layout)
        self._layout.addWidget(self.cursor_widget)
        self.button_widget = QtWidgets.QWidget()
        self._button_widget_layout = QtWidgets.QHBoxLayout()
        self._button_widget_layout.setContentsMargins(0,0,0,0)
        
        self.button_widget.setLayout(self._button_widget_layout)
        self._layout.addWidget(self.button_widget)
        self.setLayout(self._layout)
        self.create_connections()

    def add_button_widget_item(self, item):
        self._button_widget_layout.addWidget(item)
    
    def add_button_widget_spacer(self):
        self._button_widget_layout.addSpacerItem(HorizontalSpacerItem())

    def add_top_button_widget_item(self, item):
        self._top_button_widget_layout.addWidget(item)
    
    def add_top_button_widget_spacer(self):
        self._top_button_widget_layout.addSpacerItem(HorizontalSpacerItem())

    def create_connections(self):
        self.fig.fast_cursor.connect(self.update_fast_cursor)
        self.fig.cursor.connect(self.update_cursor)

    def raise_widget(self):
        self.show()

    def closeEvent(self, event):
        # Overrides close event to let controller know that widget was closed by user
        self.widget_closed.emit()

    def update_fast_cursor(self, pos):
        c = '%.3f' % (pos)
        self.cursor_fast.setText(c)
        self.fig.set_fast_cursor(pos)

    def update_cursor(self, pos):
        c = "<span style='color: #00CC00'>%0.3f</span>"  % (pos)
        self.cursor.setText(c)
        self.fig.set_cursor(pos)

    def setText(self, text, plot_ind):
        self.fig.set_plot_label(text,plot_ind)

    def raise_widget(self):
        self.show()
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
        self.activateWindow()
        

class aEDXDSpectrumWidget(customWidget):
    def __init__(self, fig_params):
        super().__init__(fig_params)
        
        button_height = 32
        button_width = 32

        icon_size = QtCore.QSize(22, 22)

        self.cut_peak_btn = FlatButton()
        self.cut_peak_btn.setCheckable(True)
        self.apply_btn = FlatButton("Apply")
        self.apply_btn.setMinimumWidth(90)

        
        self.cut_peak_btn.setIcon(QtGui.QIcon(os.path.join(icons_path, 'remove.ico')))
        self.cut_peak_btn.setIconSize(icon_size)
        self.cut_peak_btn.setMinimumHeight(button_height)
        self.cut_peak_btn.setMaximumHeight(button_height)
        self.cut_peak_btn.setMinimumWidth(button_width)
        self.cut_peak_btn.setMaximumWidth(button_width)

        

        self.add_button_widget_item(QtWidgets.QLabel('Remove peak'))
        self.add_button_widget_item(self.cut_peak_btn)
        
        self.add_button_widget_spacer()
        self.add_button_widget_item(self.apply_btn)
        
    



class aEDXDPrimaryBeamWidget(customWidget):
    def __init__(self, fig_params):
        super().__init__(fig_params)
        

class aEDXDSqWidget(customWidget):
    def __init__(self, fig_params):
        super().__init__(fig_params)
        self.export_Sq_btn = FlatButton()
        self.export_Sq_btn.setToolTip('Export S(q)')
        self.export_Sq_plot_btn = FlatButton()
        self.export_Sq_plot_btn.setToolTip('Export S(q)')

        button_height = 32
        button_width = 32

        icon_size = QtCore.QSize(22, 22)
        self.export_Sq_btn.setIcon(QtGui.QIcon(os.path.join(icons_path, 'export.ico')))
        self.export_Sq_btn.setIconSize(icon_size)
        self.export_Sq_btn.setMinimumHeight(button_height)
        self.export_Sq_btn.setMaximumHeight(button_height)
        self.export_Sq_btn.setMinimumWidth(button_width)
        self.export_Sq_btn.setMaximumWidth(button_width)

        self.add_top_button_widget_item(self.export_Sq_btn)

        
        '''self.export_Sq_plot_btn.setIcon(QtGui.QIcon(os.path.join(icons_path, 'export.ico')))
        self.export_Sq_plot_btn.setIconSize(icon_size)
        self.export_Sq_plot_btn.setMinimumHeight(button_height)
        self.export_Sq_plot_btn.setMaximumHeight(button_height)
        self.export_Sq_plot_btn.setMinimumWidth(button_width)
        self.export_Sq_plot_btn.setMaximumWidth(button_width) 

        self.add_top_button_widget_item(self.export_Sq_plot_btn)''' 

        self.sq_display_options_widget = QtWidgets.QWidget()
        self._sq_display_options_widget_layout = QtWidgets.QHBoxLayout()
        self.show_data_cb = QtWidgets.QCheckBox("Data points")
        self.show_data_cb.setChecked(True)
        self.show_Sq_cb = QtWidgets.QCheckBox("S(q)")
        self.show_Sq_cb.setChecked(True)
        self.show_Sq_uncertainty_cb = QtWidgets.QCheckBox("S(q) st.dev.")
        self.show_Sq_uncertainty_cb.setChecked(True)
      
      
        self._sq_display_options_widget_layout.addWidget(self.show_data_cb)
        self._sq_display_options_widget_layout.addWidget(self.show_Sq_cb)
        self._sq_display_options_widget_layout.addWidget(self.show_Sq_uncertainty_cb)
    
        self.sq_display_options_widget.setLayout(self._sq_display_options_widget_layout)

        self.add_top_button_widget_item(self.sq_display_options_widget)

        self.add_top_button_widget_spacer()
        

class aEDXDPDFWidget(customWidget):
    def __init__(self, fig_params):
        super().__init__(fig_params)
        self.export_PDF_btn = FlatButton()
        self.export_PDF_btn.setToolTip('Export d(r)')

        button_height = 32
        button_width = 32

        icon_size = QtCore.QSize(22, 22)
        self.export_PDF_btn.setIcon(QtGui.QIcon(os.path.join(icons_path, 'export.ico')))
        self.export_PDF_btn.setIconSize(icon_size)
        self.export_PDF_btn.setMinimumHeight(button_height)
        self.export_PDF_btn.setMaximumHeight(button_height)
        self.export_PDF_btn.setMinimumWidth(button_width)
        self.export_PDF_btn.setMaximumWidth(button_width)
        self.add_top_button_widget_item(self.export_PDF_btn)

        self.pdf_type_widget = QtWidgets.QWidget()
        self._pdf_type_widget_layout = QtWidgets.QHBoxLayout()
        self.dr_cb = QtWidgets.QRadioButton("d(r)")
        self.dr_cb.setChecked(True)
        self.tr_cb = QtWidgets.QRadioButton("t(r)")
        self.gr_cb = QtWidgets.QRadioButton("g(r)")
        self.Gr_cb = QtWidgets.QRadioButton("G(r)")
        self._pdf_type_widget_layout.addWidget(self.dr_cb)
        self._pdf_type_widget_layout.addWidget(self.tr_cb)
        self._pdf_type_widget_layout.addWidget(self.gr_cb)
        self._pdf_type_widget_layout.addWidget(self.Gr_cb)
        self.pdf_type_widget.setLayout(self._pdf_type_widget_layout)

        self.add_top_button_widget_item(self.pdf_type_widget)


        self.add_top_button_widget_spacer()

class aEDXDRDFWidget(customWidget):
    def __init__(self, fig_params):
        super().__init__(fig_params)
        self.export_PDF_btn = FlatButton()
        self.export_PDF_btn.setToolTip('Export t(r)')

        button_height = 32
        button_width = 32

        icon_size = QtCore.QSize(22, 22)
        self.export_PDF_btn.setIcon(QtGui.QIcon(os.path.join(icons_path, 'export.ico')))
        self.export_PDF_btn.setIconSize(icon_size)
        self.export_PDF_btn.setMinimumHeight(button_height)
        self.export_PDF_btn.setMaximumHeight(button_height)
        self.export_PDF_btn.setMinimumWidth(button_width)
        self.export_PDF_btn.setMaximumWidth(button_width)
        self.add_top_button_widget_item(self.export_PDF_btn)
        self.add_top_button_widget_spacer()
        

class aEDXDAllSpectraWidget(customWidget):
    def __init__(self, fig_params):
        super().__init__(fig_params)
        self.export_IE_btn = FlatButton()
        self.export_IE_btn.setToolTip('Export All I(E)')

        button_height = 32
        button_width = 32

        icon_size = QtCore.QSize(22, 22)
        self.export_IE_btn.setIcon(QtGui.QIcon(os.path.join(icons_path, 'export.ico')))
        self.export_IE_btn.setIconSize(icon_size)
        self.export_IE_btn.setMinimumHeight(button_height)
        self.export_IE_btn.setMaximumHeight(button_height)
        self.export_IE_btn.setMinimumWidth(button_width)
        self.export_IE_btn.setMaximumWidth(button_width)
        self.add_top_button_widget_item(self.export_IE_btn)
        self.add_top_button_widget_spacer()
        

class aEDXDInverseWidget(customWidget):
    def __init__(self, fig_params):
        super().__init__(fig_params)
        self.export_iSq_btn = FlatButton()
        self.export_iSq_btn.setToolTip('Export Inverse Fourier-filtered S(q)')
        button_height = 32
        button_width = 32

        icon_size = QtCore.QSize(22, 22)
        self.export_iSq_btn.setIcon(QtGui.QIcon(os.path.join(icons_path, 'export.ico')))
        self.export_iSq_btn.setIconSize(icon_size)
        self.export_iSq_btn.setMinimumHeight(button_height)
        self.export_iSq_btn.setMaximumHeight(button_height)
        self.export_iSq_btn.setMinimumWidth(button_width)
        self.export_iSq_btn.setMaximumWidth(button_width)
        self.add_top_button_widget_item(self.export_iSq_btn)
        self.add_top_button_widget_spacer()
        

class aEDXDRawWidget(customWidget):
    def __init__(self, fig_params):
        super().__init__(fig_params)
        