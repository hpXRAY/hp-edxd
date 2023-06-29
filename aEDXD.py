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

"""aEDXD.py:
   A GUI program for amorphous EDXD data analysis
"""

###############################################################################
#   v1.1.1:
#       1. Multi-file input at a 2Th angle is allowed
#   v1.1.2:
#       1. The spectrum normalization is done directly to be S(q),
#          not to be an I(q) profile, hence no I(q) data is saved.
#       2. Change the way to smooth the S(q) for UnivariateSpline parameter.
#          The sq_smoothing_factor is multiplied to the weight factor:
#          scipy.interploate.UnivariateSpline(x,y,w=sq_smoothing_factor*weight,
#          bbox=[None,None],k=3,s=None), where weight = 1/y_err.
#   v1.2a:
#       1. The version convention changed: Version.Revision+a:alphaa or b:beta    
#       2. Read aEDXD_input.cfg file: python dictionary syntax
#       3. aEDXD_class is separated from aEDXD_functions
#       4. Verbose analysis using aEDXD methods
#   v1.2b:
#       1. Automatically save the structure factor segments
#   v1.3:
#       1. Work with:
#               Python 2.7.9
#               Numpy 1.9.2
#               Scipy 0.15.1
#               matplotlib 1.4.3
#       2. Add new option to choose input file format
#       3. Fix a but that did not sum intensities from multiple input files
# 
#   0.5.0:  Author: R. Hrubiak, ANL
#       1. Upgraded to Python 3.7
#       2. Created a GUI based on PyQt5 and pyqtgraph
#       3. Visual peak removal via interactive plot
#       4. Improved interpolation under removed peaks 
#       4. GUI options editing / operation possible without a config file
#       6. Options/project saving  
#   0.6.0:  Author: R. Hrubiak, ANL
#       Renamed as CAESPES: Combined Amorphous EDXD Spectra Processing Evaluation Software
#       1. Plotting of calculated uncertainties 
#       2. Invalid config file loading does not crash program
#       3. Drag-drop files
#       4. Automatic loading of 2theta from file.
#       5. Save and save-as options
#       6. Buttons to export data
#       7. In peak cut window only show one 2th at a time
#       8. Undo, reset functionality
#       9. Input/output directory saving to config file
#       10. Warn to save on exit
#       11. Save file-use tag
#       12. Periodic table atom picker
#       13. Scan sequence files automating sorting and loading 
#       14. Peak cutting based on parametrized baseline extraction, spline still available for legacy projects
#       15. Expanded number of available elements [H.H.M. Balyuzi, Acta Cryst. (175). A31, 600]
#       16. Fixed I_incoherent normalization bug
#    0.6.5: Author: R. Hrubiak, ANL (06/28/2023)
#        1. Sq Spline algorithm re-done using weighted binning, predictable Sq smoothing 
#        2. TODO Mask data. Remove bad data at W edge. Optimal parameters not found when ~68 < Emax < ~72, prorgram crashes
#        3. TODO integrate peak editing plot view with main view
#        4. TODO improve autonaming of exported files, export all option
#        5. TODO improve parameter entry, e.g. density should be more intuitive or offer more options
#        6. TODO self consistency of S(q) check
#        7. TODO Integrate scanning data acquisition and data display
#        8. TODO Q scale for n spectrum view
#        9. TODO keep row selection consist when adding and deleting items from tableviews
#       10. TODO multiple iterations of S(q) filtering
#       12. TODO Hdf5 compatible, save datafiles in projext files, save parameters in exported files?
#       13. TODO RDF plot, with peak measurements
#       14. TODO reorganize abc mkl ab5 files in sane manner in jsons
#       15. TODO Allow different E range for different segments 
#        
###############################################################################



from axd import main

main()

