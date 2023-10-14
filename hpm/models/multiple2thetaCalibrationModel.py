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


from fileinput import filelineno

from operator import truediv
import numpy as np
from scipy import interpolate
from PyQt5 import QtCore, QtWidgets

import os, sys
import time
import copy

from .mcareaderGeStrip import *
from .mcaModel import McaCalibration, McaElapsed, McaROI, McaEnvironment, MCA
from utilities.CARSMath import fit_gaussian
from .eCalModel import calc_parabola_vertex, fit_energies
from .mcaComponents import McaROI
import hpm.models.Xrf as Xrf

from hpm.models.MaskModel import MaskModel
from pyFAI.massif import Massif
from pyFAI.blob_detection import BlobDetection
from pyFAI.calibrant import Calibrant


from .. import calibrants_path

class Multiple2ThetaModel(QtCore.QObject):  # 
    def __init__(self,  *args, **filekw):
        
        """
        Creates new Multiple Spectra object.  
    
        Example:
            m = MultipleSpectraModel()
        """
        self.mca : MCA
        self.mca = None
   
        
    def find_peaks_automatic(self, x, y, peak_ind):
        """
        Searches peaks by using the Massif algorithm
        :param float x:
            x-coordinate in pixel - should be from original image (not supersampled x-coordinate)
        :param float y:
            y-coordinate in pixel - should be from original image (not supersampled y-coordinate)
        :param peak_ind:
            peak/ring index to which the found points will be added
        :return:
            array of points found
        """
        massif = Massif(self.img_model._img_data)
        cur_peak_points = massif.find_peaks((int(np.round(x)), int(np.round(y))), stdout=DummyStdOut())
        if len(cur_peak_points):
            self.points.append(np.array(cur_peak_points))
            self.points_index.append(peak_ind)
        return np.array(cur_peak_points)

    def find_peak(self, x, y, search_size, peak_ind):
        """
        Searches a peak around the x,y position. It just searches for the maximum value in a specific search size.
        :param int x:
            x-coordinate in pixel - should be from original image (not supersampled x-coordinate)
        :param int y:
            y-coordinate in pixel - should be form original image (not supersampled y-coordinate)
        :param search_size:
            the length of the search rectangle in pixels in all direction in which the algorithm searches for
            the maximum peak
        :param peak_ind:
            peak/ring index to which the found points will be added
        :return:
            point found (as array)
        """
        left_ind = int(np.round(x - search_size * 0.5))
        if left_ind < 0:
            left_ind = 0
        top_ind = int(np.round(y - search_size * 0.5))
        if top_ind < 0:
            top_ind = 0
        search_array = self.img_model.img_data[left_ind:(left_ind + search_size), top_ind:(top_ind + search_size)]
        x_ind, y_ind = np.where(search_array == search_array.max())
        x_ind = x_ind[0] + left_ind
        y_ind = y_ind[0] + top_ind
        self.points.append(np.array([x_ind, y_ind]))
        self.points_index.append(peak_ind)
        return np.array([np.array((x_ind, y_ind))])

    def clear_peaks(self):
        self.points = []
        self.points_index = []

    def remove_last_peak(self):
        if self.points:
            num_points = int(self.points[-1].size/2)  # each peak is x, y so length is twice as number of peaks
            self.points.pop(-1)
            self.points_index.pop(-1)
            return num_points

    def setup_peak_search_algorithm(self, algorithm, mask=None):
        """
        Initializes the peak search algorithm on the current image
        :param algorithm:
            peak search algorithm used. Possible algorithms are 'Massif' and 'Blob'
        :param mask:
            if a mask is used during the process this is provided here as a 2d array for the image.
        """

        if algorithm == 'Massif':
            self.peak_search_algorithm = Massif(self.img_model.raw_img_data)
        elif algorithm == 'Blob':
            if mask is not None:
                self.peak_search_algorithm = BlobDetection(self.img_model.raw_img_data * mask)
            else:
                self.peak_search_algorithm = BlobDetection(self.img_model.raw_img_data)
            self.peak_search_algorithm.process()
        else:
            return

    def set_calibrant(self, filename):
        self.calibrant = Calibrant()
        self.calibrant.load_file(filename)
        self.pattern_geometry.calibrant = self.calibrant

    def search_peaks_on_ring(self, ring_index, delta_tth=0.1, min_mean_factor=1,
                             upper_limit=55000, mask=None):
        """
        This function is searching for peaks on an expected ring. It needs an initial calibration
        before. Then it will search for the ring within some delta_tth and other parameters to get
        peaks from the calibrant.

        :param ring_index: the index of the ring for the search
        :param delta_tth: search space around the expected position in two theta
        :param min_mean_factor: a factor determining the minimum peak intensity to be picked up. it is based
                                on the mean value of the search area defined by delta_tth. Pick a large value
                                for larger minimum value and lower for lower minimum value. Therefore, a smaller
                                number is more prone to picking up noise. typical values like between 1 and 3.
        :param upper_limit: maximum intensity for the peaks to be picked
        :param mask: in case the image has to be masked from certain areas, it need to be given here. Default is None.
                     The mask should be given as an 2d array with the same dimensions as the image, where 1 denotes a
                     masked pixel and all others should be 0.
        """
        self.reset_supersampling()
        if not self.is_calibrated:
            return

        # transform delta from degree into radians
        delta_tth = delta_tth / 180.0 * np.pi

        # get appropriate two theta value for the ring number
        tth_calibrant_list = self.calibrant.get_2th()
        if ring_index >= len(tth_calibrant_list):
            raise NotEnoughSpacingsInCalibrant()
        tth_calibrant = np.float(tth_calibrant_list[ring_index])

        # get the calculated two theta values for the whole image
        tth_array = self.pattern_geometry.twoThetaArray(self.img_model._img_data.shape)

        # create mask based on two_theta position
        ring_mask = abs(tth_array - tth_calibrant) <= delta_tth

        if mask is not None:
            mask = np.logical_and(ring_mask, np.logical_not(mask))
        else:
            mask = ring_mask

        # calculate the mean and standard deviation of this area
        sub_data = np.array(self.img_model._img_data.ravel()[np.where(mask.ravel())], dtype=np.float64)
        sub_data[np.where(sub_data > upper_limit)] = np.NaN
        mean = np.nanmean(sub_data)
        std = np.nanstd(sub_data)

        # set the threshold into the mask (don't detect very low intensity peaks)
        threshold = min_mean_factor * mean + std
        mask2 = np.logical_and(self.img_model._img_data > threshold, mask)
        mask2[np.where(self.img_model._img_data > upper_limit)] = False
        size2 = mask2.sum(dtype=int)

        keep = int(np.ceil(np.sqrt(size2)))
        try:
            sys.stdout = DummyStdOut
            res = self.peak_search_algorithm.peaks_from_area(mask2, Imin=mean - std, keep=keep)
            sys.stdout = sys.__stdout__
        except IndexError:
            res = []

        # Store the result
        if len(res):
            self.points.append(np.array(res))
            self.points_index.append(ring_index)

        self.set_supersampling()
        self.pattern_geometry.reset()

class NotEnoughSpacingsInCalibrant(Exception):
    pass
    
class DummyStdOut(object):
    @classmethod
    def write(cls, *args, **kwargs):
        pass