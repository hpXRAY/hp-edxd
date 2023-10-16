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
from .calibrant import Calibrant
from scipy.optimize import curve_fit

from .. import calibrants_path

class GSDCalibrationModel(QtCore.QObject):  # 
    def __init__(self,  *args, **filekw):
        
        """
        Creates new Multiple Spectra object.  
    
        Example:
            m = MultipleSpectraModel()
        """
        self.data_raw = None
        self.bin = 8
        self.data = None
        self.E_scale = None
        self.mask = None
        self.points = []
        self.points_index = []
        self.start_values = {'dist': 1,
                             'two_theta': 15,
                             'pixel_width': 260e-6,
                             'wavelength': 0.4e-10}

    def set_data(self, E_scale, data):
        self.data_raw = data
        self.E_scale = E_scale
        n = data.shape[0]
        m = data.shape[1]
        # Calculate the number of new columns
        bin = self.bin
        new_m = m // bin

        # Reshape and sum every 20 columns
        reshaped_E_arr = data.reshape(n, new_m, bin).sum(axis=2)
        self.data = reshaped_E_arr

        self.clear_peaks()
        
        self.setup_peak_search_algorithm('Massif')

    def flip_img_vertically(self):
        data = np.flipud(self.data_raw)
        self.set_data(data)
       
    def convert_point_E_to_channel(self, E):
        current_translate = self.E_scale[1]
        current_scale = self.E_scale[0]

        inverse_translate = -1*current_translate
        inverse_scale =  1/current_scale

        x = E + inverse_translate
        x = x * inverse_scale
        channel = x
        return channel

    def convert_point_channel_to_E(self, channel):
        current_translate = self.E_scale[1]
        current_scale = self.E_scale[0]

        x_data = channel * current_scale
        energy = x_data + current_translate

        return energy

    def add_point(self, y, x):
        x = self .convert_point_E_to_channel(x) // self.bin

        peak = self.find_peak(y,x, 4, 0)
        peaks = self.find_peaks_automatic(*peak[0],0)
        # Extract x and y values from the list of tuples
        y_data, x_data = zip(*peaks)
        x_data = np.array(x_data)* self.bin + 0.5

        x_data = self.convert_point_channel_to_E(x_data)

        y_data = np.array(y_data)
        x_max = self.data.shape[0]
        a, b, c, y_range, x_range = fit_and_evaluate_polynomial(y_data, x_data, x_max)

        return x_data, y_data

    def create_point_array(self, points, points_ind):
        res = []
        for i, point_list in enumerate(points):
            if point_list.shape == (2,):
                res.append([point_list[0], point_list[1], points_ind[i]])
            else:
                for point in point_list:
                    res.append([point[0], point[1], points_ind[i]])
        return np.array(res)

    def get_point_array(self):
        return self.create_point_array(self.points, self.points_index)
   
        
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
        massif = Massif(self.data)
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
        search_array = self.data[left_ind:(left_ind + search_size), top_ind:(top_ind + search_size)]
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
            self.peak_search_algorithm = Massif(self.data)
        elif algorithm == 'Blob':
            if mask is not None:
                self.peak_search_algorithm = BlobDetection(self.data * mask)
            else:
                self.peak_search_algorithm = BlobDetection(self.data)
            self.peak_search_algorithm.process()
        else:
            return

    def set_calibrant(self, filename):
        self.calibrant = Calibrant()
        self.calibrant.load_file(filename)


    def set_Es_for_element(self, Es = []):
        if len(Es) == 0:
            return
        


class NotEnoughSpacingsInCalibrant(Exception):
    pass
    
class DummyStdOut(object):
    @classmethod
    def write(cls, *args, **kwargs):
        pass

def second_order_polynomial(x, a, b, c):
    return a * x**2 + b * x + c

def fit_and_evaluate_polynomial(x_data, y_data, x_max):
    popt, _ = curve_fit(second_order_polynomial, x_data, y_data)

    # Extract the coefficients
    a, b, c = popt

    # Define the range for x values
    x_range = np.arange(0, int(x_max) + 1, 1)

    # Calculate the corresponding y values using the polynomial
    y_range = second_order_polynomial(x_range, a, b, c)

    return a, b, c, x_range, y_range