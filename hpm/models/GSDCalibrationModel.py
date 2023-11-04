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



import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtCore

from functools import partial
from .mcareaderGeStrip import *
from hpm.models.UnitConversions import *

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
        self.tth_calibrated = np.zeros(192)+15
        self.mask = None
        self.points = []
        self.points_index = []
        self.start_values = {'dist': 1,
                             'two_theta': 15,
                             'pixel_width': 260e-6,
                             'wavelength': 0.4e-10}

        self.calibrated_d_spacings = {}

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


    def do_2theta_calibration(self):
        points = self.get_point_array()
        cal_ds = np.asarray(self.calibrant.get_dSpacing())
        
        y, x_binned, ind = zip(*points)
        x = np.array(x_binned)* self.bin 
        
        y = np.array(y)
        ind = np.array(ind, dtype=int)
        
        d = cal_ds[ind]
        e = self.convert_point_channel_to_E(x)
        self._calibrate_2theta(e,y,d)

    def _calibrate_2theta(self, e, y, d):


        tths = []

        
        tth = 2.0 * np.arcsin(12.398 / (2.0*e*d))*180./np.pi

        unique_y = sorted(list(set(list(y))))
        unique_tth = []
        for u_y in unique_y:
            u_tth = np.mean(tth[y==u_y])
            unique_tth.append(u_tth)

        unique_tth = np.asarray(unique_tth)/180*np.pi
        unique_y = 191 - np.asarray(unique_y)
        a, b, c, y_range, tth_range_estimate = fit_and_evaluate_polynomial( unique_y,unique_tth, 191)
        guess_tth = np.mean(tth_range_estimate)
        poni_y, poni_angle, distance,tilt, y_range, tth_range = fit_poni_relationship(unique_y,unique_tth,191)
        tth_range=  np.flip(tth_range) *180/np.pi

        self.tth_calibrated = tth_range

    def get_simulated_lines(self,tth_range):
        tth_range = self.tth_calibrated
        cal_ds = np.asarray(self.calibrant.get_dSpacing())
        segments_x = []
        segments_y = []
        calibrated_d_spacings = {}
        for i in range(min(10, len(cal_ds))):
            x = np.empty_like(tth_range)
            y = np.empty_like(tth_range)
            for j, tth in enumerate(tth_range):
                
                    q = d_to_q(cal_ds[i])
                    e = q_to_E(q, tth)
                    if e <= 100 and e > 0:
                        y[j] = j
                        x[j] = e
                    else:
                        y[j] = np.nan
                        x[j] = np.nan
            segments_x.append(x)
            segments_y.append(y)
            calibrated_d_spacings[cal_ds[i]]=[x,y]
        self.calibrated_d_spacings = calibrated_d_spacings
        return segments_x, segments_y

    def refine_2theta_calibration(self, ):
        bin = self.bin
        segments_x = []
        segments_y = []
        segments_d = []
        segments_channel = []
        d_spacings = list(self.calibrated_d_spacings.keys())
        skip_ranges = self.convert_point_E_to_channel(np.asarray([66,70,77,81,15,76]))
        mask = np.zeros(self.data_raw.shape[1])
        mask[int(skip_ranges[0]):int(skip_ranges[1])] = 1
        mask[int(skip_ranges[2]):int(skip_ranges[3])] = 1
        mask[:int(skip_ranges[4])] = 1
        mask[int(skip_ranges[5]):] = 1

        for i, d in enumerate(d_spacings[:4]):
            
            energy, y = self.calibrated_d_spacings[d]
            
            channels = self.convert_point_E_to_channel(energy)
            ys = []
            centers = []
            for y, channel in enumerate(channels):
                if not np.isnan(channel):
                    ch_low_bound = int(channel - bin* 10)
                    ch_high_bound = int(channel + bin* 10)
                    skip = mask[ch_low_bound:ch_high_bound].any()==1 
                    if not skip:
                        roi_start = int(channel//bin-12)
                        roi_end = int(channel//bin+12)
                        roi = self.data[y, roi_start:roi_end]
                        roi_center = find_peak_center(roi)
                        if not np.isnan(roi_center):
                            center = roi_center + roi_start
                            centers.append(center)
                            ys.append(y)
            
            centers = np.asarray(centers)
            ys = np.asarray(ys)
            d_array = np.zeros_like(ys)+ d
        
            
            segments_x.append(centers)
            segments_y.append(ys)
            segments_d.append(d_array)

        x = np.concatenate(segments_x)* self.bin
        e = self.convert_point_channel_to_E(x)
        y = np.concatenate(segments_y)
        d = np.concatenate(segments_d)

       
        
        self._calibrate_2theta(e,y,d)

class NotEnoughSpacingsInCalibrant(Exception):
    pass
    
class DummyStdOut(object):
    @classmethod
    def write(cls, *args, **kwargs):
        pass

def second_order_polynomial(x, a, b, c):
    return a * x**2 + b * x + c

def poni_2theta_relationship(x,poni_x, poni_angle, distance):

    if np.array_equal(x, poni_x):
        result = poni_angle
    elif np.all(x < poni_x):
        result =  poni_angle - np.arctan((poni_x - x) / distance)
    elif np.all(x > poni_x):
        result =  poni_angle + np.arctan((x - poni_x) / distance)
    else:
        # Handle the case where x contains a mix of values less and greater than poni_x
        result = np.empty_like(x, dtype=float)
        result[x == poni_x] = poni_angle
        result[x < poni_x] = poni_angle - np.arctan((poni_x - x[x < poni_x]) / distance)
        result[x > poni_x] = poni_angle + np.arctan((x[x > poni_x] - poni_x) / distance)
    
    return result

def poni_2theta_relationship_fixed_poni_x(poni_x, x, poni_angle, distance):

    if np.array_equal(x, poni_x):
        result = poni_angle
    elif np.all(x < poni_x):
        result =  poni_angle - np.arctan((poni_x - x) / distance)
    elif np.all(x > poni_x):
        result =  poni_angle + np.arctan((x - poni_x) / distance)
    else:
        # Handle the case where x contains a mix of values less and greater than poni_x
        result = np.empty_like(x, dtype=float)
        result[x == poni_x] = poni_angle
        result[x < poni_x] = poni_angle - np.arctan((poni_x - x[x < poni_x]) / distance)
        result[x > poni_x] = poni_angle + np.arctan((x[x > poni_x] - poni_x) / distance)
    
    return result

def poni_with_tilt( poni_x, poni_angle, x, tilt, distance): 

    if np.array_equal(x, poni_x):
        result = poni_angle
    elif np.all(x < poni_x):
        result =  poni_angle - np.arctan((poni_x - x)* np.cos(tilt) / (distance - (poni_x - x)*np.sin(tilt)))
    elif np.all(x > poni_x):
        result =  poni_angle + np.arctan((x - poni_x)* np.cos(tilt) / (distance + (x - poni_x)*np.sin(tilt)))
    else:
        # Handle the case where x contains a mix of values less and greater than poni_x
        result = np.empty_like(x, dtype=float)
        result[x == poni_x] = poni_angle
        result[x < poni_x] = poni_angle - np.arctan((poni_x - x[x < poni_x])* np.cos(tilt) / (distance - (poni_x - x[x < poni_x])*np.sin(tilt)))
        result[x > poni_x] = poni_angle + np.arctan((x[x > poni_x] - poni_x)* np.cos(tilt) / (distance + (x[x > poni_x] - poni_x)*np.sin(tilt)))
    
    return result 


def third_order_polynomial(x, a, b, c, d):
    return a * x**2 + b * x + c + d * x**3

def fit_and_evaluate_polynomial(x_data, y_data, x_max):
    popt, _ = curve_fit(second_order_polynomial, x_data, y_data)

    # Extract the coefficients
    a, b, c = popt

    # Define the range for x values
    x_range = np.arange(0, int(x_max) + 1, 1)

    # Calculate the corresponding y values using the polynomial
    y_range = second_order_polynomial(x_range, a, b, c)

    return a, b, c, x_range, y_range 

def fit_poni_relationship(x, tth, x_max=191):
    det_size = 50 #mm
    num_elements = 192

    distance_0_in_mm = 1000
    element_size = det_size / num_elements
    distance_0_in_pixels = distance_0_in_mm / element_size
    tth_0 = np.mean(tth)

    poni_x_0, poni_angle_0, distance_0 = num_elements // 2, tth_0 , distance_0_in_pixels
    popt, _ = curve_fit( partial(poni_2theta_relationship_fixed_poni_x,poni_x_0), x, tth, p0= [poni_angle_0, distance_0])

    # Extract the coefficients
    poni_angle, distance = popt
    print(poni_x_0, ' ', poni_angle * 180 / np.pi, ' ', distance)

    tilt_0 = 0.
    popt, _ = curve_fit( partial(poni_with_tilt,poni_x_0, poni_angle),x,tth,p0= [tilt_0, distance])
    tilt, distance = popt
    print('tilt ', tilt * 180 / np.pi, ' distance ', distance)

    # Define the range for x values
    x_range = np.arange(0, int(x_max) + 1, 1)

    # Calculate the corresponding y values using the polynomial
    y_range = poni_with_tilt(poni_x_0, poni_angle,  x_range, tilt, distance)
    y_range_0_tilt = poni_with_tilt(poni_x_0, poni_angle, x_range, 0, distance)

    '''# Plot the results
    plt.plot(x_range, y_range *180 /np.pi, c='blue')
    plt.plot(x_range, y_range_0_tilt *180 /np.pi, c='orange')
    plt.scatter(x, tth *180 /np.pi, c='red', marker='o', s=20, label='Data Points')
    plt.xlabel('x')
    plt.ylabel('Angle (radians)')
    plt.title('Angle vs. x')
    plt.grid(True)
    plt.show()'''


    return poni_x_0, poni_angle * 180 / np.pi, distance, tilt, x_range, y_range


def find_peak_center(data, num_points=2):
    n = len(data)
    x = np.arange(n)

    # Calculate the background using an average of multiple points at the start and end
    background_start = np.mean(data[:num_points])
    background_end = np.mean(data[-num_points:])
    # Create a trimmed data array to match the background start and end positions
    trimmed_data = data[num_points//2:-num_points//2]
    # Create a corresponding x array for the trimmed data
    trimmed_x = x[num_points//2:-num_points//2]
    # Calculate a linear background for the trimmed data
    background = (background_end - background_start) / (len(trimmed_data) - 1) * (trimmed_x - trimmed_x[0]) + background_start
    data_adjusted = trimmed_data - background
    # Normalize the data to 1
    normalized_data = data_adjusted / np.max(data_adjusted)
    # Find the FWHM points
    half_max = 0.5
    above_half = normalized_data > half_max
    fwhm_points = np.where(above_half)[0]
    fwhm_center = int((fwhm_points[0] + fwhm_points[-1]) // 2.0)
    # Create a tighter background within 1.5 of FWHM distance from the new peak center
    fwhm_distance = int((fwhm_points[-1] - fwhm_points[0])*1.5 )
    background_start_index = fwhm_center - fwhm_distance 
    background_end_index = fwhm_center + fwhm_distance 

    if background_start_index >=0 and background_end_index < len(normalized_data):
        background_start = normalized_data[background_start_index]
    
        background_end = normalized_data[background_end_index]
        indexes_surrounding_center = fwhm_center - fwhm_distance + np.arange(2* fwhm_distance+1)
        x_tight = trimmed_x[indexes_surrounding_center]
        tighter_background = (background_end - background_start) / (2 * fwhm_distance) * (x_tight-x_tight[0] ) + background_start
        normalized_data_tight = normalized_data[indexes_surrounding_center]
        data_adjusted_tight = normalized_data_tight - tighter_background
        data_squared = data_adjusted_tight**2
        # Compute the center of mass of the square of the data, squaring suppresses the contribution from background
        center_of_mass = np.sum(x_tight * data_squared) / np.sum(data_squared)
        return center_of_mass #, fwhm_points, x_tight, data_adjusted_tight
    else:
        return np.nan