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

"""aEDXD_functions.py:
   Python functions for aEDXD_class.py"""
__author__    = "Changyong Park (cpark@carnegiescience.edu)"
__version__   = "1.3"
__date__      = "04 August 2015"
__copyright__ = "Copyright (c) 2015 Changyong Park"
__license__   = "LICENSE-CIW(Free)"

import numpy as np
from scipy import interpolate
from scipy.special import erfc
from scipy.optimize import curve_fit
from scipy import signal
import json
import matplotlib.pyplot as plt
import copy

def fversion():
    return __version__

def onpick(event):
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    print('onpick points:'+zip(xdata[ind], ydata[ind]))

def spline_fit(x,y,w,bbox,k,s):
    spl = interpolate.UnivariateSpline(x,y,w,bbox,k,s)
    return spl

def polynomial_fit(x,y,deg):
    pln = np.polyfit(x,y,deg,rcond=None,full=False)
    return pln
    
def fastbin(data,bin_size):
    bin_means = []
    data = np.array(data)
    nbins = int(len(data)/bin_size)
    for i in range(nbins-1):
        if bin_size*(i+1) <= (len(data)-1):
            bin_means.append(data[bin_size*i:(bin_size*(i+1))].mean())
        else:
            bin_means.append(data[bin_size*i:(len(data)-1)].mean())
    return np.array(bin_means) 

def custom_fit(func,x,y,p0,yb):
    p_opt,p_cov = curve_fit(func, x, y, p0, sigma=yb)
    # calculate stdev and correlatoin matrix of estimated parameters
    p_stdev = np.sqrt(np.diag(p_cov))
    d = np.sqrt(np.abs(np.diag(p_cov)))
    d = np.matrix(d)        
    p_corp = p_cov/(d.T*d)
    yp = func(x, *p_opt)
    if yb is None :
        schi2 = np.sum((y-yp)**2)/(len(y)-len(p_opt))
    else:
        schi2 = np.sum(((y-yp)/yb)**2)/(len(y)-len(p_opt))
    r = np.corrcoef(y,yp)
    '''
    print("\nSciPy.optimize.curve_fit:"+
      "\np_opt  : \n" + str(p_opt)+
      "\np_stdev: \n" + str(p_stdev)+
      "\np_corp : \n" + str(p_corp) +
      "\nscaled chi^2 = " + str(schi2) +
      "\nr = " + str(r[0][1]))
    '''
    return p_opt,p_stdev,p_cov,p_corp,schi2,r[0][1]

def stepped_polynomial(x,*p0):
    s = p0[0]; w = p0[1]; x0 = p0[2]; pln = p0[3:]
    poly_y = []
    deg = len(pln)-1
    for i in range(len(x)):
        poly_y.append(sum(pln[deg-j]*x[i]**j for j in range(deg+1)))
    poly_y = np.array(poly_y)
    y = (1+s*erfc(1/w*(x-x0)))*poly_y
    return y

def simple_polynomial(x,*p0):
    pln = p0[0:]
    poly_y = []
    deg = len(pln)-1
    for i in range(len(x)):
        poly_y.append(sum(pln[deg-j]*x[i]**j for j in range(deg+1)))
    y = np.array(poly_y)
    return y

def I_base_calc(q,q_comp,p):
    """*par must have the following structure:
    [ Z, Atomic fraction, a1,b1,a2,b2,a3,b3,a4,b4,c, M,K,L ]
    """

    

    s = np.array(q/4/np.pi)
    s_comp = np.array(q_comp/4/np.pi)
    mean_fqsquare = np.zeros(len(q))
    mean_fq = np.zeros(len(q))
    mean_I_inc = np.zeros(len(q))
    for i in range(len(p)):

        abc = p[i][2:11]
        
        fqi = p[i][2]*np.exp(-p[i][3]*s**2)+\
              p[i][4]*np.exp(-p[i][5]*s**2)+\
              p[i][6]*np.exp(-p[i][7]*s**2)+\
              p[i][8]*np.exp(-p[i][9]*s**2)+p[i][10]
        mean_fqsquare += p[i][1]*fqi**2 # fractionized
        mean_fq += p[i][1]*fqi
        # Compton I_inc(Q')
        fqi_comp = p[i][2]*np.exp(-p[i][3]*s_comp**2)+\
              p[i][4]*np.exp(-p[i][5]*s_comp**2)+\
              p[i][6]*np.exp(-p[i][7]*s_comp**2)+\
              p[i][8]*np.exp(-p[i][9]*s_comp**2)+p[i][10]

        Z = p[i][0]
        I_inci = np.zeros(len(q))
        if len(p[i])> 16:
            # ab5 contains parameters for calculating incoherent scattering in different form than legacy MKL table
            # from Ref: H.H.M. Balyuzi, Acta Cryst. (175). A31, 600
            ab5 = [Z, *p[i][14:24]]
            I_inci = I_inc_new(s_comp, ab5)
        else:
            # 2020-09-05 added parenthesis around "p[i][0]-np.array(fqi_comp)**2/p[i][0]"
            # since they were missing in previous versions
            # Ref. F. Hajdu, Acta Cryst. (1971). A27, 73
            mkl = p[i][11:14]
            if not mkl[0] is None and not mkl[1] is None and not mkl[2] is None:
                I_inci = (Z-np.array(fqi_comp)**2/Z)*\
                        (1-p[i][11]*(np.exp(-p[i][12]*s_comp)-np.exp(-p[i][13]*s_comp)))
                
        mean_I_inc += p[i][1]*I_inci
    return mean_fqsquare, mean_fq, mean_I_inc

def I_inc_new(s, param):
    """
    s = lambda^-1 * sin(theta)
    param must have the following structure:
    [ Z, A1,	B1,	A2,	B2,	A3,	B3,	A4,	B4,	A5,	B5]
    Ref: H.H.M. Balyuzi, Acta Cryst. (175). A31, 600
    """
    Z = param[0]
    
    ab = param[1:]
    #s = np.array(q /4/np.pi)
    mean_I_inc = Z - (ab[0]*np.exp(-ab[1]*s**2)+\
                     ab[2]*np.exp(-ab[3]*s**2)+\
                     ab[4]*np.exp(-ab[5]*s**2)+\
                     ab[6]*np.exp(-ab[7]*s**2)+\
                     ab[8]*np.exp(-ab[9]*s**2))

    return mean_I_inc

    
def find_Iq_scale(Iq,sq_base):
    return np.linalg.solve(Iq,sq_base)

# to resolve py2exe error, from http://www.pyinstaller.org/ticket/596
def dependencies_for_myprogram():
    from scipy.sparse.csgraph import _validation



def is_e(val):
    
    s=str(val-int(val))
    if 'e-' in s:

        return True
    return False


def rebin_weighted(x,y,weights, rebin_x):
    rebin_x_bkp = copy.deepcopy(rebin_x)

    first_value_A = rebin_x[0]
    last_value_A = (rebin_x[-1] - rebin_x[0]) * 0.3
    last_value_B = (rebin_x[-1] - rebin_x[0]) * 0.6
    last_value_C = rebin_x[-1]
    step_A = rebin_x[1] - rebin_x[0]
    step_C = step_A * 10

    q_range = rebin_x[-1] - rebin_x[0]
    first_value_A = rebin_x[0]
    last_value_A = rebin_x[0] + q_range * 0.1
    last_value_B = rebin_x[0] + q_range * 0.3
    last_value_C = rebin_x[0] + q_range * 0.4
    last_value_D = rebin_x[0] + q_range * 0.6
    last_value_E = rebin_x[-1]
    step_A = rebin_x[1] - rebin_x[0]
    step_B = step_A * 3
    step_C = step_A * 6
    step_D = step_A * 12
    step_E = step_A * 18

    variable_bins, t = generate_array_with_index(first_value_A, last_value_A, last_value_B, last_value_C, last_value_D, last_value_E, step_A, step_B, step_C, step_D, step_E)

    '''plt.plot(t, variable_bins, '-o')
    plt.xlabel('Index')
    plt.ylabel('x')
    plt.title('Array Plot')
    plt.grid(True)
    plt.show()'''

    num_bins = len(variable_bins)
    # get first step
    step = variable_bins[1]-variable_bins[0]
    

    # Calculate the bin edges
    bin_edges = []
    bin_edges.append(variable_bins[0]-step/2)
    for i,r_x in enumerate(variable_bins):
        if i < len(variable_bins)-1:
            step = variable_bins[i+1]-variable_bins[i]
        bin_edges.append(r_x+step/2)
        
    # Initialize empty lists for rebinned data
    rebin_x = []
    rebin_y = []

    # Distribute data into bins
    for i in range(num_bins):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
        if np.sum(weights[mask]) != 0:  # Check if the weights sum to zero
            rebin_x.append((bin_edges[i] + bin_edges[i + 1]) / 2)  # Use bin centers as rebinned x values
            rebin_y.append(np.average(y[mask], weights=weights[mask]))  # Weighted average of y values within each bin

    last_step = rebin_x[-1]-rebin_x[-2]
    for i in range(20):
        rebin_x.append(rebin_x[-1] + last_step)
        rebin_y.append(rebin_y[-1])

    spl = interpolate.interp1d(rebin_x,rebin_y,kind='quadratic', fill_value='extrapolate')

    sq_even = spl(rebin_x_bkp) # evenly spaced I(q)

    #sq_even = np.array(variable_cutoff_filter(np.array(sq_even) -1, .99999, 0.5)) + 1
    
    return rebin_x_bkp, sq_even

def quadratic_sequence(start, end, num_points):
    x = np.linspace(0, 1, num_points)
    y = x**2
    sequence = start + (end - start) * y
    sequence[-1] = end  # Set the last value to the specified end_value
    return x, sequence

def generate_array_with_index(first_value_A, last_value_A, last_value_B, last_value_C, last_value_D, last_value_E, step_A, step_B, step_C, step_D, step_E):
    

    # Generate array A with the specified first and last values and step size
    array_A = np.arange(first_value_A, last_value_A, step_A)

    # Generate array B with the calculated average step size, matching the last value of array A
    array_B = np.arange(array_A[-1], last_value_B, step_B)

    # Generate array C with the specified last value and step size
    array_C = np.arange(array_B[-1], last_value_C, step_C)

    # Generate array D with the specified last value and step size
    array_D = np.arange(array_C[-1], last_value_D, step_D)

    # Generate array E with the calculated average step size, matching the last value of array D
    array_E = np.arange(array_D[-1], last_value_E, step_E)

    array_A = array_A[:-1]
    array_B = array_B[:-1]
    array_C = array_C[:-1]
    array_D = array_D[:-1]
    
    # Concatenate arrays A, B, C, D, and E
    result_array = np.concatenate((array_A, array_B, array_C, array_D, array_E))
    result_index = np.arange(len(result_array))

    return result_array, result_index