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


# Principal author: R. Hrubiak (hrubiak@anl.gov)
# Copyright (C) 2018-2019 ANL, Lemont, USA



import numpy as np
import utilities.CARSMath as CARSMath



def fit_energies(roi, degree, calibration):
    """ Private method """
    
    use = []
    nrois = len(roi)
    for i in range(nrois):
        if (roi[i].use): use.append(i)
    nuse = len(use)
    if ((degree == 1) and (nuse < 2)):
        #tkMessageBox.showerror(title='mcaCalibateEnergy Error', 
        message='Must have at least two valid points for linear calibration'
        print(message)
        return
    elif ((degree == 2) and (nuse < 3)):
        #tkMessageBox.showerror(title='mcaCalibateEnergy Error', 
        message='Must have at least three valid points for quadratic calibration'
        print(message)
        return
    chan=np.zeros(nuse)
    energy=np.zeros(nuse)
    weights=np.ones(nuse)
    for i in range(nuse):
        chan[i] = roi[use[i]].centroid
        energy[i] = roi[use[i]].energy
    coeffs = CARSMath.polyfitw(chan, energy, weights, degree)
    calibration.offset = coeffs[0]
    calibration.slope = coeffs[1]
    if (degree == 2):
        calibration.quad = coeffs[2]
    else:
        calibration.quad = 0.0
    calibration.set_dx_type('edx')

def calc_parabola_vertex( x1:float, y1:float, x2:float, y2:float, x3:float, y3:float):
    '''
    Adapted and modifed to get the unknowns for defining a parabola:
    http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
    '''

    denom = (x1-x2) * (x1-x3) * (x2-x3)
    A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
    B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
    C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom

    return A,B,C