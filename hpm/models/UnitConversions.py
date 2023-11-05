import numpy as np
from math import sin, pi

def E_to_q(e,tth):
    #0.008726646259972
    q = 6.28318530718 /(6.199 / e / sin(tth*0.008726646259972))
    #q = 6.28318530718 /(6.199 / e / sin(tth/180.*pi/2.))
    return q

def q_to_E(q, tth):
    e = 6.199 / (sin(tth * 0.008726646259972) * (6.28318530718 / q))
    return e

def E_to_d(e,tth):
    q = E_to_q(e,tth)
    d = q_to_d(q)
    return d

def d_to_q(d):
    q = 2. * pi / d
    return q

def q_to_d(q):
    d = 2. * pi / q
    return d

def tth_to_d(tth,wavelength):
    
    d = wavelength/(2.0 * np.sin(tth * 0.008726646259972))
    return d

def tth_to_q(tth,wavelength):
    d = tth_to_d(tth,wavelength)
    q = d_to_q(d)
    return q

def y_scale(y, scale,offset,log_scale):
    if log_scale:
        y = np.clip(y,0.5,None)
    y = y*scale+offset
    if log_scale:
        y = np.clip(y,10e-8,None)
    return y