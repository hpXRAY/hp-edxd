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

from __future__ import absolute_import

import numpy as np
import scipy.optimize as opt

from burnman.eos import equation_of_state as eos
#from burnman.tools import bracket
import warnings
#from burnman.eos.birch_murnaghan import bulk_modulus, birch_murnaghan, volume




class JCPDS4(eos.EquationOfState):

    """
    Class based on legacy JCPDS4 style thermal Birch Murnaghan equation of state. 
    The following keywords are supported:
               K_0:         The bulk modulus in GPa.
               Kprime_0:        The change in K0 with pressure, for Birch-Murnaghan
                           equation of state.  Dimensionless.
               dk0dt:      The temperature derivative of K0, GPa/K.
               dk0pdt:     The temperature derivative of K0P, 1/K.
               V_0:     The unit cell volume
               alpha_t0:     The thermal expansion coefficient, 1/K
               d_alpha_dt:   The temperature derivative of the thermal expansion
                           coefficient, 1/K^2
    """

    def __init__(self):
        self.order = 3


    def volume(self, pressure, temperature, params):
        """
        Returns volume :math:`[m^3]` as a function of pressure :math:`[Pa]`.
        """

        params['alpha_t'] = params['alpha_t0'] + params['d_alpha_dt'] * (temperature - 298.)
        params['Kprime_0'] = params['Kprime_0'] + params['dk0pdt'] * (temperature - 298.)
        params['K_0'] = params['K_0'] + params['dk0dt'] * (temperature - 298.)

        if pressure == 0.:
            v= params['V_0'] * (1 + params['alpha_t'] * (temperature - 298.))
            return v
        if pressure < 0:
            if params['K_0'] <= 0.:
                return params['V_0']
            else:
                return params['V_0'] * (1 - pressure / params['K_0'])
        else:
            if params['K_0'] <= 0.:
                return params['V_0']
            else:
                self.mod_pressure = pressure - \
                        params['alpha_t'] * params['K_0'] * (temperature - 298.)

                if self.mod_pressure < 0:
                    return params['V_0'] * (1 - pressure / params['K_0'])
                self.params = params
                res = opt.minimize(self.bm3_inverse, 1.)
                v = self.params['V_0'] / res.x[0]
                return v
        
    def bm3_inverse(self, v0_v):
        """
        Returns the value of the third order Birch-Murnaghan equation minus
        pressure.  It is used to solve for V0/V for a given
           P, K0 and K0'.

        Inputs:
           v0_v:   The ratio of the zero pressure volume to the high pressure
                   volume
        Outputs:
           This function returns the value of the third order Birch-Murnaghan
           equation minus pressure.  \

        Procedure:
           This procedure simply computes the pressure using V0/V, K0 and K0',
           and then subtracts the input pressure.

        Example:
           Compute the difference of the calculated pressure and 100 GPa for
           V0/V=1.3 for alumina
           jcpds = obj_new('JCPDS')
           jcpds->read_file,  'alumina.jcpds'
           common bm3_common mod_pressure, k0, k0p
           mod_pressure=100
           k0 = 100
           k0p = 4.
           diff = jcpds_bm3_inverse(1.3)
        """

        return (1.5 * self.params['K_0'] * (v0_v ** (7. / 3.) - v0_v ** (5. / 3.)) *
                (1 + 0.75 * (self.params['Kprime_0'] - 4.) * (v0_v ** (2. / 3.) - 1.0)) -
                self.mod_pressure) ** 2

    def pressure(self, temperature, volume, params):
        v0 = params['V_0']
        params['alpha_t'] = params['alpha_t0'] + params['d_alpha_dt'] * (temperature - 298.)
        v_over_v0 = volume/v0
        v0_v = 1/v_over_v0
        P = 1.5 * params['K_0'] * (v0_v ** (7. / 3.) - v0_v ** (5. / 3.)) * \
                (1 + 0.75 * (params['Kprime_0'] - 4.) * (v0_v ** (2. / 3.) - 1.0)) + \
                        params['alpha_t'] * params['K_0'] * (temperature - 298.)
        return P


    def validate_parameters(self, params):
        """
        Check for existence and validity of the parameters
        """

        if 'P_0' not in params:
            params['P_0'] = 0.

       
        # Check that all the required keys are in the dictionary
        expected_keys = ['V_0', 'K_0', 'Kprime_0']
        for k in expected_keys:
            if k not in params:
                raise KeyError('params object missing parameter : ' + k)

        # Finally, check that the values are reasonable.
        if params['P_0'] < 0.:
            warnings.warn('Unusual value for P_0', stacklevel=2)
        if params['V_0'] < 1.e-7 or params['V_0'] > 1.e-3:
            warnings.warn('Unusual value for V_0', stacklevel=2)
        if params['K_0'] < 1.e9 or params['K_0'] > 1.e13:
            warnings.warn('Unusual value for K_0', stacklevel=2)
        if params['Kprime_0'] < 0. or params['Kprime_0'] > 10.:
            warnings.warn('Unusual value for Kprime_0', stacklevel=2)
        

  