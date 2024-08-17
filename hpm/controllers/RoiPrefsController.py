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




from utilities.hpMCAutilities import compare
from utilities.hpMCAutilities import Preferences
from hpm.widgets.RoiPrefsWidget import RoiPreferencesWidget
from PyQt6.QtCore import QObject
import json
from utilities.hpMCAutilities import displayErrorMessage, json_compatible_dict, readconfig

import os.path, os
from pathlib import Path
home_path = os.path.join(str(Path.home()), 'hpMCA')
if not os.path.exists(home_path):
   os.mkdir(home_path)

class RoiPreferences(Preferences):
    def __init__(self, phase_controller):
        self.phase_controller = phase_controller
        self.config_file = os.path.join(home_path,'hpMCA_roi_settings.json')
        prefs, values = opt_fields(self.config_file)
        self.widget = RoiPreferencesWidget(prefs, 
                            title='ROI preferences control')
        params = list(prefs.keys())
        super().__init__(params)
        self.name = "ROI Preferences"
        self.note = ''
        self.make_connections()
        self.auto_process = True
        self.set_config(values)

    def make_connections(self):
        self.widget.apply_clicked_signal.connect(self.apply_callback)

    def apply_callback(self, params):
        self.set_config(params)
        save_config(params, self.config_file)

    # the update method gets called automatically by the parent class, 
    # depending if any parameters actually changed
    def update(self):
        updated_params = {}
        for c in self.config_modified:
            updated_params[c] = self.params[c]
        if len(updated_params):
            # this is where we can do something with the updated parameters
            self.phase_controller.set_prefs(updated_params)

    def show(self):
        prefs, values = opt_fields(self.config_file)
        p = {}
        for pref in prefs:
            p[pref]=prefs[pref]['val']
        self.widget.set_params(p)
        self.widget.raise_widget()

    
def save_config(params, filename):
    options_out = json_compatible_dict(params)
    try:
        with open(filename, 'w') as outfile:
            json.dump(options_out, outfile,sort_keys=True, indent=4)
            outfile.close()
    except:
        pass


def opt_fields(config_file):
    values = readconfig(config_file)

    opts = {'roi':
                {
                    'a_0': {'val': 7,
                     'desc': '',
                     'label': 'a0'},
                    'a_1': {'val': 0.08,
                     'desc': '',
                     'label': 'a1'},
                    'e_min': {'val': 0,
                     'desc': '',
                     'label': 'E min'},
                    'e_max': {'val':  100,
                     'desc': '',
                     'label': 'E max'}
                }
            }

    prefs = opts['roi']
    for c in values:
        if c in prefs:
            prefs[c]['val']= values[c]
    return prefs, values
    
