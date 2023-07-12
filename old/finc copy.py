from axd.models.aEDXD_functions import I_base_calc, I_inc_new
import numpy as np
#import matplotlib
from axd.models.aEDXD_atomic_parameters import aEDXDAtomicParameters
import json
#import matplotlib.pyplot as plt

qp = range(1000)
qp = np.asarray(qp)/50

elements_file='utilities/elements.json'
with open(elements_file,'r') as f:
        elements = json.loads(f.read())

        f.close()



ap = aEDXDAtomicParameters()

for e in elements:



        symbol = e['symbol']

        ss = ap.ab5['str_data']
        if symbol in ss:

                opt = ap.get_ab5_options(symbol)

                i_inc = I_inc_new(qp, opt[symbol])


                #plt.plot(qp, i_inc, linewidth=0.5)

#plt.show()

'''Z = []
        I = []
        if symbol in ss:

                opt = ap.get_ab5_options(symbol)[symbol]
                z = opt[0]

                i_inc = I_inc_new(qp, opt)[-1]

                Z.append(z)
                I.append(i_inc)

        plt.plot(np.asarray(Z), np.asarray(I), 'ro')'''