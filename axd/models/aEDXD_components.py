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


from axd.models.aEDXD_functions import *
import copy
import os
import time
import pyqtgraph as pg


from utilities.hpMCAutilities import compare
from utilities.hpMCAutilities import Preferences as Calculator
from scipy.interpolate import interp1d

class primaryBeam(Calculator):
    def __init__(self):
        params = ['inputdataformat',
                'inputdatadirectory',
                'itr_comp',
                'polynomial_deg',
                'Emax',
                'Emin',
                'sq_par',
                'E_cut',
                'x',
                'y',
                'thh']
        super().__init__(params)
        self.name = "primary beam"
        self.note = ''

    def update(self):
        Emin = self.params['Emin']
        Emax = self.params['Emax']
        polynomial_deg = int(self.params['polynomial_deg'])
        itr_comp = int(self.params['itr_comp'])
        sq_par = self.params['sq_par']
        x = self.params['x']
        #print('len x = '+str(len(x)))
        y = self.params['y']
        tth = self.params['thh']

        """primary beam analysis"""
        
        Emin_indx = (np.abs(x-Emin)).argmin() # find the nearest index to Emin
        Emax_indx = (np.abs(x-Emax)).argmin() # find the nearest index to Emax
        # pre-fitting with polynomial fit
        
        if Emax > 69:
            model_func = stepped_polynomial
        elif Emax <= 69:
            model_func = simple_polynomial
        elif Emin > 69:
            model_func = simple_polynomial
        xp = x[Emin_indx:Emax_indx]
        yp = y[Emin_indx:Emax_indx]
        pln = polynomial_fit(xp,yp,deg=polynomial_deg)
        if model_func == stepped_polynomial:
            p_erf = [0.4, 1.0, 70] # s:scale, w: width, x0: reference
            p0 = p_erf + list(pln)
            p = '['
            for i in p0:
                s = '%.3e'%(i)
                p = p + s + ', '
            p = p[:-2]+']'
            self.note = [p, 
            
                        "[s, w, x<sub>0</sub>, p[0], p[1],...p[n]] for stepped polynomial",
                        "y(x) = (1+s*erfc(1/w*(x-x<sub>0</sub>)))*(p[0]*x<sup>n</sup>+p[1]*x<sup>n-1</sup>+...+p[n])"]
             
        elif model_func == simple_polynomial:
            p0 = list(pln)
            p = '['
            for i in p0:
                s = '%.3e'%(i)
                p = p + s + ', '
            p = p[:-2]+']'
            self.note = [p, 
            "[p[0], p[1],...p[n]] for 'n'-order polynomial",
                        "y(x) = p[0]*x<sup>n</sup>+p[1]*x<sup>n-1</sup>+...+p[n]"]
            
        # from I_observed = Ip(E)*(I_coh + [Ip(E')/Ip(E)]I_inc)
        # find [Ip(E')/Ip(E)] ratio iteratively:
        fs = np.ones(len(xp),dtype=float) # initial relative scale assumed one
        qp = 4*np.pi/(12.3984/xp)*np.sin(np.radians(tth/2))
        xpc = xp#-2.4263e-2*(1-np.cos(np.radians(tth/2))) # E' for Compton source
        qpc = qp#4*np.pi/(12.3984/xpc)*np.sin(np.radians(tth/2)) # q' for the Compton source
        '''print('tth '+ str(tth))
        print('xp '+ str(xp[0]) + ' ... '+str(xp[-1]))
        print('xpc '+ str(xpc[0]) + ' ... '+str(xpc[-1]))
        print('xqpp '+ str(qp[0]) + ' ... '+str(qp[-1]))
        print('qpc '+ str(qpc[0]) + ' ... '+str(qpc[-1]))'''
        
        #print('sq_par primary beam '+ str (sq_par))
        mean_fqsquare,mean_fq,mean_I_inc = I_base_calc(qp,qpc,sq_par)
        
        
        # start iteration; UPDATED: iretation was replaced at some point in time by custom_fit
        #for itr in range(itr_comp):
        # re-adjust the primary beam model to fit mean_fqsqure + mean_I_inc 
        Iq_base = mean_fqsquare + fs*mean_I_inc 
        #pg.plot(Iq_base, title="Iq_base")
        # custom fit(func,x,y,p0,yb):
        ypb = np.sqrt(yp) # Poisson distribution error
        p_opt,p_stdev,p_cov,p_corp,schi2,r = \
                        custom_fit(model_func,xp,yp/Iq_base,p0,ypb/Iq_base)
        y_model = model_func(xp,*p_opt)*Iq_base
        '''I_p = model_func(xp,*p_opt)
        I_p_inc = model_func(xpc,*p_opt)
        fs = I_p_inc/I_p'''

        '''self.win = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples")
        self.win.resize(1000,600)
        self.win.setWindowTitle('pyqtgraph example: Plotting')

        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        self.p1 = self.win.addPlot(title="model_func(xp,*p_opt)", x=qp , y=model_func(xp,*p_opt))
        self.p2 = self.win.addPlot(title="Iq_base", x=qp , y=Iq_base)
        self.p2 = self.win.addPlot(title="yp", x=qp , y=yp)
        self.win.show()'''
        # propagate the mean residual error
        model_mre = np.sqrt(sum((yp/Iq_base - y_model/Iq_base))**2/len(yp))
        # Note that the mean residual error defined here is non-standard.
        # The concept of best primary beam estimation here is obtaining a function
        # that fulfils sum(y-y_model)/N ~ 0, not sum((y-y_model)**2)/N ~ 0, i.e.,
        # minimizing the sum of deviation not the deviation itself.
        # Also note that the final model_mre is for the function for the primary beam
        # estimated.
        self.out_params['primary_beam_y'] = y_model
        self.out_params['primary_beam_x'] = xp 
        self.out_params['model_func'] = model_func
        self.out_params['model_mre'] = model_mre
        self.out_params['p_opt'] = p_opt
       
class structureFactor(Calculator):
    def __init__(self):

        params = ['model_func',
                'p_opt',
                'Emin',
                'Emax',
                'polynomial_deg',
                'itr_comp',
                'sq_par',
                'model_mre',
                'sq_smoothing_factor',
                'q_spacing',
                'outputsavedirectory',
                'dataarray',
                'ttharray']
        super().__init__(params)
        self.name = 'structure factor'

    def update(self):
        """normalize individual spectrum to be S(q) in [[qi],[Sqi],[Sqi_err]] array"""

        dataarray=self.params['dataarray']
        ttharray=self.params['ttharray']
        model_func = self.params['model_func']
        p_opt = self.params['p_opt']
        Emin = self.params['Emin']
        Emax = self.params['Emax']
        polynomial_deg = self.params['polynomial_deg']
        itr_comp = self.params['itr_comp']
        sq_par = self.params['sq_par']
        model_mre = self.params['model_mre']
        sq_smoothing_factor = self.params['sq_smoothing_factor']
        q_spacing = self.params['q_spacing']

        #print(sq_par)
      
        S_q = []
        #tth_used = []
        for i in range(len(dataarray)):
            
            xi = []; yi = []; y_primary = []
            xi = dataarray[i][0]
            yi = dataarray[i][1]
            
            Emin_indx = (np.abs(xi-Emin)).argmin() # find the nearest index to Emin
            Emax_indx = (np.abs(xi-Emax)).argmin() # find the nearest index to Emax
            xn = xi[Emin_indx:Emax_indx]
            yn = yi[Emin_indx:Emax_indx]
            tth = ttharray[i]
            qi = 4*np.pi/(12.3984/xn)*np.sin(np.radians(tth/2.0))
            xnc = xn-2.4263e-2*(1-np.cos(np.radians(tth/2))) # E' for Compton source
            qic = 4*np.pi/(12.3984/xnc)*np.sin(np.radians(tth/2)) # q' for the Compton source
            '''print('tth '+ str(tth))
            print('xn '+ str(xn[0]) + ' ... '+str(xn[-1]))
            print('xpc '+ str(xnc[0]) + ' ... '+str(xnc[-1]))
            print('qi '+ str(qi[0]) + ' ... '+str(qi[-1]))
            print('qic '+ str(qic[0]) + ' ... '+str(qic[-1]))'''
            #print('sq_par sq ' + str(i)+ ' ' + str (sq_par))
            mean_fqsquare,mean_fq,mean_I_inc = I_base_calc(qi,qic,sq_par)
            y_primary = model_func(xn,*p_opt)
            Iq_base = mean_fqsquare + mean_I_inc
            #pg.plot(Iq_base, title="Iq_base "+ str(i))
            s = (Iq_base*y_primary).mean()/yn.mean()
            sqi = (s*yn-Iq_base*y_primary)/y_primary/mean_fq**2 + 1.0
            sqi_err = s*yn/y_primary/mean_fq**2*np.sqrt(1.0/yn+(model_mre/y_primary)**2)
            S_q.append([qi,sqi,sqi_err])
            #tth_used.append(tth)
            
        self.out_params['S_q_fragments'] = S_q = np.array(S_q)
        #self.out_params['tth']
        
        # find consequencial scale               
        for i in range(len(S_q)):
            if (len(S_q)-1-i) >= 1:
                data1 = S_q[len(S_q)-1-i]
                data2 = S_q[len(S_q)-2-i]
                
                q1 = np.abs(data1[0][0]-data2[0]).argmin()
                q2 = np.abs(data1[0]-data2[0][-1]).argmin()
                y1_mean = data1[1][0:q2].mean()
                y2_mean = data2[1][q1:].mean()
                s = y1_mean/y2_mean
                S_q[len(S_q)-2-i][1] = s*S_q[len(S_q)-2-i][1] # scale I(Q)
                S_q[len(S_q)-2-i][2] = s*S_q[len(S_q)-2-i][2] # scale I_err(Q)
        
        
        # respace and smooth the merged S(q) data using UnivariateSpine fucntion
        q_all = []; q_sort =[]
        S_q_all = []; sq_sort =[]
        S_q_err_all = []; sq_sort_err =[]
        
        # combine all data
        for i in range(len(S_q)):
            q_all += list(S_q[i][0][:])
            S_q_all += list(S_q[i][1][:])
            S_q_err_all += list(S_q[i][2][:])
        
        # sort in q
        sort_index = np.argsort(q_all)
        for j in sort_index:
            q_sort.append(q_all[j])
            sq_sort.append(S_q_all[j])
            sq_sort_err.append(S_q_err_all[j])
        
        q_sort = np.array(q_sort)
        sq_sort = np.array(sq_sort)
        sq_sort_err = np.array(sq_sort_err)

        q_even = np.arange(q_sort[0],q_sort[-1],q_spacing) # evenly spaced q
        
        # make evenly spaced [q,sq,sq_err] array using spline interpolation
        weight = sq_smoothing_factor/sq_sort_err

        '''pg.plot(weight, title="weight ")
        pg.plot(sq_sort_err, title="sq_sort_err ")'''

        # TODO use bins and a weighted average for all the points that fall into bins
        '''
        1. Sort the input arrays based on the x-values.
        2. Calculate the cumulative sum of the weights.
        3. Determine the bin edges for the evenly spaced bins.
        4. Compute the indices of the input data points that fall into each bin.
        5. Calculate the weighted sum of the y-values for each bin.
        6. Compute the weighted average y-values for each bin.
        7. Create evenly spaced x-values for the bins.
        8. Plot the rebinned data.
        '''

        q_rebinned, sq_rebinned = rebin_weighted(q_sort,sq_sort,weight,q_even)


        spl = interp1d(q_rebinned,sq_rebinned, kind='linear', fill_value='extrapolate' )
        #spl = interpolate.UnivariateSpline(
        #    q_rebinned,sq_rebinned,bbox=[None,None],k=3,s=None)
        
        sq_even = spl(q_even) # evenly spaced I(q)
        # estimate the root mean squre error for each spline smoothed point
        sq_even_err = []
        for i in range(len(q_even)):
            q_box_min = q_even[i]-0.5*q_spacing
            q_box_max = q_even[i]+0.5*q_spacing
            indxb = np.abs(q_box_min-q_sort).argmin()
            indxe = np.abs(q_box_max-q_sort).argmin()
            if indxe-indxb == 0:
                sq_even_err.append(sq_sort_err[indxb])
            else:
                # mean square error from the original data
                mserr = sum([err**2 for err in sq_sort_err[indxb:indxe]])/len(sq_sort_err[indxb:indxe])
                # mean square residaul for the spline fit at original data points
                msres = sum((spl(q_sort[indxb:indxe])-sq_sort[indxb:indxe])**2)/len(sq_sort[indxb:indxe])
                rmserr = np.sqrt(mserr+msres) # geometric average of the errors
                sq_even_err.append(rmserr)
        sq_even_err = np.array(sq_even_err)    
        
     

        self.out_params['q_even'] = q_even
        self.out_params['sq_even'] = sq_even
        self.out_params['sq_even_err'] = sq_even_err
        
    
    def save_structure_factor(self, filename):
        try:
            outfilename = filename
            q_even = self.out_params['q_even']
            sq_even = self.out_params['sq_even']
            sq_even_err = self.out_params['sq_even_err']
            outputsavedirectory = os.path.dirname(filename)
            self.params['outputsavedirectory'] =  outputsavedirectory
            np.savetxt(outfilename, np.transpose([q_even,sq_even,sq_even_err]),fmt='%.4e')
            S_q = self.out_params['S_q_fragments']
            for i in range(len(S_q)):
                fname = os.path.join(outputsavedirectory,'S_q_'+str("%03d" % i))
                q = S_q[i][0]
                sq =S_q[i][1]
                sq_err=S_q[i][2]
                np.savetxt(fname,np.transpose([q,sq,sq_err]),fmt='%.4e')
        except:
            print("\nThe file has not been saved!")

            
class Pdf(Calculator):
    def __init__(self):
        params = ['qmax',
                'r_spacing',
                'rmax',
                'outputsavedirectory',
                'q_even',
                'sq_even',
                'sq_even_err',
                'q_spacing']

        super().__init__(params)
        self.name = 'pdf'

    def update(self):
        """update G(r) and Inverse Fourier Filtered S(q)"""
        
        q_even = self.params['q_even']
        sq_even = self.params['sq_even']
        sq_even_err = self.params['sq_even_err']
        q_spacing = self.params['q_spacing']
        
        qmax = self.params['qmax']
        r_spacing = self.params['r_spacing']
        rmax = self.params['rmax']
        
        

        # restrict the qmax by the user input
        q_max_indx = np.abs(q_even - qmax).argmin()
        self.out_params['qq'] = qcalc = q_even[:q_max_indx]
        self.out_params['sq'] = sqcalc = sq_even[:q_max_indx]
        self.out_params['sq_err'] = sqcalc_err = sq_even_err[:q_max_indx]
        
        

        gr = []; gr_err = []
        r = np.arange(r_spacing,rmax,r_spacing) 
        del_r = np.pi/qcalc[-1]
        for i in range(len(r)):
            gri = 2.0/np.pi/r[i]*sum(
                qcalc*r[i]*(sqcalc-1.0)*np.sin(qcalc*r[i])*q_spacing*\
                np.sin(qcalc*del_r)/(qcalc*del_r))
            gr.append(gri)
            gr_erri = 2.0/np.pi/r[i]*\
                np.sqrt(sum((qcalc*r[i]*np.sin(qcalc*r[i])*q_spacing*\
                             np.sin(qcalc*del_r)/(qcalc*del_r)*sqcalc_err)**2))
            gr_err.append(gr_erri)
            
        gr = np.array(gr)
        gr_err = np.array(gr_err)
        
        #r_save = r
        self.out_params['r'] = copy.copy(r)
        #gr_save = gr
        self.out_params['gr'] = copy.copy(gr)
        #gr_err_save = gr_err
        self.out_params['gr_err'] = copy.copy(gr_err)
        
        

    def save_pdf(self, filename):
        try:
            outfilename = filename
            r_save=self.out_params['r']
            gr_save=self.out_params['gr']
            gr_err_save=self.out_params['gr_err']
            outputsavedirectory = os.path.dirname(filename)
            self.params['outputsavedirectory'] =  outputsavedirectory

            np.savetxt(outfilename, np.transpose([r_save,gr_save,gr_err_save]),fmt='%.4e')
        except Exception as e:
            print(e)
            print("\nThe file has not been saved!")

class PdfInverse(Calculator):     
    # inverse Fourier filter for the final S(q)
    # re-calc full G(r) without broadening
    #        gr = []; gr_err = []
    #        r = np.arange(r_spacing,np.pi/q_spacing,r_spacing) # true r max window
    #        for i in range(len(r)):
    #            gri = 2.0/np.pi/r[i]*sum(
    #                qcalc*r[i]*(sqcalc-1.0)*np.sin(qcalc*r[i])*q_spacing)
    #            gr.append(gri)
    #            gr_erri = 2.0/np.pi/r[i]*\
    #                np.sqrt(sum((qcalc*r[i]*np.sin(qcalc*r[i])*q_spacing*sqcalc_err)**2))
    #            gr_err.append(gr_erri)

    def __init__(self):
        params_sq = ['model_func',
                'p_opt',
                'Emin',
                'Emax',
                'polynomial_deg',
                'itr_comp',
                'sq_par',
                'model_mre',
                'sq_smoothing_factor',
                'q_spacing',
                'outputsavedirectory',
                'dataarray',
                'ttharray']

        out_params_sq = ['S_q_fragments',
                        'q_even','sq_even',
                        'sq_even_err'] 

        params_gr = ['qmax',
                'r_spacing',
                'rmax',
                'outputsavedirectory',
                'q_even',
                'sq_even',
                'sq_even_err',
                'q_spacing']

        out_params_gr = ['qq',
                        'sq',
                        'sq_err',
                        'r',
                        'gr',
                        'gr_err']

        params = ['r',
                'gr',
                'gr_err',
                'qmax',
                'r_spacing',
                'rmax',
                'rho',
                'hard_core_limit',
                'qq',
                'sq',
                'sq_err',
                'outputsavedirectory']

        out_params_f = ['r_f',
                        'gr_f',
                        'gr_err_f',
                        'q_inv',
                        'sq_inv',
                        'sq_inv_err']

        super().__init__(params)
        self.name = 'inverse'

    def update(self):
        """update Inverse Fourier Filtered S(q)"""
        
        qmax = self.params['qmax']
        r_spacing = self.params['r_spacing']
        rmax = self.params['rmax']
        r_min_cut = self.params['hard_core_limit']
        rho = self.params['rho']
        r = copy.deepcopy(self.params['r'])
        gr = copy.deepcopy(self.params['gr'])
        gr_err = copy.deepcopy(self.params['gr_err'])
        qcalc = self.params['qq']
        sqcalc = self.params['sq']
        sqcalc_err = self.params['sq_err']
            
        r_min_cut_indx = np.abs(r-r_min_cut).argmin()
        if rho == None:
            gr[0:r_min_cut_indx] = r[0:r_min_cut_indx]*gr[r_min_cut_indx]/r_min_cut
            gr_err[0:r_min_cut_indx] = r[0:r_min_cut_indx]*gr_err[r_min_cut_indx]/r_min_cut
        else:
            gr[0:r_min_cut_indx] = -4.0*np.pi*rho*r[0:r_min_cut_indx]
            gr_err[0:r_min_cut_indx] = 4.0*np.pi*rho*r[0:r_min_cut_indx]*gr_err[r_min_cut_indx]
        gr = np.array(gr)

        self.out_params['r_f'] = r
        self.out_params['gr_f'] = gr
        self.out_params['gr_err_f']= gr_err
        
        
    

        sq_inv = []; sq_inv_err = []

        self.out_params['q_inv'] = q_inv = qcalc # update only up to the initially given q_calc

        
        
        for i in range(len(q_inv)):
            sq_invi = sum(gr*np.sin(q_inv[i]*r)/q_inv[i]*r_spacing)
            sq_inv.append(sq_invi)  
                                    # sq_inv_erri = np.sqrt(sum(
                                    # (gr_err*np.sin(q_inv[i]*r)/q_inv[i]*r_spacing)**2))
                                    # sq_inv_err.append(sq_inv_erri)
        self.out_params['sq_inv'] = sq_inv = 1.0+np.array(sq_inv)
        self.out_params['sq_inv_err'] = sq_inv_err = sqcalc_err # enforce original errors

        
        

    def save_sf_inverse(self, outfilename):
        try:
            q_inv=self.out_params['q_inv']
            sq_inv=self.out_params['sq_inv']
            sq_inv_err=self.out_params['sq_inv_err']
            outputsavedirectory = os.path.dirname(outfilename)
            self.params['outputsavedirectory'] =  outputsavedirectory
            np.savetxt(outfilename, np.transpose([q_inv,sq_inv,sq_inv_err]),fmt='%.4e')
        except Exception as e:
            print(e)
            print("\nThe file has not been saved!")
       