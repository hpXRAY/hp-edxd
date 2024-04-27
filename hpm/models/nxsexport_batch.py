# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:34:27 2020

@author: farlarob

Updates:
RH: remove the TK stuff and wrap into a function 2024-04-27
"""

import h5py
import numpy as np

import os


def read_nxs(fname, Channel='both'):

    out = {}

    if Channel == '0':
        
        f = h5py.File(fname, 'r')
        
        try:
            dset = f['/entry/instrument/xspress3/channel00/histogram']
        except KeyError:
            print("Error","Channel does not exist")
            return out
                    
        dset_transpose = np.transpose(dset)

        dset_corr = np.delete(dset_transpose, 4095, 0)
        
        dset_corr2 = np.sum(dset_corr, axis = 1)

        out[0] = dset_corr2
            
        #num_rows = dset_corr2.shape
        #print(num_rows)
        
        '''new_column = np.arange(0,4095)
            
        dset_fix = np.column_stack((new_column,dset_corr2))
                
        output = os.path.basename(fname)
        pre, ext = os.path.splitext(fname)
        
        output = pre + '_Ch0' + '.csv'
        
        np.savetxt(output, dset_fix, delimiter=",")'''
            
        print("Conversion", "The nxs to csv conversion is complete")    
        return out    

    if Channel == '1':
    
        f = h5py.File(fname, 'r')
        
        try:
            dset = f['/entry/instrument/xspress3/channel01/histogram']
        except KeyError:
            print("Error","Channel does not exist")
            return out
            
        dset_transpose = np.transpose(dset)

        dset_corr = np.delete(dset_transpose, 4095, 0)
        
        dset_corr2 = np.sum(dset_corr, axis = 1)

        out[1] = dset_corr2
            
        #num_rows = dset_corr2.shape
        #print(num_rows)
        
        '''new_column = np.arange(0,4095)
            
        dset_fix = np.column_stack((new_column,dset_corr2))
                
        output = os.path.basename(fname)
        pre, ext = os.path.splitext(fname)
        
        output = pre + '_Ch1' + '.csv'
        
        np.savetxt(output, dset_fix, delimiter=",")'''

        print("Conversion", "The nxs to csv conversion is complete")      
        return out    
            
    if Channel == 'both':
    
        
        f = h5py.File(fname, 'r')
        
        try: 
            dset = f['/entry/instrument/xspress3/channel00/histogram']
        except KeyError:
            print("Error","Channel does not exist")
            return out
            
        dset_transpose = np.transpose(dset)
        
        dset_corr = np.delete(dset_transpose, 4095, 0)
            
        dset_corr2 = np.sum(dset_corr, axis = 1)
        out[0] = dset_corr2
                
        #num_rows = dset_corr2.shape
        #print(num_rows)
            
        '''new_column = np.arange(0,4095)
                
        dset_fix = np.column_stack((new_column,dset_corr2))
                    
        output = os.path.basename(fname)
        pre, ext = os.path.splitext(fname)
            
        output = pre + '_Ch0' + '.csv'
            
        np.savetxt(output, dset_fix, delimiter=",")   '''  
                
        

        f = h5py.File(fname, 'r')
        
        try: 
            dset = f['/entry/instrument/xspress3/channel01/histogram']
        except KeyError:
            print("Error","Channel does not exist")
            return out
                
        dset_transpose = np.transpose(dset)

        dset_corr = np.delete(dset_transpose, 4095, 0)
        
        dset_corr2 = np.sum(dset_corr, axis = 1)

        out[1] = dset_corr2
            
        #num_rows = dset_corr2.shape
        #print(num_rows)
        
        '''new_column = np.arange(0,4095)
            
        dset_fix = np.column_stack((new_column,dset_corr2))
                
        output = os.path.basename(fname)
        pre, ext = os.path.splitext(fname)
        
        output = pre + '_Ch1' + '.csv'
        
        np.savetxt(output, dset_fix, delimiter=",")'''
            
        print("Conversion", "The nxs to csv conversion is complete")  
        return out