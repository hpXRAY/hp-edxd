

import  time, logging, os, struct, sys, copy

from pyqtgraph.Qt import QtCore
from PyQt6.QtCore import QThread, pyqtSignal, QObject
import queue 
from functools import partial
import json
from epics import PV, caget, caput
from mypyeqt.epicsMonitor import epicsMonitor

class pvServer():
    pvs = {}
    pv_monitors = {}
    
   
    def set_pv(self, pv_name, pv):
        self.pvs[pv_name] = pv
        monitor = epicsMonitor(pv, autostart=True)  
        self.pv_monitors[pv_name] = monitor

    def get_pv(self, pv_name):
        #print(pv_name)
        if pv_name in self.pvs:
            pv = self.pvs[pv_name]
        else:
            pv = PV(pv_name)
            #print(pv.info)
            self.set_pv(pv_name, pv)
        
        return pv

    def get_pv_monitor(self, pv_name):
        if pv_name in self.pvs and pv_name in self.pv_monitors:
            monitor = self.pv_monitors[pv_name]
        else:
            pv = PV(pv_name)
            
            self.set_pv(pv_name, pv)
            monitor = self.pv_monitors[pv_name]
        return monitor


