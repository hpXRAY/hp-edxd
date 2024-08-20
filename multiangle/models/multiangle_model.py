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



from PyQt6.QtCore import  Qt
from PyQt6.QtGui import QStandardItemModel

from multiangle.models.colimation_depth import get_collimation_depth

class multiangleModel(QStandardItemModel):
    
 
    display_order = {'ind':0, 
            'tth':1, 
            'stime':2, 
            'pvsize':3, 
            'phsize':4, 
            'svsize':5, 
            'shsize':6, 
            'dvsize':7, 
            'dhsize':8,
            'd0':9,
            'd1':10}
    data_order =['tth', 
            'pvsize', 
            'phsize', 
            'svsize', 
            'shsize', 
            'dvsize', 
            'dhsize', 
            'stime']
    header = [ '       #', 
            f'2\N{GREEK SMALL LETTER THETA}', 
            'Exp. (s)',
            '1st V',  
            '1st H', 
            '2nd V',
            '2nd H',
            'Det. V', 
            'Det. H',
            'D0',
            'D1'
             ]
    
    def __init__(self, parent):
        header = self.header
        super().__init__(0, len(header), parent)
        for ind, h in enumerate(header):
            self.setHeaderData(ind, Qt.Orientation.Horizontal, h)
        self.itemChanged.connect(self.on_item_changed)
        self.det_slit_distance = 600.
        self.tip_slit_distance = 60.
        self.tip_slit_size = 0.05
        self.block_update = False

    def recalc_collimation(self):
        rows = self.rowCount() 
        for r in range(rows):
            self.recalc_collimation_by_row(r)

    def recalc_collimation_by_row(self, row):
        order = self.data_order
        d = {}
        for key in order:
            c = self.display_order[key]
            d[key]= self.data(self.index(row, c))

        cd = self.calc_collimation_depth(d)
        self.block_update = True
        self.setData(self.index(row, self.display_order['d0']), cd['d0'])
        self.setData(self.index(row, self.display_order['d1']), cd['d1'])
        self.block_update = False

    def on_item_changed(self, item):
        if not self.block_update:
            row = item.row()
            self.recalc_collimation_by_row(row)

    def calc_collimation_depth(self, d):
        beam_params = { 'det_slit_distance':self.det_slit_distance, 
                        'tip_slit_distance':self.tip_slit_distance,
                        'tip_slit_size'    :self.tip_slit_size}
        params = {**d, **beam_params}
        cd = get_collimation_depth(params)
        return cd

    def add_row(self, d):
        self.block_update = True
        rows = self.rowCount() 
        self.insertRow(rows)
        
        cd = self.calc_collimation_depth(d)
        d = {**d,**cd}
        for i in d:
            self.setData(self.index(rows, self.display_order[i]), d[i])
        self.sort(1, Qt.SortOrder.AscendingOrder)
        rows = self.rowCount() 
        for r in range(rows):
            ind = r+1
            self.setData(self.index(r, self.display_order['ind']), ind)
            self.item(r, self.display_order['ind']).setEditable(False)
            self.item(r, self.display_order['d0']).setEditable(False)
            self.item(r, self.display_order['d1']).setEditable(False)
        self.block_update = False
        

    def set_data(self, data):
        order = self.data_order
        for d in data:
            row = {}
            for ind, o in enumerate(order):
                row[o]=d[ind]
            self.add_row(row)

    def get_data(self ):
        order = self.data_order
        cols = []
        for o in order:
            col = self.display_order[o]
            cols.append(col)
        rows = self.rowCount() 
        tth_set= []
        for r in range(rows):
            tth = []
            for c in cols:
                data = self.data(self.index(r, c))
                tth.append(data)
            tth_set.append(tth)
        return tth_set
    
    def get_tth(self ):
        rows = self.rowCount()
        tth = []
        for r in range(rows):
            t = self.data(self.index(r, self.display_order['tth']))
            tth.append(t)
        return tth