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

import gsas.GSASIIscriptable as G2sc
from utilities.HelperModule import convert_d_to_two_theta, calculate_color
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import QObject, pyqtSignal, Qt
from PyQt6 import QtWidgets

class Cell():
    def __init__(self, name, color, use=True ):
        print('init Cell')
        self.use = use
        self.color = color
        self.name = name
        self.params = {}
        self.params['symmetry'] = 'triclinic'
        self.params['spacegroup'] = 'P 1'
        self.params['a'] = 3.
        self.params['b'] = 3.
        self.params['c'] = 3.
        self.params['alpha'] = 90.
        self.params['beta'] = 90.
        self.params['gamma'] = 90.
        #self.params['zero'] = 0.
        #self.params['conditions'] = []
        self.min_d = 1
        self.symmetries = ('CUBIC',
                                'TETRAGONAL',
                                'ORTHORHOMBIC',
                                'HEXAGONAL',
                                "TRIGONAL",
                                'RHOMBOHEDRAL',
                                'MONOCLINIC',
                                'TRICLINIC')
        self.default_spacegroup = {'CUBIC':'P 2 3',
                                'TETRAGONAL': 'P 4',
                                'ORTHORHOMBIC': 'P 2 2 2',
                                'HEXAGONAL': 'P 6',
                                "TRIGONAL": 'P 3',
                                'RHOMBOHEDRAL': 'R 3 r',
                                'MONOCLINIC':'P 2',
                                'TRICLINIC':'P 1'}

    def set_symmetry(self, symmetry):
        symmetry = symmetry.upper()
        if not symmetry in self.symmetries:
            print('Unknown symmetry: {0}.'.format(symmetry))
            return
        self.params['symmetry'] = symmetry
        self.enforce_symmetry()

    def enforce_symmetry(self):
        if self.params['symmetry'] == 'CUBIC':
            self.params['b'] = self.params['a']
            self.params['c'] = self.params['a']
            self.params['alpha'] = 90.
            self.params['beta'] = 90.
            self.params['gamma'] = 90.
        elif self.params['symmetry'] == 'TETRAGONAL':
            self.params['b'] = self.params['a']
            self.params['alpha'] = 90.
            self.params['beta'] = 90.
            self.params['gamma'] = 90.
        elif self.params['symmetry'] == 'ORTHORHOMBIC':
            self.params['alpha'] = 90.
            self.params['beta'] = 90.
            self.params['gamma'] = 90.
        elif self.params['symmetry'] == 'HEXAGONAL' or self.params['symmetry'] == "TRIGONAL":
            self.params['b'] = self.params['a']
            self.params['alpha'] = 90.
            self.params['beta'] = 90.
            self.params['gamma'] = 120.
        elif self.params['symmetry'] == 'RHOMBOHEDRAL':
            self.params['b'] = self.params['a']
            self.params['c'] = self.params['a']
            self.params['beta'] = self.params['alpha']
            self.params['gamma'] = self.params['alpha']
        elif self.params['symmetry'] == 'MONOCLINIC':
            self.params['alpha'] = 90.
            self.params['gamma'] = 90.
        elif self.params['symmetry'] == 'TRICLINIC':
            pass

    def set_lattice_param(self, param):
        for key in param:
            self.params[key] = param[key]
        self.enforce_symmetry()

    def set_spacegroup(self, spacegroup):
        if spacegroup == '':
            spacegroup = 'P 1'
        s = G2sc.G2spc.SpcGroup(spacegroup)
        d = s[-1]
        if not len(d):
            return
        symmetry = d['SGSys']
        self.params['spacegroup'] = spacegroup
        self.set_symmetry(symmetry)

    def get_symmetry(self):
        return self.params['symmetry']

    def get_spacegroup(self):
        return self.params['spacegroup']

    def get_hkl_reflections(self):
        tth = convert_d_to_two_theta(self.min_d,0.5)
        a = self.params['a']
        b = self.params['b']
        c = self.params['c']
        alpha = self.params['alpha']
        beta = self.params['beta']
        gamma = self.params['gamma']
        spacegroup = self.params['spacegroup']
        refs = G2sc.GenerateReflections(spacegroup, 
            (a,b,c,alpha,beta,gamma), 
            TTmax=tth,wave=.5) 
        refs_P1 = G2sc.GenerateReflections('P 1', 
            (a,b,c,alpha,beta,gamma), 
            TTmax=tth,wave=.5) 
        return refs, refs_P1

    def get_lattice_params(self):
        return self.params

    def get_params_enabled(self):
        symmetry = self.params['symmetry'].upper()
        enabled = {}
        if symmetry == 'CUBIC':
            enabled['a']=True
            enabled['b']=False
            enabled['c']=False
            enabled['alpha']=False
            enabled['beta']=False
            enabled['gamma']=False
        elif symmetry == 'TETRAGONAL':
            enabled['a']=True
            enabled['b']=False
            enabled['c']=True
            enabled['alpha']=False
            enabled['beta']=False
            enabled['gamma']=False
        elif symmetry == 'ORTHORHOMBIC':
            enabled['a']=True
            enabled['b']=True
            enabled['c']=True
            enabled['alpha']=False
            enabled['beta']=False
            enabled['gamma']=False
        elif symmetry == 'HEXAGONAL' or symmetry == 'TRIGONAL':
            enabled['a']=True
            enabled['b']=False
            enabled['c']=True
            enabled['alpha']=False
            enabled['beta']=False
            enabled['gamma']=False
        elif symmetry == 'RHOMBOHEDRAL':
            enabled['a']=True
            enabled['b']=False
            enabled['c']=False
            enabled['alpha']=True
            enabled['beta']=False
            enabled['gamma']=False
        elif symmetry == 'MONOCLINIC':
            enabled['a']=True
            enabled['b']=True
            enabled['c']=True
            enabled['alpha']=False
            enabled['beta']=True
            enabled['gamma']=False
        elif symmetry == 'TRICLINIC':
            enabled['a']=True
            enabled['b']=True
            enabled['c']=True
            enabled['alpha']=True
            enabled['beta']=True
            enabled['gamma']=True
        else:
            print('Unknown symmetry: {0}.'.format(symmetry))
            return
        return enabled

class hklGenModel_viewModel(QStandardItemModel):

    phase_added = pyqtSignal()
    phase_removed = pyqtSignal()

    phase_changed = pyqtSignal()
    phase_changed = pyqtSignal()
    phase_changed = pyqtSignal()
    phase_changed = pyqtSignal()

    reflection_added = pyqtSignal()
    reflection_deleted = pyqtSignal()

    def __init__(self, parent=None):
        header = [ 'Use', 'Color', 'Cell']
        super().__init__(0, len(header), parent)
        for ind, h in enumerate(header):
            self.setHeaderData(ind, Qt.Horizontal, h)
        self.itemChanged.connect(self.on_item_changed)
        print('init hklGenModel')
        self.cells = []

    def removeRow(self, ind):
        del self.cells[ind]
        return super().removeRow(ind)

    def on_item_changed(self, item):
        
        col = item.column()
        row = item.row()
        if col == 0 :
            data  = self.item(row, col).checkState()
            use = data == 2
            self.cells[row].use = use
        elif col == 1 :
            color = self.data(self.index(row, col))
            self.cells[row].color = color
        elif col == 2 :
            name = self.data(self.index(row, col))
            self.cells[row].name = name

    def set_cell_name(self,ind, name):
        self.block_update = True
        self.cells[ind].name = name
        self.setData(self.index(ind, 2), name)
        self.block_update = False

    def add_cell(self , cell=None):
        if cell is None:
            color  = calculate_color(len(self.cells)-1) 
            c = (int(color[0]), int(color[1]),int(color[2]))
            c_str = '#%02x%02x%02x' % c
            name = 'cell '+str(len(self.cells))
            new_cell = Cell(name, c_str, use=True)
            cell = new_cell
        else:
            c_str = cell.color
            name = cell.name
            cell.use = True
        checked, text = (Qt.Checked, '')
        check_item = QStandardItem(text)
        check_item.setCheckable(True)
        check_item.setCheckState(checked)

        self.cells.append(cell)
        
        parent_item = self.invisibleRootItem()  # type: QStandardItem
        parent_item.appendRow([check_item, QStandardItem(c_str), QStandardItem(name)])
        
        return cell

    def remove_cell_by_index(self, ind):
        self.removeRow(ind)

    def get_cell_by_index(self, ind):
        return self.cells[ind]

    def get_num_of_cells(self):
        return len(self.cells)

    def get_cell_index(self, cell):
        return self.cells.index(cell)
    
    def set_data(self, cells):
        for cell in cells:
            self.add_cell(cell)

    def get_data(self ):
        return self.cells
    
def make_jcpds(self, cell):
    return 'filename.jcpds'
