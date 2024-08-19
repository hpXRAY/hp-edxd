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


from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import pyqtSignal, QByteArray, QDataStream, QIODevice, Qt
from PyQt6.QtGui import QDrag
from  PyQt6.QtWidgets import QMainWindow, QApplication, QInputDialog, QWidget, QLabel, QTreeWidget, QTreeWidgetItem, QAbstractItemView
from hpm.widgets.CustomWidgets import FlatButton, DoubleSpinBoxAlignRight, VerticalSpacerItem, NoRectDelegate, \
    HorizontalSpacerItem, ListTableWidget, VerticalLine, DoubleMultiplySpinBoxAlignRight, HorizontalLine
from hpm.widgets.PltWidget import plotWindow
from functools import partial
import copy



class aEDXDFilesWidget(QWidget):
    

    
    widget_closed = pyqtSignal()
    
    
    delete_clicked_signal = pyqtSignal(dict)

    

    

    def __init__(self):
        super().__init__()
        self.setupUi()
   
    def raise_widget(self):
        self.show()
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowState.WindowMinimized | QtCore.Qt.WindowState.WindowActive)
        self.activateWindow()
        self.raise_()    
    
    def setupUi(self):
        self._layout = QtWidgets.QVBoxLayout()
        #self._layout.setSpacing(5)
        self.setWindowTitle('EDXD Files Input Control')
        self.button_widget = QtWidgets.QWidget(self)
        self.button_widget.setObjectName('files_control_button_widget')
        self._button_layout = QtWidgets.QHBoxLayout()
        self._button_grid_layout = QtWidgets.QGridLayout()
        self._button_layout.setContentsMargins(0, 0, 0, 0)
        #self._button_layout.set
        self.add_tth_btn = FlatButton('+ '+f'2\N{GREEK SMALL LETTER THETA}')
        self._button_grid_layout.addWidget(self.add_tth_btn, 0,1)
        self.add_btn = FlatButton('+ EDXD')
        self._button_grid_layout.addWidget(self.add_btn,1,1)
        self.add_all_btn = FlatButton('+ All EDXD')
        self._button_grid_layout.addWidget(self.add_all_btn,1,0)
        self.add_scan = FlatButton(f'+ All 2\N{GREEK SMALL LETTER THETA}')
        self._button_grid_layout.addWidget(self.add_scan,0,0)
        self.del_btn = FlatButton('Delete')
        self._button_grid_layout.addWidget(self.del_btn,0,2)
        self.clear_btn = FlatButton('Clear')
        self._button_grid_layout.addWidget(self.clear_btn,0,3)
        self.expand_btn = FlatButton('Expand')
        self._button_grid_layout.addWidget(self.expand_btn,1,2)
        self.collapse_btn = FlatButton('Collapse')
        self._button_grid_layout.addWidget(self.collapse_btn,1,3)
        self._button_layout.addLayout(self._button_grid_layout)
        self._button_layout.addSpacerItem(HorizontalSpacerItem())
        #self.from_config_btn = QtWidgets.QPushButton('From config.')
        #self._button_layout.addWidget(self.from_config_btn,0)
        #self._button_layout.addWidget(VerticalLine())
        #self._button_layout.addSpacerItem(HorizontalSpacerItem())
        #self._button_layout.addWidget(VerticalLine())
        self.button_widget.setLayout(self._button_layout)
        self._body_layout = QtWidgets.QHBoxLayout()
        '''self.file_tw = ListTableWidget(columns=4)
        header_view = QtWidgets.QHeaderView(QtCore.Qt.Orientation.Horizontal, self.file_tw)
        self.file_tw.setHorizontalHeader(header_view)
        header_view.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)'''
        self.file_trw = treeWidget()
        
        #self.file_trw.setHeaderLabels([' ' + f'  2\N{GREEK SMALL LETTER THETA}'])
        self.file_trw.setHeaderHidden(True)
        #self.file_trw.setAlternatingRowColors(True)
        #self.file_trw.setItemDelegate(NoRectDelegate())
        
        header = self.file_trw.header()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Fixed)
        self._body_layout.addWidget(self.file_trw )
        
        self._layout.addWidget(self.button_widget)
        self._layout.addLayout(self._body_layout)

        
        self.button_2_widget = QtWidgets.QWidget(self)
        self.button_2_widget.setObjectName('files_control_button_2_widget')
        self._button_2_layout = QtWidgets.QHBoxLayout()
        self._button_2_layout.setContentsMargins(0, 0, 0, 0)
        self._button_2_layout.setSpacing(6)
        self.apply_btn = FlatButton('Apply')
        self._button_2_layout.addWidget(self.apply_btn,0)
        #self._button_2_layout.addWidget(VerticalLine())
        self._button_2_layout.addSpacerItem(HorizontalSpacerItem())
        #self._button_2_layout.addWidget(VerticalLine())
        self.button_2_widget.setLayout(self._button_2_layout)
        self._layout.addWidget(HorizontalLine())
        self._layout.addWidget(self.button_2_widget)
        

        

        self.setLayout(self._layout)
        self.retranslateUi(self)
        self.style_widgets()

        self.del_btn.clicked.connect(self.delete_clicked)

    


    def delete_clicked(self):

        trw = self.file_trw
        if len(trw.top_level_items):
            si=trw.selectedItems()
            if len(si):
                item = si[0]
                ind, files = trw.identify_item(item)
                self.delete_clicked_signal.emit({'ind':ind, 'files':files, 'item':item})

   

    def remove_trw_item(self, item):
        trw = self.file_trw
        root = trw.invisibleRootItem()
        (item.parent() or root).removeChild(item)

    
    # ###############################################################################################
    # Now comes all the file tw stuff
    ################################################################################################

    

    def style_widgets(self):
        #self.file_trw.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        #self.file_trw.setMinimumWidth(405)
        
        #self.file_trw.setMinimumHeight(120)
        self.setStyleSheet("""
            
            #files_control_button_widget FlatButton {
                min-width: 70;
                max-width: 70;
            }
            #files_control_button_2_widget FlatButton {
                min-width: 70;
                max-width: 70;
            }
        """)

  
    def retranslateUi(self, aEDXDWidget):
        pass




class treeWidget(QtWidgets.QTreeWidget):
    customMimeType = "application/x-customTreeWidgetdata"
    cb_state_changed_signal = pyqtSignal(list, list, bool)
    file_selection_changed_signal = pyqtSignal(list, list)
    color_btn_clicked = pyqtSignal(int, QtWidgets.QWidget)
    top_level_selection_changed_signal = pyqtSignal(int,float)
    drag_drop_signal = pyqtSignal(dict)
    files_dragged_in = pyqtSignal(dict)
    

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True) 
        self.expandAll()
        #self.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.setDragEnabled(True)
        #self.viewport().setAcceptDrops(True)
 
        #self.setDropIndicatorShown(True)

        self.clear_arrays()

        self.currentItemChanged.connect(self.file_selection_changed)
        

    def clear_arrays(self):
        self.top_level_color_btns = []
        self.top_level_items = []
        self.file_items = []
        self.file_show_cbs = []
        self.tth_items = []
        self.files = {}


    def file_cb_changed(self, checkbox):
        i = self.file_show_cbs.index(checkbox)
        file_item=self.file_items[i]
        ind, files = self.identify_item(file_item)
        self.cb_state_changed_signal.emit(ind, files, checkbox.isChecked())


    def file_selection_changed(self, current, previous):
        if len(self.top_level_items):
            ind,files = self.identify_item(current)

            if len(ind)==1:
                self.top_level_selection_changed_signal.emit(ind[0],files[0])
            if len(ind)==2:
                self.file_selection_changed_signal.emit(ind,files)

    def rename_file(self, ind, name):
        name_item = self.item(ind, 2)
        name_item.setText(name)
    
    def set_file_tth(self, ind, P):
        tth_item = self.item(ind, 3)
        try:
            tth_item.setText("{0:.2f} GPa".format(P))
        except ValueError:
            tth_item.setText("{0} GPa".format(P))

    def get_file_tth(self, ind):
        tth_item = self.item(ind, 3)
        tth = float(str(tth_item.text()).split()[0])
        return tth

    def change_tth_color(self, ind, color):
        self.top_level_color_btns[ind].setStyleSheet('background-color:' + color.name())
    
    

    def Tth_edit_finished(self, tth_lineEdit):
        print(tth_lineEdit.text())

    


    def get_selected_file_row(self):

        selected = self.selectionModel().selectedRows()
        try:
            row = selected[0].row()
        except IndexError:
            row = -1
        return row

    def select_file(self, ind):
        self.setCurrentIndex(ind)

    def file_show_cb_set_checked(self, ind, state):
        checkbox = self.file_show_cbs[ind]
        checkbox.setChecked(state)

    def file_show_cb_is_checked(self, ind):
        checkbox = self.file_show_cbs[ind]
        return checkbox.isChecked()

    def clear_trw(self):
        self.blockSignals(True)
        root = self.invisibleRootItem()
        for item in self.top_level_items:
            (item.parent() or root).removeChild(item)
        self.blockSignals(False)
        self.clear_arrays()

    def del_file(self, file_item ):

        self.blockSignals(True)
        ind, files = self.identify_item(file_item)
        tth = files[0]
        filename = files[1]
        i = self.file_items.index(file_item)
        del self.file_items[i]
        del self.file_show_cbs[i]
        for t in self.files:
            if str(tth) == t:
                for i, f in enumerate(self.files[t]):
                    if filename == f:
                        del self.files[t][i]
        file_item.parent() .removeChild(file_item)
        self.blockSignals(False)

    def del_group(self, top_item ):

        ind, files = self.identify_item(top_item)
        tth = files[0]
        i = self.top_level_items.index(top_item)
        root = self.invisibleRootItem()
        for item in self.top_level_items:
            if top_item==item:
                del self.top_level_color_btns[i]
                del self.top_level_items[i]
                del self.tth_items[i]
                if str(tth) in self.files:
                    del self.files[str(tth)]
                (item.parent() or root).removeChild(item)
                break

    def get_selected_tth(self):

        trw = self
        if len(self.top_level_items):
            si=trw.selectedItems()
            if len(si):
                item = si[0]
                ind, files = self.identify_item(item)
                tth = files[0]
                return tth
            else :
                return None
        else :
            return None

    def identify_item(self,item:QtWidgets.QTreeWidgetItem):
        #txt = str(item)
        #print(txt)
        ind=[0]
        if item in self.top_level_items:
            p_ind = self.top_level_items.index(item)
            ind = [p_ind]
            #print('item in self.top_level_items' + str(ind))
        if item in self.file_items:
            p = item.parent()
            c_ind = p.indexOfChild(item)
            p_ind = self.top_level_items.index(p)
            ind = [p_ind,c_ind]
            #print('item in self.file_items' + str(ind))
        tth = float(self.tth_items[ind[0]].tthItem.text()) 
        #print('tth ' + str(tth))
        files = [tth]
        #print('files ' + str(files))
        if len(ind)==2:  
            filename = self.files[str(tth)][ind[1]]
            files.append(filename)
            #print('len(ind)== 2')
            #print('files '+ str(files))
        return ind, files

    def add_file_group(self, filenames, color, tth,use):
        self.blockSignals(True)
        current_rows = self.currentIndex()
        h = QtWidgets.QTreeWidgetItem()
        h.setExpanded(True)
        self.top_level_items.append(h)
        self.addTopLevelItem(h)

        tth_item = TthItem(color, tth)
        tth_item.file_dragged_in.connect(self.tth_item_file_dragged_in_callback)
        tth_item.file_dragged_in_from_treewidget.connect(self.myDropEvent)
        tth_item.color_button.clicked.connect(partial(self.color_btn_click, tth_item.color_button))
        
       
        self.top_level_color_btns.append(tth_item.color_button)
        self.setItemWidget(h,0,tth_item)
        
        
        #self.setItemWidget(h,1,tth_item)
        self.tth_items.append(tth_item)
        for i, filename in enumerate(filenames):
            file_use = use[i]
            self.add_file(filename,file_use,tth)
        self.setColumnWidth(0, 60)
        self.select_file(current_rows)
        self.blockSignals(False)

    def color_btn_click(self, button):
        self.color_btn_clicked.emit(self.top_level_color_btns.index(button), button)    

    def add_file(self, filename,file_use,tth):
        tths = []
        for tth_i in self.tth_items:
            tths.append(float(tth_i.tthItem.text()))
        ind = tths.index(float(tth))
        top_level_item = self.top_level_items[ind]

        file_item = QtWidgets.QTreeWidgetItem(top_level_item)
        self.file_items.append(file_item)

        file_label = FileItem(file_use, filename)

        
        file_label.show_cb.stateChanged.connect(partial(self.file_cb_changed, file_label.show_cb))
        
        self.setItemWidget(file_item, 0, file_label)
        self.file_show_cbs.append(file_label.show_cb)
    

        if tth in self.files:
            self.files[tth].append(filename)
        else:
            self.files[tth]=[filename]

        self.expandAll()

    

    def mimeTypes(self):
        mimetypes = QTreeWidget.mimeTypes(self)
        mimetypes.append(self.customMimeType)
        return mimetypes

    

    def startDrag(self, supportedActions):
        drag = QDrag(self)

        si = self.selectedItems()
        item = si[0]
        ind, files = self.identify_item(item)
        #print(ind)
        #print (files)
        if len(ind)>1:
            mimedata = self.model().mimeData(self.selectedIndexes())

            encoded = QByteArray()
            stream = QDataStream(encoded, QIODevice.WriteOnly)
            self.encodeData(self.selectedItems(), stream)
            mimedata.setData(self.customMimeType, encoded)

            drag.setMimeData(mimedata)
            drag.exec_(supportedActions)

    def myDropEvent(self, e):
        
        for key in e:
            tth_target = float(key)
            event = e[key]
        if event.mimeData().hasFormat(self.customMimeType):
            encoded = event.mimeData().data(self.customMimeType)
            parent = self.itemAt(event.pos())
            decoded_ind = self.decodeData(encoded, event.source())
            tth = float(self.tth_items[decoded_ind[0]].tthItem.text()) 
            filename = self.files[str(tth)][decoded_ind[1]]

            ind, files = self.identify_item(parent)
            
            file_item = self.file_items[decoded_ind[2]]
            
            param = {}
            param['ind'] = [decoded_ind[0], decoded_ind[1] ]
            param['files'] = [tth, filename]
            param['item'] = file_item

            drag_drop = {}
            drag_drop['source']=param
            drag_drop['target']=tth_target
            if tth != tth_target:
                self.drag_drop_signal.emit(drag_drop)
                event.acceptProposedAction()
            


    def encodeData(self, items, stream):
        stream.writeInt32(len(items))
        item = items[0]
        file_item_index = self.file_items.index(item)
        ind, _ = self.identify_item(item)
        #print(' encoded [' + str(ind[0]) + ', '+ str(ind[1])+']')
        
        stream.writeInt32(int(ind[0]))
        stream.writeInt32(int(ind[1]))
        stream.writeInt32(int(file_item_index))
        
        return stream

    def decodeData(self, encoded, tree):
        items = []
        rows = []
        stream = QDataStream(encoded, QIODevice.ReadOnly)
        
        _ = stream.readInt32()
        ind1 = stream.readInt32()
        ind2 = stream.readInt32()
        ind3 = stream.readInt32()
        
        #print('dencoded [' + str(ind1) + ', '+ str(ind2)+']')

        return [ind1, ind2, ind3]
    

    def tth_item_file_dragged_in_callback(self, dragged_in):
        self.files_dragged_in.emit(dragged_in)


    def dragEnterEvent(self, e):
        
        if e.mimeData().hasUrls:
            e.accept()
        else: e.reject()
        
        

    def dragMoveEvent(self, e):
        
        if e.mimeData().hasUrls:
            e.accept()
        else: e.reject()
        

    def dropEvent(self, e):
        """
        Drop files directly onto the widget

        File locations are stored in fname
        :param e:
        :return:
        """
        
        tth = 'auto'
        if e.mimeData().hasUrls:
            e.setDropAction(QtCore.Qt.CopyAction)
            e.accept()
            fnames = list()
            for url in e.mimeData().urls():
                fname = str(url.toLocalFile())
                fnames.append(fname)
            d = {}
            d[tth]=fnames
            self.files_dragged_in.emit(d)
        
    
class FileItem(QtWidgets.QWidget):


    file_dragged_in = pyqtSignal(dict)
    color_btn_clicked_signal = pyqtSignal(QtWidgets.QWidget)
   
    
    def __init__(self,file_use, filename ):
        super().__init__()

        
        self.show_cb = QtWidgets.QCheckBox()
        self.show_cb.setChecked(file_use)
        
        self.show_cb.setStyleSheet("background-color: transparent")
      
        self.file_label = QLabel(filename)
        self.file_label.setStyleSheet("background-color: transparent")

        self._layout = QtWidgets.QHBoxLayout()
        self._layout.setContentsMargins(1,1,1,1)
        self._layout.addWidget(self.show_cb)
        self._layout.addWidget(self.file_label)
        self._layout.addSpacerItem(HorizontalSpacerItem())
        self.setLayout(self._layout)

        #self.setAcceptDrops(True) 


class TthItem(QtWidgets.QWidget):


    file_dragged_in = pyqtSignal(dict)
    color_btn_clicked_signal = pyqtSignal(QtWidgets.QWidget)
    file_dragged_in_from_treewidget = pyqtSignal(dict)
   
    
    def __init__(self,color, text ):
        super().__init__()

        self.color_button = FlatButton()
        self.color_button.setStyleSheet("background-color: " + color)
        self.color_button.setMaximumHeight(18)
        self.color_button.setMinimumWidth(18)
        self.color_button.setMaximumWidth(18)
        
        self.tthItem = QtWidgets.QLabel(text)
        self.tthItem.setStyleSheet("background-color: transparent")
        self._layout = QtWidgets.QHBoxLayout()
        self._layout.setContentsMargins(1,1,1,1)
        self._layout.addWidget(self.color_button)
        self._layout.addWidget(self.tthItem)
        self._layout.addSpacerItem(HorizontalSpacerItem())
        self.setLayout(self._layout)

        self.setAcceptDrops(True) 
        
        #tth_item.editingFinished.connect(partial(self.Tth_edit_finished, tth_item))
        

    def dragEnterEvent(self, e):
        md = e.mimeData()
        
        if md.hasUrls:
            e.accept()
        else: e.reject()
        
        

    def dragMoveEvent(self, e):
        md = e.mimeData()
        if md.hasUrls:
            e.accept()
        else: e.reject()
        

    def dropEvent(self, e):
        """
        Drop files directly onto the widget

        File locations are stored in fname
        :param e:
        :return:
        """
        TW = isinstance(e.source(), QTreeWidget)
        if TW:
            tth = self.tthItem.text()
            self.file_dragged_in_from_treewidget.emit({tth:e})
            return
        
        tth = float(self.tthItem.text())
        md = e.mimeData()
        
            
        if md.hasUrls:
            e.setDropAction(QtCore.Qt.CopyAction)
            e.accept()
            fnames = list()
            for url in e.mimeData().urls():
                fname = str(url.toLocalFile())
                fnames.append(fname)
            d = {}
            d[tth]=fnames
            self.file_dragged_in.emit(d)
        