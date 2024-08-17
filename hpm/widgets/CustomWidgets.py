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

# Based on code from Dioptas - GUI program for fast processing of 2D X-ray diffraction data

from PyQt6.QtCore import QObject, pyqtSignal, Qt
from PyQt6 import QtCore, QtWidgets, QtGui
from math import floor, log10

class TabWidget(QtWidgets.QWidget):
    ''' 
    A tab widget with tabs on the left with text displayed horizontally 
    tab bar is made from regular qpushbuttons since i could not figure out how to rotate the text in a vertical qtabbar
    '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.btn_grp = QtWidgets.QButtonGroup()
        self.btn_grp.setExclusive(True)
        self.tabwidget = QtWidgets.QStackedWidget()
        self.btns = []
        self.btn_grp.buttonClicked.connect(self.handle_button_click)

        self.button_layout = QtWidgets.QVBoxLayout()
        self.button_layout.setSpacing(0)
        self.button_layout.setContentsMargins(2,2,2,2)
        self.button_layout.addSpacerItem(VerticalSpacerItem())
        self.button_layout.addSpacerItem(VerticalSpacerItem())
        
        self.main_layout = QtWidgets.QHBoxLayout()
        self.main_layout.setSpacing(2)
        self.main_layout.setContentsMargins(2,2,2,2)
        self.main_layout.addLayout(self.button_layout)
        self.main_layout.addWidget(self.tabwidget)
        self.setLayout(self.main_layout)

        #self.tabwidget.tabBar().hide()

    def handle_button_click(self, button: QtWidgets.QPushButton):
        button.setChecked(True)
        self.tabwidget.setCurrentIndex(self.btns.index(button))

    def addTab(self, widget, label, desc):
        self.tabwidget.addWidget( widget)
        btn = QtWidgets.QPushButton(label + ' : ' + desc)
        btn.setCheckable(True)
        self.btns.append(btn)
        self.btn_grp.addButton(btn)
       
        stylesheet = """
                        QPushButton {
                            background: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 #3C3C3C, stop:1 #505050);
                            border: 1px solid  #5B5B5B;
                            border-right: 0px solid white;
                            color: #F1F1F1;
                            font: normal 12px;
                            border-radius:2px;
                            min-width: 120px;
                            height: 21px;
                            padding: 1px;
                            padding-left: 5px;
                            margin-top: 1px ;
                            margin-bottom: 1px;
                            Text-align:left;
                            }
                        QPushButton:checked {

                            background: qlineargradient(
                                x1: 0, y1: 1,
                                x2: 0, y2: 0,
                                stop: 0 #727272,
                                stop: 1 #444444
                            );
                            border:1px solid  rgb(255, 120,00);/*#ADADAD; */
                        }
                        QPushButton:hover {
                            border: 1px solid #ADADAD;
                        }
                        """
        stylesheet_last =  """QPushButton {
                                border-bottom-left-radius: 10px;
                                border-bottom-right-radius: 10px;
                            }
                            """
        if len(self.btns) == 1:

            stylesheet_first =  """QPushButton {
                                border-top-left-radius: 10px;
                                border-top-right-radius: 10px;
                            }
                            """
            btn.setStyleSheet(stylesheet + stylesheet_first)
            btn.setChecked(True)
        else:

            btn.setStyleSheet(stylesheet + stylesheet_last)

        self.button_layout.insertWidget(self.button_layout.count() - 1, btn)
        
        if len(self.btns) >2:
            second_to_last_widget = self.button_layout.itemAt(self.button_layout.count() - 3).widget()
            second_to_last_widget.setStyleSheet(stylesheet)

        
        
        
       

    def setCurrentIndex(self, index):
        self.tabwidget.setCurrentIndex(index)
        self.btns[index].setChecked(True)

class NumberTextField(QtWidgets.QLineEdit):
    def __init__(self, *args, **kwargs):
        super(NumberTextField, self).__init__(*args, **kwargs)
        
        self.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.min = None
        self.max = None

        validator = QtGui.QDoubleValidator()
        self.setValidator(validator)
        

    def text(self):
        return super(NumberTextField, self).text().replace(",", ".")

    def value(self):
        return float(self.text())

    def setValue(self, value):
        if self.min is not None:
            if self.min > value:
                value=self.min
            if self.max < value:
                value=self.max    
        self.setText(str(value))

    def setMinimum(self, Minimum):
        if isinstance(Minimum, float) or isinstance(Minimum ,int):
            self.min=float(Minimum)
            validator = self.validator()
            validator.setRange(self.min,validator.top())

    def setMaximum(self, Maximum):
        if isinstance(Maximum, float) or isinstance(Maximum ,int):
            self.max=float(Maximum)
            validator = self.validator()
            validator.setRange(validator.bottom(),self.max)


class IntegerTextField(QtWidgets.QLineEdit):
    def __init__(self, *args, **kwargs):
        super(IntegerTextField, self).__init__(*args, **kwargs)
        self.setValidator(QtGui.QIntValidator())
        self.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)


class LabelAlignRight(QtWidgets.QLabel):
    def __init__(self, *args, **kwargs):
        super(LabelAlignRight, self).__init__(*args, **kwargs)
        self.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)


class CleanLooksComboBox(QtWidgets.QComboBox):
    cleanlooks = QtWidgets.QStyleFactory.create('motif')

    def __init__(self, *args, **kwargs):
        super(CleanLooksComboBox, self).__init__(*args, **kwargs)
        self.setStyle(CleanLooksComboBox.cleanlooks)


class SpinBoxAlignRight(QtWidgets.QSpinBox):
    def __init__(self, *args, **kwargs):
        super(SpinBoxAlignRight, self).__init__(*args, **kwargs)
        self.setAlignment(QtCore.Qt.AlignRight)


class DoubleSpinBoxAlignRight(QtWidgets.QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super(DoubleSpinBoxAlignRight, self).__init__(*args, **kwargs)
        self.setAlignment(QtCore.Qt.AlignRight)



class DoubleMultiplySpinBoxAlignRight(QtWidgets.QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super(DoubleMultiplySpinBoxAlignRight, self).__init__(*args, **kwargs)
        self.setAlignment(QtCore.Qt.AlignRight)

    def stepBy(self, p_int):
        self.setValue(self.calc_new_step(self.value(), p_int))

    def calc_new_step(self, value, p_int):
        pow10floor = 10**floor(log10(value))
        if p_int > 0:
            if value / pow10floor < 1.9:
                return pow10floor * 2.0
            elif value / pow10floor < 4.9:
                return pow10floor * 5.0
            else:
                return pow10floor * 10.0
        else:
            if value / pow10floor < 1.1:
                return pow10floor / 2.0
            elif value / pow10floor < 2.1:
                return pow10floor
            elif value / pow10floor < 5.1:
                return pow10floor * 2.0
            else:
                return pow10floor * 5.0


class FlatButton(QtWidgets.QPushButton):
    def __init__(self, *args):
        super(FlatButton, self).__init__(*args)
        pass
        #self.setFlat(True)


class CheckableFlatButton(FlatButton):
    def __init__(self, *args):
        super(CheckableFlatButton, self).__init__(*args)
        self.setCheckable(True)


class RotatedCheckableFlatButton(CheckableFlatButton):
    def __init__(self, *args):
        super(RotatedCheckableFlatButton, self).__init__(*args)

    def paintEvent(self, event):
        painter = QtWidgets.QStylePainter(self)
        painter.rotate(270)
        painter.translate(-1*self.height(), 0)
        painter.drawControl(QtWidgets.QStyle.CE_PushButton, self.getSyleOptions())

    def minimumSizeHint(self):
        size = super(RotatedCheckableFlatButton, self).minimumSizeHint()
        size.transpose()
        return size

    def sizeHint(self):
        size = super(RotatedCheckableFlatButton, self).sizeHint()
        size.transpose()
        return size

    def getSyleOptions(self):
        options = QtWidgets.QStyleOptionButton()
        options.initFrom(self)
        size = options.rect.size()
        size.transpose()
        options.rect.setSize(size)
        if self.isFlat():
            options.features |= QtWidgets.QStyleOptionButton.Flat
        if self.menu():
            options.features |= QtWidgets.QStyleOptionButton.HasMenu
        if self.autoDefault() or self.isDefault():
            options.features |= QtWidgets.QStyleOptionButton.AutoDefaultButton
        if self.isDefault():
            options.features |= QtWidgets.QStyleOptionButton.DefaultButton
        if self.isDown() or (self.menu() and self.menu().isVisible()):
            options.state |= QtWidgets.QStyle.State_Sunken
        if self.isChecked():
            options.state |= QtWidgets.QStyle.State_On
        if not self.isFlat() and not self.isDown():
            options.state |= QtWidgets.QStyle.State_Raised

        options.text = self.text()
        options.icon = self.icon()
        options.iconSize = self.iconSize()
        return options


class HorizontalLine(QtWidgets.QFrame):
    def __init__(self):
        super(HorizontalLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class VerticalLine(QtWidgets.QFrame):
    def __init__(self):
        super(VerticalLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.VLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class ListTableWidget(QtWidgets.QTableWidget):
    def __init__(self, columns=3):
        super(ListTableWidget, self).__init__()

        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setColumnCount(columns)
        self.horizontalHeader().setVisible(False)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)
        self.setShowGrid(False)

class MultirowListTableWidget(QtWidgets.QTableWidget):
    def __init__(self, columns=3):
        super(ListTableWidget, self).__init__()

        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.setColumnCount(columns)
        self.horizontalHeader().setVisible(False)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)
        self.setShowGrid(False)


class NoRectDelegate(QtWidgets.QItemDelegate):
    def __init__(self):
        super(NoRectDelegate, self).__init__()

    def drawFocus(self, painter, option, rect):
        option.state &= ~QtWidgets.QStyle.State_HasFocus
        QtWidgets.QItemDelegate.drawFocus(self, painter, option, rect)




def HorizontalSpacerItem(minimum_width=0):
    return QtWidgets.QSpacerItem(minimum_width, 0, QtWidgets.QSizePolicy.MinimumExpanding,
                                 QtWidgets.QSizePolicy.Minimum)


def VerticalSpacerItem():
    return QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
