# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'aEDXDWidget.ui'
#
# Created by: PyQt6 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt6 import QtCore, QtGui, QtWidgets

class Ui_aEDXDWidget(object):
    def setupUi(self, aEDXDWidget):
        aEDXDWidget.setObjectName("aEDXDWidget")
        aEDXDWidget.resize(793, 574)
        self.centralwidget = QtWidgets.QWidget(aEDXDWidget)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")

        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.vertical_layout_tab1 = QtWidgets.QVBoxLayout(self.tab)
        self.vertical_layout_tab1.setObjectName("vertical_layout_tab1")
        self.tabWidget.addTab(self.tab, "")

        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.tab_2)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.tabWidget.addTab(self.tab_2, "")

        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.tab_3)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.tabWidget.addTab(self.tab_3, "")

        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.tab_4)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.tabWidget.addTab(self.tab_4, "")

        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.tab_5)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.tabWidget.addTab(self.tab_5, "")

        self.tab_6 = QtWidgets.QWidget()
        self.tab_6.setObjectName("tab_6")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.tab_6)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.tabWidget.addTab(self.tab_6, "")

        self.horizontalLayout.addWidget(self.tabWidget)
        aEDXDWidget.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(aEDXDWidget)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 793, 22))
        self.menubar.setObjectName("menubar")
        aEDXDWidget.setMenuBar(self.menubar)

        self.retranslateUi(aEDXDWidget)
        
        QtCore.QMetaObject.connectSlotsByName(aEDXDWidget)

    

    def retranslateUi(self, aEDXDWidget):
        _translate = QtCore.QCoreApplication.translate
        aEDXDWidget.setWindowTitle(_translate("aEDXDWidget", "aEDXD"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("aEDXDWidget", "Bin / cut"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("aEDXDWidget", "All spectra"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("aEDXDWidget", "Incident beam"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("aEDXDWidget", "S(q)"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("aEDXDWidget", "G(r)"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_6), _translate("aEDXDWidget", "S(q) filtered"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    aEDXDWidget = QtWidgets.QMainWindow()
    ui = Ui_aEDXDWidget()
    ui.setupUi(aEDXDWidget)
    aEDXDWidget.show()
    sys.exit(app.exec())

