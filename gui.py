# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(650, 400)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(650, 400))
        MainWindow.setMaximumSize(QtCore.QSize(2500, 2000))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setMaximumSize(QtCore.QSize(800, 480))
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 650, 400))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setMinimumSize(QtCore.QSize(650, 400))
        self.tabWidget.setMaximumSize(QtCore.QSize(1600, 960))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_1 = QtWidgets.QWidget()
        self.tab_1.setObjectName("tab_1")
        self.layoutWidget = QtWidgets.QWidget(self.tab_1)
        self.layoutWidget.setGeometry(QtCore.QRect(-2, 2, 631, 371))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(5)
        self.gridLayout.setObjectName("gridLayout")
        self.stopbutton = QtWidgets.QPushButton(self.layoutWidget)
        self.stopbutton.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.stopbutton.sizePolicy().hasHeightForWidth())
        self.stopbutton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(40)
        self.stopbutton.setFont(font)
        self.stopbutton.setCheckable(False)
        self.stopbutton.setObjectName("stopbutton")
        self.gridLayout.addWidget(self.stopbutton, 2, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(60)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 1, 1, 1)
        self.startbutton = QtWidgets.QPushButton(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.startbutton.sizePolicy().hasHeightForWidth())
        self.startbutton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(40)
        self.startbutton.setFont(font)
        self.startbutton.setAutoFillBackground(False)
        self.startbutton.setObjectName("startbutton")
        self.gridLayout.addWidget(self.startbutton, 0, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(60)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 2, 1, 1)
        self.horizontalSlider = QtWidgets.QSlider(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(3)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.horizontalSlider.sizePolicy().hasHeightForWidth())
        self.horizontalSlider.setSizePolicy(sizePolicy)
        self.horizontalSlider.setMinimumSize(QtCore.QSize(100, 55))
        self.horizontalSlider.setSizeIncrement(QtCore.QSize(0, 0))
        self.horizontalSlider.setBaseSize(QtCore.QSize(10, 10))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.horizontalSlider.setFont(font)
        self.horizontalSlider.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.horizontalSlider.setMaximum(120)
        self.horizontalSlider.setPageStep(5)
        self.horizontalSlider.setSliderPosition(0)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setInvertedAppearance(False)
        self.horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.horizontalSlider.setTickInterval(5)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.gridLayout.addWidget(self.horizontalSlider, 2, 1, 1, 2)
        self.tabWidget.addTab(self.tab_1, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabWidget.addTab(self.tab_2, "")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.horizontalSlider.valueChanged['int'].connect(self.label.setNum)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.startbutton, self.stopbutton)
        MainWindow.setTabOrder(self.stopbutton, self.tabWidget)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.stopbutton.setText(_translate("MainWindow", "Stop"))
        self.label_2.setText(_translate("MainWindow", "Duration: "))
        self.startbutton.setToolTip(_translate("MainWindow", "\'Start collecting data\'"))
        self.startbutton.setText(_translate("MainWindow", "Start"))
        self.label.setText(_translate("MainWindow", "0"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_1), _translate("MainWindow", "Collect"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Process"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
