# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(773, 445)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMaximumSize(QtCore.QSize(2500, 2000))
        MainWindow.setBaseSize(QtCore.QSize(700, 450))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_1 = QtWidgets.QWidget()
        self.tab_1.setObjectName("tab_1")
        self.gridLayout = QtWidgets.QGridLayout(self.tab_1)
        self.gridLayout.setObjectName("gridLayout")
        self.stopbutton = QtWidgets.QPushButton(self.tab_1)
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
        self.gridLayout.addWidget(self.stopbutton, 3, 0, 1, 1)
        self.horizontalSlider = QtWidgets.QSlider(self.tab_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
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
        self.gridLayout.addWidget(self.horizontalSlider, 3, 3, 1, 3)
        self.startbutton = QtWidgets.QPushButton(self.tab_1)
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
        self.label_2 = QtWidgets.QLabel(self.tab_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(60)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 3, 1, 2)
        self.label = QtWidgets.QLabel(self.tab_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(60)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 5, 1, 1)
        self.tabWidget.addTab(self.tab_1, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.plotLayout = QtWidgets.QVBoxLayout()
        self.plotLayout.setObjectName("plotLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.openbttn = QtWidgets.QPushButton(self.tab_2)
        self.openbttn.setObjectName("openbttn")
        self.horizontalLayout.addWidget(self.openbttn)
        self.psdbox = QtWidgets.QCheckBox(self.tab_2)
        self.psdbox.setObjectName("psdbox")
        self.horizontalLayout.addWidget(self.psdbox)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.plotLayout.addLayout(self.horizontalLayout)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.plotLayout.addItem(spacerItem1)
        self.gridLayout_2.addLayout(self.plotLayout, 0, 0, 3, 2)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_TF = QtWidgets.QWidget()
        self.tab_TF.setObjectName("tab_TF")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.tab_TF)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_TF = QtWidgets.QVBoxLayout()
        self.verticalLayout_TF.setObjectName("verticalLayout_TF")
        self.horizontalLayout_TF = QtWidgets.QHBoxLayout()
        self.horizontalLayout_TF.setObjectName("horizontalLayout_TF")
        self.loadbttn = QtWidgets.QPushButton(self.tab_TF)
        self.loadbttn.setObjectName("loadbttn")
        self.horizontalLayout_TF.addWidget(self.loadbttn)
        self.combo_TF = QtWidgets.QComboBox(self.tab_TF)
        self.combo_TF.setObjectName("combo_TF")
        self.horizontalLayout_TF.addWidget(self.combo_TF)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_TF.addItem(spacerItem2)
        self.verticalLayout_TF.addLayout(self.horizontalLayout_TF)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_TF.addItem(spacerItem3)
        self.horizontalLayout_2.addLayout(self.verticalLayout_TF)
        self.tabWidget.addTab(self.tab_TF, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setEnabled(True)
        self.tab_3.setObjectName("tab_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab_3)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.calibutton = QtWidgets.QPushButton(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.calibutton.setFont(font)
        self.calibutton.setObjectName("calibutton")
        self.gridLayout_3.addWidget(self.calibutton, 0, 5, 1, 1)
        self.comboBox = QtWidgets.QComboBox(self.tab_3)
        self.comboBox.setObjectName("comboBox")
        self.gridLayout_3.addWidget(self.comboBox, 0, 3, 1, 1)
        self.initbttn = QtWidgets.QPushButton(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.initbttn.setFont(font)
        self.initbttn.setObjectName("initbttn")
        self.gridLayout_3.addWidget(self.initbttn, 0, 2, 1, 1)
        self.linkSensor = QtWidgets.QPushButton(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.linkSensor.setFont(font)
        self.linkSensor.setObjectName("linkSensor")
        self.gridLayout_3.addWidget(self.linkSensor, 0, 4, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem4, 1, 3, 1, 1)
        self.tabWidget.addTab(self.tab_3, "")
        self.gridLayout_4.addWidget(self.tabWidget, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.horizontalSlider.valueChanged['int'].connect(self.label.setNum)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "DATANOG"))
        self.stopbutton.setText(_translate("MainWindow", "Stop"))
        self.startbutton.setToolTip(_translate("MainWindow", "\'Start collecting data\'"))
        self.startbutton.setText(_translate("MainWindow", "Start"))
        self.label_2.setText(_translate("MainWindow", "Duration: "))
        self.label.setText(_translate("MainWindow", "0"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_1), _translate("MainWindow", "DAQ"))
        self.openbttn.setText(_translate("MainWindow", "Open"))
        self.psdbox.setText(_translate("MainWindow", "PSD"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Process"))
        self.loadbttn.setText(_translate("MainWindow", "Load"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_TF), _translate("MainWindow", "Time Frequency"))
        self.calibutton.setText(_translate("MainWindow", "Calibrate"))
        self.initbttn.setText(_translate("MainWindow", "Init Devs"))
        self.linkSensor.setText(_translate("MainWindow", "Link"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Calibration"))
