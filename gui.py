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
        MainWindow.resize(750, 450)
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
        self.gridLayout_3 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_3.setObjectName("gridLayout_3")
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
        self.comboBox_2 = QtWidgets.QComboBox(self.tab_1)
        self.comboBox_2.setObjectName("comboBox_2")
        self.gridLayout.addWidget(self.comboBox_2, 0, 2, 1, 1)
        self.pushButton_5 = QtWidgets.QPushButton(self.tab_1)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setObjectName("pushButton_5")
        self.gridLayout.addWidget(self.pushButton_5, 0, 3, 1, 1)
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
        self.gridLayout.addWidget(self.startbutton, 2, 0, 1, 1)
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
        self.gridLayout.addWidget(self.label_2, 2, 1, 1, 2)
        self.comboBox = QtWidgets.QComboBox(self.tab_1)
        self.comboBox.setObjectName("comboBox")
        self.gridLayout.addWidget(self.comboBox, 0, 1, 1, 1)
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
        self.gridLayout.addWidget(self.horizontalSlider, 3, 1, 1, 3)
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
        self.gridLayout.addWidget(self.label, 2, 3, 1, 1)
        self.radioButton = QtWidgets.QRadioButton(self.tab_1)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.radioButton.setFont(font)
        self.radioButton.setObjectName("radioButton")
        self.gridLayout.addWidget(self.radioButton, 1, 1, 1, 1)
        self.radioButton_2 = QtWidgets.QRadioButton(self.tab_1)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.radioButton_2.setFont(font)
        self.radioButton_2.setObjectName("radioButton_2")
        self.gridLayout.addWidget(self.radioButton_2, 1, 2, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.tab_1)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 1, 3, 1, 1)
        self.pushButton_4 = QtWidgets.QPushButton(self.tab_1)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout.addWidget(self.pushButton_4, 0, 0, 2, 1)
        self.tabWidget.addTab(self.tab_1, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton = QtWidgets.QPushButton(self.tab_2)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.plottype = QtWidgets.QComboBox(self.tab_2)
        self.plottype.setObjectName("plottype")
        self.plottype.addItem("")
        self.plottype.addItem("")
        self.plottype.addItem("")
        self.horizontalLayout.addWidget(self.plottype)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.gridLayout_2.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.gridLayout_2.addLayout(self.verticalLayout, 1, 0, 1, 1)
        self.tabWidget.addTab(self.tab_2, "")
        self.gridLayout_3.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.horizontalSlider.valueChanged['int'].connect(self.label.setNum)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "DATANOG"))
        self.pushButton_5.setText(_translate("MainWindow", "Link"))
        self.startbutton.setToolTip(_translate("MainWindow", "\'Start collecting data\'"))
        self.startbutton.setText(_translate("MainWindow", "Start"))
        self.label_2.setText(_translate("MainWindow", "Duration: "))
        self.stopbutton.setText(_translate("MainWindow", "Stop"))
        self.label.setText(_translate("MainWindow", "0"))
        self.radioButton.setText(_translate("MainWindow", "Scaled"))
        self.radioButton_2.setText(_translate("MainWindow", "Raw"))
        self.pushButton_2.setText(_translate("MainWindow", "Calibrate"))
        self.pushButton_4.setText(_translate("MainWindow", "Init Devs"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_1), _translate("MainWindow", "Collect"))
        self.pushButton.setText(_translate("MainWindow", "Open"))
        self.plottype.setItemText(0, _translate("MainWindow", "Time Data"))
        self.plottype.setItemText(1, _translate("MainWindow", "Spectrogram"))
        self.plottype.setItemText(2, _translate("MainWindow", "PSD"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Process"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
