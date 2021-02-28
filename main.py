from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as Navi
from matplotlib.figure import Figure
import datanog
import numpy as np

class MatplotlibCanvas(FigureCanvasQTAgg):
	def __init__(self,parent=None, dpi = 120):
		fig = Figure(dpi = dpi)
		self.axes = fig.add_subplot(111)
		super(MatplotlibCanvas,self).__init__(fig)
		fig.tight_layout()


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 480)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(800, 480))
        MainWindow.setMaximumSize(QtCore.QSize(800, 480))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setMaximumSize(QtCore.QSize(800, 480))
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(-4, -1, 811, 491))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_1 = QtWidgets.QWidget()
        self.tab_1.setObjectName("tab_1")
        self.stopbutton = QtWidgets.QPushButton(self.tab_1)
        self.stopbutton.setEnabled(False)
        self.stopbutton.setGeometry(QtCore.QRect(120, 10, 100, 27))
        self.stopbutton.setCheckable(False)
        self.stopbutton.setObjectName("stopbutton")
        self.startbutton = QtWidgets.QPushButton(self.tab_1)
        self.startbutton.setGeometry(QtCore.QRect(10, 10, 100, 27))
        self.startbutton.setObjectName("startbutton")
        self.label = QtWidgets.QLabel(self.tab_1)
        self.label.setGeometry(QtCore.QRect(230, 10, 76, 19))
        self.label.setObjectName("label")
        self.frame = QtWidgets.QFrame(self.tab_1)
        self.frame.setGeometry(QtCore.QRect(10, 40, 781, 401))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalSlider = QtWidgets.QSlider(self.tab_1)
        self.horizontalSlider.setGeometry(QtCore.QRect(310, -1, 301, 31))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.checkBox = QtWidgets.QCheckBox(self.tab_1)
        self.checkBox.setGeometry(QtCore.QRect(640, 10, 104, 25))
        self.checkBox.setObjectName("checkBox")
        self.tabWidget.addTab(self.tab_1, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabWidget.addTab(self.tab_2, "")
        MainWindow.setCentralWidget(self.centralwidget)
        
        self.startbutton.clicked.connect(self.plotdata)


        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.horizontalSlider.sliderMoved['int'].connect(self.label.setNum)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.canv = MatplotlibCanvas(self)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.stopbutton.setText(_translate("MainWindow", "Stop"))
        self.startbutton.setToolTip(_translate("MainWindow", "\'Start collecting data\'"))
        self.startbutton.setText(_translate("MainWindow", "Start"))
        self.label.setText(_translate("MainWindow", "Duration"))
        self.checkBox.setText(_translate("MainWindow", "LivePlot"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_1), _translate("MainWindow", "Collect"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Process"))

    def plotdata(self):
        _d = np.random.i2cderand(1660,13)
        self.canv = MatplotlibCanvas(self)
        self.canv.axes.cla()
        ax = self.canv.axes
        ax.plot(_d)
        self.canv.draw()



if __name__ == "__main__":
    import sys
    dn = datanog.daq()
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
