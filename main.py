import datanog as nog
from gui import Ui_MainWindow
import scipy.signal as signal
import os
from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw
from PyQt5 import Qt as qt
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as Navi
from matplotlib.figure import Figure
import numpy as np
import sigprocess as sp
import pandas as pd
root = os.getcwd()

class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self,parent=None, dpi = 100):
        fig = Figure(dpi = dpi)
        self.axes = fig.add_subplot(111)
        super(MatplotlibCanvas,self).__init__(fig)
        fig.tight_layout()

class Worker(qtc.QRunnable):
    def __init__(self, fn):
        super(Worker, self).__init__()
        self.fn = fn
    @qtc.pyqtSlot()
    def run(self):
        try:
            result = self.fn()
        except Exception as e:
            print(e)

class appnog(qtw.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.msg = ""
        self.ui.startbutton.clicked.connect(self.collect)
        self.ui.stopbutton.clicked.connect(self.interrupt)
        self.ui.openbttn.clicked.connect(self.getFile)
        self.ui.calibutton.clicked.connect(self.calibrate)
        self.ui.linkSensor.clicked.connect(self.linkSens)
        self.ui.linkSensor.setEnabled(False)
        self.ui.calibutton.setEnabled(False)
        self.ui.pushButton_4.clicked.connect(self.initDevices)
        self.ui.plotbttn.clicked.connect(self.updatePlot)
        self.ui.processbttn.clicked.connect(self.processData)

        
        
        

        
            

        self.threadpool = qtc.QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        
        self.canv = MatplotlibCanvas(self)
        self.toolbar = None

    def initDevices(self):
        global dn, fs, dt
        dn = nog.daq()
        
        self.devsens={}
        for _dev in dn.dev:
            self.devsens[str(_dev[0])] = str(_dev[-1])
        self.loadDevices()
        self.ui.linkSensor.setEnabled(True)
        self.ui.calibutton.setEnabled(True)

    def pull(self):
        data = dn.savedata(dn.pulldata(self.ui.label.text()))
        sp.PSD(data, dn.fs)
        self.ui.startbutton.setEnabled(True)

    def collect(self):
        self.ui.startbutton.setEnabled(False)
        worker = Worker(self.pull)
        self.threadpool.start(worker)
        
    def loadDevices(self):
        try:
            self.ui.comboBox.clear()
        except :
            pass
        for _sens in dn.dev:
            self.devsens[str(_sens[0])] = str(_sens[-1])
            self.ui.comboBox.addItem(str(_sens[0])+'--'+str(_sens[-1]))
        print(self.devsens)

    def interrupt(self):
        dn.state = 0
    
    def getFile(self):
        """ This function will get the address of the csv file location
            also calls a readData function 
        """
        os.chdir(root)
        try:
            os.chdir('DATA')
        except :
            pass

        self.filename = qtw.QFileDialog.getOpenFileName()[0]
        print("File :", self.filename)
        try:
            self.readData()
        except Exception:
            pass

    def calib(self):
        dn.calibrate(dn.dev[self.ui.comboBox.currentIndex()])

    def calibrate(self):
        workal = Worker(self.calib)
        self.threadpool.start(workal)

    def linkSens(self):
        os.chdir(root)
        try:
            os.chdir('sensors')
        except :
            pass

        self.filename = qtw.QFileDialog.getOpenFileName()[0]
        print("File :", self.filename)
        ii = self.ui.comboBox.currentIndex()
        dn.dev[ii][-1] = self.filename[25:]
        self.loadDevices()
        os.chdir(root)
        np.save('devsens.npy', self.devsens)


    def readData(self):
        self.data = pd.read_csv(self.filename, index_col='t')
        
        
    def processData(self):
        self.ui.processbttn.setEnabled(False)
        self.ui.openbttn.setEnabled(False)
        self.ui.plotbttn.setEnabled(False)
        _path = self.filename[:-4]
        _path = _path.replace('DATA', 'PROCDATA')
        try:
            os.chdir(_path)
        except:
            os.mkdir(_path)
            os.chdir(_path)
        print(_path)

        cma = np.array([-4.4019e-004	, 1.2908e-003,	-1.9633e-002])
        La = np.array([-8.3023e-019, 	-8.1e-002,	-8.835e-002])
        posa = La-cma

        cmb = np.array([8.0563e-005,	5.983e-004,	-6.8188e-003])
        Lb = np.array([5.3302e-018, -7.233e-002, 3.12e-002+2.0e-003])
        posb = Lb-cmb
        data, t, fs, dt = sp.prep_data(self.data, 1660, 415, 10)
        self.Fs = fs

        A = sp.imu2body(data[:,2:8],t, fs, posa)
        A.to_csv('A.csv')
        B = sp.imu2body(data[:,8:], t, fs, posb)
        B.to_csv('B.csv')
        c = {'cur': data[:,1],
             'rot': data[:,0]}
        C = pd.DataFrame(c,t)
        C.to_csv('C.csv')
                
        self.ui.processbttn.setEnabled(True)
        self.ui.openbttn.setEnabled(True)
        self.ui.plotbttn.setEnabled(True)
    
    def updatePlot(self):
        global dn
        plt.clf()
        try:
            self.ui.horizontalLayout.removeWidget(self.toolbar)
            self.ui.plotLayout.removeWidget(self.canv)
            self.toolbar = None
            self.canv = None
        except Exception as e:
            print(e)
            pass
        self.canv = MatplotlibCanvas(self)
        self.toolbar = Navi(self.canv,self.ui.tab_2)
        self.ui.horizontalLayout.addWidget(self.toolbar)
        self.ui.plotLayout.addWidget(self.canv)
        self.canv.axes.cla()
        ax = self.canv.axes
            
        try:        
                       
            ax.plot(self.data)
            ax.legend(self.data.columns)            
            

        except Exception as e:
            print('==>',e)
        self.canv.draw()

    

    



if __name__ == '__main__':
    app = qtw.QApplication([])
    widget = appnog()
    widget.show()
    app.exec_()