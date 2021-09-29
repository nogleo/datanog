import datanog as nog
from gui import Ui_MainWindow
import scipy.signal as signal
import os
import time
import pickle
from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw
from PyQt5 import Qt as qt
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as Navi
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import gc
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
        self.ui.calibutton.clicked.connect(self.calibration)
        self.ui.linkSensor.clicked.connect(self.linkSens)
        self.ui.linkSensor.setEnabled(False)
        self.ui.calibutton.setEnabled(False)
        self.ui.initbttn.clicked.connect(self.initDevices)
        
        

        
        
        

        
            

        self.threadpool = qtc.QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        
        self.canv = MatplotlibCanvas(self)
        self.toolbar = None

    def initDevices(self):
        global dn, fs, dt
        dn = nog.daq()
        
        self.devsens={}
        try:
            with open(root+'sensors.data', 'rb') as f:
                dn.dev = pickle.load(f)
            print(root+'sensors.data loaded')
        except:
            print('no previous sensor data')
        for _dev in dn.dev:
            self.devsens[str(_dev[0])] = str(_dev[-1])
        self.loadDevices()
        self.ui.linkSensor.setEnabled(True)
        self.ui.calibutton.setEnabled(True)

    def pull(self):
        data = dn.savedata(dn.pulldata(self.ui.label.text()))
        
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
        with open(root+'sensors.data', 'wb') as f:
            pickle.dump(dn.dev, f)
        os.chdir(root)
        np.save('devsens.npy', self.devsens)


    def readData(self):
        data = pd.read_csv(self.filename, index_col='t')
        self.updatePlot(data)
        
        
       
    def updatePlot(self, plotdata):
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
                       
            ax.plot(plotdata)
            ax.legend(self.data.columns)            
            

        except Exception as e:
            print('==>',e)
        self.canv.draw()

    def calibration(self):
        device = dn.dev[self.ui.comboBox.currentIndex()]
        self.calibrationdata = []
        msg, ok = qtw.QInputDialog().getText(self,
                                        'Name your IMU',
                                        'Type the name of your IMU for calibration: ',
                                        qtw.QLineEdit.Normal)
        if ok and msg:
            sensor ={'name': msg} 
        
        NS, ok = qtw.QInputDialog().getInt(self,
                                        'Sample Length', 
                                        'Number seconds per Position: ', 
                                        value=5, minValue=1, maxValue=10)
        if ok and NS:
            self.NS = NS//dn.dt
        
        for ii in range(6):
            ok = qtw.QMessageBox(self, 'Position {}'.format(ii+1),
                                        'Position Calibration Dice with the side {} upwards'.format(ii+1))
            if ok:
                i=0
                tf = time.perf_counter()
                while i<self.NS:
                    ti=time.perf_counter()
                    if tf-ti>=dn.dt:
                        tf = ti
                        i+=1
                        self.calibrationdata.append(dn.pull(device))
        ND, ok = qtw.QInputDialog().getInt()(self,
                                        'Sample Length', 
                                        'Number seconds per Rotation: ', 
                                        value=5, minValue=1, maxValue=10)
        if ok and ND:
            self.ND = ND//dn.dt
        for ii in range(0,6,2):
            ok = qtw.QMessageBox(self, 'Rotation axis {}-{}'.format(ii+1,ii+2),
                                        'Rotate 180 deg around axis {}-{}'.format(ii+1,ii+2))
            if ok:
                i=0
                tf = time.perf_counter()
                while i<self.ND:
                    ti=time.perf_counter()
                    if tf-ti>=dn.dt:
                        tf = ti
                        i+=1
                        self.calibrationdata.append(dn.pull(device))

        self.calibrationdata = np.array(self.calibrationdata)
        self.updatePlot(self.calibrationdata)
        sensor['acc_p'] = dn.calibacc(self.calibrationdata[0:6*self.NS,3:6])
        gc.collect()
        sensor['gyr_p'] = dn.calibgyr(self.calibrationdata[:,0:3])        
        np.savez('./sensors/'+sensor['name'], sensor['gyr_p'], sensor['acc_p'])

            



if __name__ == '__main__':
    app = qtw.QApplication([])
    widget = appnog()
    widget.show()
    app.exec_()