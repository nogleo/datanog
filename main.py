import datanog as nog
from gui import Ui_MainWindow
import scipy
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
        self.ui.pushButton.clicked.connect(self.getFile)
        self.ui.calibutton.clicked.connect(self.calibrate)
        self.ui.linkSensor.clicked.connect(self.linkSens)
        self.ui.linkSensor.setEnabled(False)
        self.ui.calibutton.setEnabled(False)
        self.ui.pushButton_4.clicked.connect(self.initDevices)
        self.ui.comboBox_2.currentIndexChanged.connect(self.updatePlot)
        
        

        
            

        self.threadpool = qtc.QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        
        self.canv = MatplotlibCanvas(self)
        self.toolbar = None

    def initDevices(self):
        global dn
        dn = nog.daq(fs=float(self.ui.label_4.text()))
        
        self.devsens={}
        for _dev in dn.dev:
            self.devsens[str(_dev[0])] = 'None'
        self.loadDevices()
        self.ui.linkSensor.setEnabled(True)
        self.ui.calibutton.setEnabled(True)

    def pull(self):
        dn.savedata(dn.pulldata(self.ui.label.text()))
        self.ui.startbutton.setEnabled(True)

    def collect(self):
        self.ui.startbutton.setEnabled(False)
        worker = Worker(self.pull)
        self.threadpool.start(worker)
        
    def loadDevices(self):
        print(self.devsens)
        try:
            self.ui.comboBox.clear()
        except :
            pass
        for _sens in dn.dev:
            self.ui.comboBox.addItem(str(_sens[0])+'--'+str(_sens[-1]))

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
        self.plotdata = np.load(self.filename)
        self.updatePlot()
    
    def updatePlot(self):
        global dn
        plt.clf()
        try:
            self.ui.horizontalLayout.removeWidget(self.toolbar)
            self.ui.verticalLayout.removeWidget(self.canv)
            self.toolbar = None
            self.canv = None
        except Exception as e:
            print(e)
            pass
        self.canv = MatplotlibCanvas(self)
        self.toolbar = Navi(self.canv,self.ui.tab_2)
        self.ui.horizontalLayout.addWidget(self.toolbar)
        self.ui.verticalLayout.addWidget(self.canv)
        self.canv.axes.cla()
        ax = self.canv.axes
            
        try:
            if self.ui.comboBox_2.currentText() == 'Time':
                _t = np.arange(len(self.plotdata))*dn.dt              
                ax.plot(_t, self.plotdata)
            elif self.ui.comboBox_2.currentText() == 'Frequency':
                ax.psd(self.plotdata, Fs=dn.fs, NFFT=dn.fs//2, noverlap=dn.fs//4, scale_by_freq=False, detrend='linear', axis=0)
            elif self.ui.comboBox_2.currentText() == 'Time-Frequency':
                for ii in range(self.plotdata.shape[1]):
                    plt.subplot(self.plotdata.shape[1]*100+10+ii+1)
                    f, t, Sxx = scipy.signal.spectrogram(self.plotdata[:,ii], self.fs, axis=0, scaling='spectrum', nperseg=self.fs//4, noverlap=self.fs//8, detrend='linear', mode='psd', window='hann')
                    Sxx[Sxx==0] = 10**(-20)
                    ax.pcolormesh(t, f, 20*np.log10(abs(Sxx)), shading='gouraud', cmap=plt.inferno())
                    ax.ylim((0, dn.fs//8))
                    ax.colorbar()
                    ax.ylabel('Frequency [Hz]')
                    ax.xlabel('Time [sec]')

        except Exception as e:
            print('==>',e)
        self.canv.draw()

    

    



if __name__ == '__main__':
    app = qtw.QApplication([])
    widget = appnog()
    widget.show()
    app.exec_()