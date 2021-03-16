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
import sip
import numpy as np

root = os.getcwd()

class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self,parent=None, dpi = 120):
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
        self.ui.calibutton.clicked.connect(self.calib)
        self.ui.linkSensor.clicked.connect(self.linkSens)
        
        self.ui.pushButton_4.clicked.connect(self.initDevices)
        self.threadpool = qtc.QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        
        self.canv = MatplotlibCanvas(self)

    def initDevices(self):
        global dn
        dn = nog.daq()
        self.loadDevices()

    def pull(self):
        dn.savedata(dn.pulldata(self.ui.label.text()))
        self.ui.startbutton.setEnabled(True)
        #qtw.QMessageBox.about(self, 'Data Collected', 'File saved')

    def collect(self):
        self.ui.startbutton.setEnabled(False)
        worker = Worker(self.pull)
        self.threadpool.start(worker)
        
    def loadDevices(self):
        self.devsens={}
        for _dev in dn.dev:
            self.devsens[str(_dev[0])]=''
            self.ui.comboBox.addItem(str(_dev[0]))
        


    def interrupt(self):
        dn.state = 0
    
    def getFile(self):
        """ This function will get the address of the csv file location
            also calls a readData function 
        """
        os.chdir(dn.root)
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
        dn.calibrate(self.ui.comboBox.currentText())

    def linkSens(self):
        os.chdir(dn.root)
        try:
            os.chdir('DATA')
        except :
            pass

        self.filename = qtw.QFileDialog.getOpenFileName()[0]
        print("File :", self.filename)
        self.devsens[self.ui.comboBox.currentText] = self.filename


    def readData(self):
        self.plotdata = np.load(self.filename)
        self.updatePlot()
    
    def updatePlot(self):
        plt.clf()
        try:
            self.ui.horizontalLayout.removeWidget(self.toolbar)
            self.ui.verticalLayout.removeWidget(self.canv)
            sip.delete(self.toolbar)
            sip.delete(self.canv)
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

        if self.ui.plottype.currentText == "Time Data":
            try:
                t = np.arange(0, dn.dt*len(self.plotdata))
                ax.plot(t,self.plotdata)
            except Exception as e:
                print('==>',e)
        elif self.ui.plottype.currentText == "Spectrogram":
            try:
                f, t, Sxx = scipy.signal.spectrogram(self.plotdata,dn.fs, axis=0)
                ax.pcolormesh(t, f, 20*np.log10(abs(Sxx)), shading='gouraud', cmap=plt.cm.viridis)
                ax.ylim((0, 830))
                ax.colorbar()
                ax.ylabel('Frequency [Hz]')
                ax.xlabel('Time [sec]')
                ax.show()
            except Exception as e:
                print('==>',e)    
        self.canv.draw()
        ax.tight_layout()

    

    



if __name__ == '__main__':
    app = qtw.QApplication([])
    widget = appnog()
    widget.show()
    app.exec_()