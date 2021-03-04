import datanog as nog
from gui import Ui_MainWindow
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

dn = nog.daq()

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
        self.threadpool = qtc.QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        
        self.canv = MatplotlibCanvas(self)


    def pull(self):
        dn.savedata(dn.pulldata(self.ui.label.text()))
        self.ui.startbutton.setEnabled(True)
        qtw.QMessageBox.about(self, 'Data Collected', 'File saved')

    def collect(self):
        self.ui.startbutton.setEnabled(False)
        worker = Worker(self.pull)
        self.threadpool.start(worker)
        


    def interrupt(self):
        dn.state = False
    
    def getFile(self):
        """ This function will get the address of the csv file location
            also calls a readData function 
        """
        os.chdir(dn.root)
        os.chdir('DATA')
        self.filename = qtw.QFileDialog.getOpenFileName()[0]
        print("File :", self.filename)
        self.readData()

    
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
            self.ui.verticalLayout.removeItem(self.spacerItem1)
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
            ax.plot(self.plotdata)
            
        except Exception as e:
            print('==>',e)
        self.canv.draw()
        ax.tight_layout()

    



if __name__ == '__main__':
    app = qtw.QApplication([])
    widget = appnog()
    widget.show()
    app.exec_()