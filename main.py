import sys
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import queue
import asyncio
import numpy as np
from PyQt5 import QtCore, QtWidgets,QtGui
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
import datanog
import time


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        fig.tight_layout()

class datanogAPP(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = uic.loadUi('main.ui',self)
        self.resize(800, 480)
        self.q = queue.Queue(maxsize=10000)
        self.pushButton.clicked.connect(self.pulldata)

    def pulldata(self):
        i=0
        t0=tf = time.perf_counter()
        while i<10000:
            ti=time.perf_counter()
            if ti-tf>=dn.dt:
                tf = ti
                i+=1
                self.q.put(dn.pull(dn.devices[0]))
        t1 = time.perf_counter()
        print(t1-t0)
        print(self.q)
        #self.data = np.array(self.q)
        #np.save('test.npy', self.data)


app = QtWidgets.QApplication(sys.argv)
dn = datanog.daq()
mainWindow = datanogAPP()
mainWindow.show()
sys.exit(app.exec_())
