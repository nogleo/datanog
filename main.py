import sys
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import numpy as np
from PyQt5 import QtCore, QtWidgets,QtGui
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
import queue
import asyncio
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
        self.q = queue.Queue()
        self.pushButton.clicked.connect(self.pulldata)
        q = queue.Queue()

    def pulldata(self, _size = 1):
        i=0
        t0=tf = time.perf_counter()
        while i< _size//dn.dt:
            ti=time.perf_counter()
            if ti-tf>=dn.dt:
                tf = ti
                i+=1
                q.put(dn.pull(dn.devices[0]))
        t1 = time.perf_counter()
        print(t1-t0)

    def savedata(self):
        if 'DATA' not in os.listdir():
            os.mkdir('DATA')
        data = []
        while q.qsize()>0:
            data.append(q.get())
        arr = np.array(data)
        os.chdir('DATA')
        np.save('test{}.npy'.format(len(os.listdir())), arr)
        os.chdir('..')


app = QtWidgets.QApplication(sys.argv)
dn = datanog.daq()
mainWindow = datanogAPP()
mainWindow.show()
sys.exit(app.exec_())
