import datanog as nog
from gui import Ui_MainWindow

from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw

dn = nog.daq()


class Worker(qtc.QRunnable):

    msg_in = qtc.pyqtSignal(str)

    def __init__(self, fn):
        super(Worker, self).__init__()
        self.fn = fn
    @qtc.pyqtSlot()
    def run(self):
        try:
            result = self.fn()
        except Exception:
            pass

class appnog(qtw.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.startbutton.clicked.connect(self.collect)
        self.ui.stopbutton.clicked.connect(self.interrupt)
        self.threadpool = qtc.QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

    def pull(self):
        self.msg = dn.savedata(dn.pulldata(self.ui.label.text()))
        qtw.QMessageBox.about(self, 'Data Collected', '{} saved'.format(_msg))

    def collect(self):
        worker = Worker(self.pull)
        self.threadpool.start(worker)
        


    def interrupt(self):
        dn.state = False

    



if __name__ == '__main__':
    app = qtw.QApplication([])
    widget = appnog()
    widget.show()
    app.exec_()