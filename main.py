import datanog as nog
from gui import Ui_MainWindow

from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw

dn = nog.daq()
class appnog(qtw.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.startbutton.clicked.connect(self.collect)
        self.ui.stopbutton.clicked.connect(self.interrupt)

    def collect(self):
        dn.savedata(dn.pulldata(self.ui.label.text()))
    

    def interrupt(self):
        dn.state = False




if __name__ == '__main__':
    app = qtw.QApplication([])
    widget = appnog()
    widget.show()
    app.exec_()