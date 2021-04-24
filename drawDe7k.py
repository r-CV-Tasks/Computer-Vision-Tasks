import sys
from PyQt5 import QtWidgets, QtGui, QtCore, uic


class GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('gui.ui', self)
        self.setFixedSize(self.size())
        self.show()
        self.points = QtGui.QPolygon()

    def mousePressEvent(self, e):
        self.points = e.pos()
        self.update()

    def paintEvent(self, ev):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        pen = QtGui.QPen(QtCore.Qt.red, 5)
        brush = QtGui.QBrush(QtCore.Qt.red)
        qp.setPen(pen)
        qp.setBrush(brush)
        for i in range(self.points.count()):
            qp.drawEllipse(self.points.point(i), 5, 5)
        # or
        # qp.drawPoints(self.points)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = GUI()
    sys.exit(app.exec_())