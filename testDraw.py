import sys
from PyQt5.QtCore    import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui     import *

class MyScribbling(QMainWindow):

    def __init__(self):
        super().__init__()

        self.penOn = QAction(QIcon('Image/ok.png'), 'Включить рисование', self)
        self.penOn.triggered.connect(self.drawingOn)
        self.penOff = QAction(QIcon('Image/exit.png'), 'ВЫКЛЮЧИТЬ рисование', self)
        self.penOff.triggered.connect(self.drawingOff)
        toolbar = self.addToolBar('Инструменты')
        toolbar.addAction(self.penOn)
        toolbar.addAction(self.penOff)

        self.scribbling = False
        self.myPenColor = Qt.red      # +
        self.myPenWidth = 3           # +

        self.lastPoint = QPoint()
        self.image     = QPixmap("src/Images/cat256.jpg")
        self.setFixedSize(600, 600)
        self.resize(self.image.width(), self.image.height())
        self.show()

    # +++
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.image)

    def mousePressEvent(self, event):
        # if event.button() and event.button() == Qt.LeftButton:    # -
        if (event.button() == Qt.LeftButton) and self.scribbling:   # +
            self.lastPoint = event.pos()
            # self.scribbling = True                                # -

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.scribbling:

            # self.drawLineTo(event.pos())                          # -

            # +++
            painter = QPainter(self.image)
            painter.setPen(QPen(self.myPenColor, self.myPenWidth,
                                Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos())
            # self.modified = True                                  # ?
            self.lastPoint = event.pos()
            self.update()

            # ?
            #rad = self.myPenWidth / 2 + 2
            #self.update(QRect(self.lastPoint, event.pos()).normalized().adjusted(-rad, -rad, +rad, +rad))
            #self.lastPoint = QPoint(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.scribbling:
            #self.drawLineTo(event.pos())
            #self.scribbling = False
            pass

#    Перенес в mouseMoveEvent
#    def drawLineTo(self, endPoint):
#        painter = QPainter(self.image)
#        painter.setPen(QPen(self.myPenColor, self.myPenWidth, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
#        painter.drawLine(self.lastPoint, endPoint)
#        self.modified = True
#        rad = self.myPenWidth / 2 + 2
#        self.update(QRect(self.lastPoint, endPoint).normalized().adjusted(-rad, -rad, +rad, +rad))
#        self.lastPoint = QPoint(endPoint)

    # +++
    def drawingOn(self):
        self.scribbling = True

    # +++
    def drawingOff(self):
        self.scribbling = False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainMenu = MyScribbling()
    sys.exit(app.exec_())