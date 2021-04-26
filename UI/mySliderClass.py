from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal


class MySlider(QtWidgets.QSlider):
    # Signal to send to other slots, containing int which identifies the exact type of the sent
    signal = pyqtSignal(int, int)

    def __init__(self, *args):
        super().__init__(*args)
        self.id = ...  # to be added
        self.sliderValue = ...

    def mousePressEvent(self, QMouseEvent):
        super().mousePressEvent(QMouseEvent)
        self.signal.emit(self.id, self.value())
