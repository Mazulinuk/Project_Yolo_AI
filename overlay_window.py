from PyQt5 import QtWidgets, QtGui, QtCore

class OverlayWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint | 
            QtCore.Qt.WindowStaysOnTopHint | 
            QtCore.Qt.Tool | 
            QtCore.Qt.X11BypassWindowManagerHint
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        self.objects = []

    def update_objects(self, objects):
        self.objects = objects
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        for x1, y1, x2, y2, label in self.objects:
            pen = QtGui.QPen(QtGui.QColor(255, 0, 0), 4)
            painter.setPen(pen)
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            painter.setBrush(QtGui.QColor(0, 0, 0, 150))
            painter.drawRect(x1, max(20, y1 - 30), len(label) * 10, 25)
            painter.setPen(QtGui.QColor(255, 255, 255))
            painter.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Bold))
            painter.drawText(x1 + 5, max(35, y1 - 10), label)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    overlay = OverlayWindow()
    overlay.show()
    sys.exit(app.exec_())
