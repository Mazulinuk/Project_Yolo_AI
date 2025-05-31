import threading
import time
import sys
from PyQt5 import QtWidgets
from overlay_window import OverlayWindow
from object_detection import detect_objects
import mss
import numpy as np
import cv2

class ObjectDetectionApp:
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.overlay = OverlayWindow()
        self.overlay.show()
        self.running = True

    def capture_and_detect(self):
        with mss.mss() as sct:
            monitor = sct.monitors[1]

            while self.running:
                screenshot = sct.grab(monitor)
                img = np.array(screenshot)
                detected_objects = detect_objects(img)
                self.overlay.update_objects(detected_objects)
                time.sleep(0.015)

    def run(self):
        detection_thread = threading.Thread(target=self.capture_and_detect, daemon=True)
        detection_thread.start()
        sys.exit(self.app.exec_())

if __name__ == "__main__":
    app = ObjectDetectionApp()
    app.run()
