# Loosely based on http://benhoff.net/face-detection-opencv-pyqt.html


from PySide2 import QtCore, QtWidgets, QtGui
# from PyQt5 import QtCore, QtWidgets, QtGui

import sys
from os import path

import cv2
import numpy as np


from centerface import CenterFace


class RecordVideo(QtCore.QObject):
    image_data = QtCore.Signal(np.ndarray)

    def __init__(self, camera_port=0, parent=None):
        super().__init__(parent)
        self.camera = cv2.VideoCapture(camera_port)

        self.timer = QtCore.QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return

        read, data = self.camera.read()
        if read:
            self.image_data.emit(data)


class FaceDetectionWidget(QtWidgets.QWidget):
    def __init__(self, onnx_filepath, parent=None):
        super().__init__(parent)
        self.detector = CenterFace(onnx_path=onnx_filepath)
        self.image = QtGui.QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)

    def detect_faces(self, image: np.ndarray):
        dets, lms = self.detector(image, threshold=0.5)
        faces = []
        for det in dets:
            boxes, score = det[:4], det[4]
            x1, y1, x2, y2 = boxes.astype(int)
            faces.append((x1, y1, x2, y2))
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)

        # faces = [(30, 30, 30, 30)]
        return faces

    def image_data_slot(self, image_data):
        faces = self.detect_faces(image_data)
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(image_data,
        #                   (x, y),
        #                   (x+w, y+h),
        #                   self._red,
        #                   self._width)
        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(image_data, (x1, y1), (x2, y2), (0, 0, 0), -1)

        self.image = self.get_qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()


class MainWidget(QtWidgets.QWidget):
    def __init__(self, onnx_filepath, parent=None):
        super().__init__(parent)
        fp = onnx_filepath

        # TODO: set video port
        self.record_video = RecordVideo()
        self.face_detection_widget = FaceDetectionWidget(fp)


        image_data_slot = self.face_detection_widget.image_data_slot
        self.record_video.image_data.connect(image_data_slot)

        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.face_detection_widget)
        self.text_input = QtWidgets.QLineEdit('0')
        layout.addWidget(self.text_input)
        self.run_button = QtWidgets.QPushButton('Start')
        layout.addWidget(self.run_button)

        self.run_button.clicked.connect(self.record_video.start_recording)
        self.setLayout(layout)


def main(onnx_filepath):
    app = QtWidgets.QApplication(sys.argv)

    main_window = QtWidgets.QMainWindow()
    main_widget = MainWidget(onnx_filepath)
    main_window.setCentralWidget(main_widget)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    # script_dir = path.dirname(path.realpath(__file__))
    # cascade_filepath = path.join(script_dir,
                                #  '..',
                                #  'data',
                                #  'haarcascade_frontalface_default.xml')

    cascade_filepath = 'uga'#path.abspath(cascade_filepath)
    main(cascade_filepath)
