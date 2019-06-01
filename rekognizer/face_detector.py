import numpy as np
from mtcnn.mtcnn import MTCNN

face_detector = MTCNN()


class FaceDetector:
    @staticmethod
    def detect_faces(image: np.array):
        return face_detector.detect_faces(image)
