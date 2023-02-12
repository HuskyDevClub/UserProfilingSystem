from os import path as OS_PATH
from typing import Sequence

import cv2


class Images:

    # face recognition classifier
    __FACE_CASCADE: cv2.CascadeClassifier = cv2.CascadeClassifier(
        OS_PATH.join(
            OS_PATH.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml"
        )
    )
    # image shape
    SIZE: tuple[int, int] = (64, 64)

    @staticmethod
    def load(path: str) -> cv2.Mat:
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

    @classmethod
    def resize(cls, img: cv2.Mat) -> cv2.Mat:
        return cv2.resize(
            img,
            cls.SIZE,
            interpolation=cv2.INTER_AREA,
        )

    @classmethod
    def find_faces(cls, img) -> Sequence:
        return cls.__FACE_CASCADE.detectMultiScale(
            img,
            minNeighbors=5,
            minSize=(32, 32),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

    @classmethod
    def obtain_classified_face(cls, path: str) -> cv2.Mat:
        _img: cv2.Mat = cls.load(path)
        faces = cls.find_faces(_img)
        if len(faces) <= 0:
            return cls.resize(_img)
        else:
            x, y, w, h = faces[0]
            return cls.resize(_img[y : y + h, x : x + w])
