from os import path as OS_PATH
from typing import Sequence

import cv2  # type: ignore


class Images:
    # face recognition classifier
    __FACE_CASCADE: cv2.CascadeClassifier = cv2.CascadeClassifier(
        OS_PATH.join(
            OS_PATH.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml"
        )
    )
    # image shape
    SIZE: tuple[int, int] = (128, 128)

    @staticmethod
    def load(path: str) -> cv2.Mat:
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    @classmethod
    def resize(cls, img: cv2.Mat) -> cv2.Mat:
        return cv2.resize(
            img,
            cls.SIZE,
            interpolation=cv2.INTER_AREA,
        )

    @classmethod
    def find_faces(cls, img: cv2.Mat) -> Sequence:
        return cls.__FACE_CASCADE.detectMultiScale(
            img,
            minNeighbors=5,
            minSize=(32, 32),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

    @classmethod
    def obtain_classified_face(
        cls, path: str, disableFaceFinding: bool = False
    ) -> cv2.Mat:
        _img: cv2.Mat = cls.load(path)
        if disableFaceFinding is True:
            return cls.resize(_img)
        # find faces
        faces = cls.find_faces(_img)
        if len(faces) <= 0:
            return cls.resize(_img)
        else:
            x, y, w, h = faces[0]
            return cls.resize(_img[y : y + h, x : x + w])

    @staticmethod
    def __get_image_rotations(img: cv2.Mat) -> list[cv2.Mat]:
        return [
            img,
            # cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
            # cv2.rotate(img, cv2.ROTATE_180),
            # cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]

    @classmethod
    def __get_image_in_all_form(cls, img: cv2.Mat) -> list[cv2.Mat]:
        return cls.__get_image_rotations(img) + cls.__get_image_rotations(
            cv2.flip(img, 0)
        )

    @classmethod
    def obtain_training_images(cls, path: str) -> list[cv2.Mat]:
        """
        theImage = cls.load(path)
        result: list[cv2.Mat] = cls.__get_image_in_all_form(cls.resize(theImage))
        # find faces
        faces: Sequence = cls.find_faces(theImage)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            result.extend(
                cls.__get_image_in_all_form(cls.resize(theImage[y : y + h, x : x + w]))
            )
        """
        result: list[cv2.Mat] = []
        result.append(cls.resize(cls.load(path)))
        # find faces
        faces: Sequence = cls.find_faces(result[0])
        if len(faces) > 0:
            x, y, w, h = faces[0]
            result.append(cls.resize(result[0][y : y + h, x : x + w]))
        # return result
        return result
