from os import path as OS_PATH
from typing import Optional, Sequence

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
        return cv2.imread(path)

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

    @classmethod
    def try_obtain_classified_face(cls, path: str) -> Optional[cv2.Mat]:
        _img: cv2.Mat = cls.load(path)
        # find faces
        faces = cls.find_faces(_img)
        if len(faces) >= 1:
            x, y, w, h = faces[0]
            return cls.resize(_img[y : y + h, x : x + w])
        else:
            return None

    @classmethod
    def obtain_images_for_prediction(
        cls, path: str
    ) -> tuple[cv2.Mat, Optional[cv2.Mat]]:
        _img: cv2.Mat = cls.load(path)
        # find faces
        faces = cls.find_faces(_img)
        if len(faces) >= 1:
            x, y, w, h = faces[0]
            return cls.__obtain_greatest_square(_img), cls.resize(
                _img[y : y + h, x : x + w]
            )
        else:
            return cls.__obtain_greatest_square(_img), None

    @classmethod
    def __obtain_greatest_square(cls, _img: cv2.Mat) -> cv2.Mat:
        _height: int = _img.shape[0]
        _width: int = _img.shape[1]
        if _height == _width:
            return cls.resize(_img)
        elif _height > _width:
            return cls.resize(_img[:_width, :_width])
        else:
            return cls.resize(_img[:_height, (_width - _height) // 2 : _height])

    @classmethod
    def obtain_greatest_square(cls, path: str) -> cv2.Mat:
        return cls.__obtain_greatest_square(cls.load(path))

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
        theImage = cls.load(path)
        result: list[cv2.Mat] = cls.__get_image_in_all_form(cls.resize(theImage))
        # find faces
        faces: Sequence = cls.find_faces(theImage)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            result.extend(
                cls.__get_image_in_all_form(cls.resize(theImage[y : y + h, x : x + w]))
            )
        # return result
        return result
