import cv2


class Images:

    # image shape
    SIZE: tuple[int, int] = (128, 128)

    @classmethod
    def load(cls, path: str):
        return cv2.resize(
            cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY),
            cls.SIZE,
            interpolation=cv2.INTER_AREA,
        )
