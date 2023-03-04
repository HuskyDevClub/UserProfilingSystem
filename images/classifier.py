import os
import shutil
import time
from typing import Optional

import cv2  # type: ignore

from utils.user import User, Users

from .images import Images
from .model import ImageModels


class Classifier:
    CACHE_DIR: str = "image_sorted"

    @staticmethod
    def __copy_image_to(src: str, dst: str, mode: str):
        os.makedirs(dst, exist_ok=True)
        if mode == "default":
            shutil.copy2(src, dst)
        elif mode == "ideal":
            image: Optional[cv2.Mat] = Images.try_obtain_classified_face(src)
            if image is not None:
                cv2.imwrite(os.path.join(dst, os.path.basename(src)), image)
        elif mode == "greatest_square":
            cv2.imwrite(
                os.path.join(dst, os.path.basename(src)),
                Images.obtain_greatest_square(src),
            )

    @classmethod
    def classify(cls, _input: str, mode: str = "default"):
        # if a previous cache folder already exists, then remove it
        _out_dir: str = os.path.join(_input, cls.CACHE_DIR, mode)
        if os.path.exists(_out_dir):
            shutil.rmtree(_out_dir)
        os.makedirs(_out_dir)
        # create folder for all the target attributes
        for key in ImageModels.ALL_TARGET_ATTRIBUTES:
            attribute_out_dir: str = os.path.join(_input, cls.CACHE_DIR, mode, key)
            os.mkdir(attribute_out_dir)
        # load users database
        userProfileData: dict[str, User] = Users.load_database(
            os.path.join(_input, "profile", "profile.csv")
        )
        start_time: float = time.time()
        currentIndex: int = 0
        for _user in userProfileData.values():
            current_image_path: str = os.path.join(
                _input, "image", "{}.jpg".format(_user.get_id())
            )
            # classify by age
            cls.__copy_image_to(
                current_image_path, os.path.join(_out_dir, "age", "image"), mode
            )
            # classify by gender
            cls.__copy_image_to(
                current_image_path,
                os.path.join(_out_dir, "gender", _user.get_gender()),
                mode,
            )
            # classify by ocean
            for _ocean in ImageModels.OCEAN:
                cls.__copy_image_to(
                    current_image_path, os.path.join(_out_dir, _ocean, "image"), mode
                )
            # print current progress
            currentIndex += 1
            approximate_min_left: float = round(
                (time.time() - start_time)
                * (len(userProfileData) - currentIndex)
                / currentIndex
                / 60,
                1,
            )
            print(
                "Image classified: {0}/{1}, about {2} min left".format(
                    currentIndex, len(userProfileData), approximate_min_left
                ),
                end="\r",
                flush=True,
            )
