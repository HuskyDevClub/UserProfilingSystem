import glob
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
    __start_time: float = 0.0
    __currentIndex: int = 0

    @classmethod
    def __reset_counter(cls) -> None:
        cls.__currentIndex = 0
        cls.__start_time = time.time()

    @classmethod
    def __update_counter(cls, _length: int) -> None:
        # update the counter
        cls.__currentIndex += 1
        approximate_min_left: float = round(
            (time.time() - cls.__start_time)
            * (_length - cls.__currentIndex)
            / cls.__currentIndex
            / 60,
            1,
        )
        # print current progress
        print(
            "Image classified: {0}/{1}, about {2} min left".format(
                cls.__currentIndex, _length, approximate_min_left
            ),
            end="\r",
            flush=True,
        )

    @staticmethod
    def __copy_image_to(
        src: str, dst: str, mode: str, disable_ideal_face: bool = False
    ) -> None:
        os.makedirs(dst, exist_ok=True)
        if mode == "default" or (mode == "ideal" and disable_ideal_face is True):
            shutil.copy2(src, dst)
        elif mode == "ideal":
            image: Optional[cv2.Mat] = Images.try_obtain_classified_face(src)
            if image is not None:
                cv2.imwrite(os.path.join(dst, os.path.basename(src)), image)
        elif mode == "greatest_square":
            try:
                cv2.imwrite(
                    os.path.join(dst, os.path.basename(src)),
                    Images.obtain_greatest_square(src),
                )
            except Exception:
                print("Cannot process image:", src, "- will be removed")
                os.remove(src)

    @classmethod
    def classify(cls, _input: str, mode: str = "default") -> None:
        # output folder path
        _out_dir: str = os.path.join(_input, cls.CACHE_DIR, mode)
        # if a previous cache folder already exists, then remove it
        if os.path.exists(_out_dir):
            shutil.rmtree(_out_dir)
        # and create a new one
        os.makedirs(_out_dir)
        # create folder for all the target attributes
        for key in ImageModels.ALL_TARGET_ATTRIBUTES:
            os.mkdir(os.path.join(_input, cls.CACHE_DIR, mode, key))
        # load users database
        userProfileData: dict[str, User] = Users.load_database(
            os.path.join(_input, "profile", "profile.csv")
        )
        cls.__reset_counter()
        for _user in userProfileData.values():
            current_image_path: str = os.path.join(
                _input, "image", "{}.jpg".format(_user.get_id())
            )
            # classify by age
            cls.__copy_image_to(
                current_image_path,
                os.path.join(_out_dir, "age", _user.get_age_group()),
                mode,
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
            cls.__update_counter(len(userProfileData))

    @classmethod
    def utf_face_summary(cls, _input: str) -> None:
        result: dict[str, dict[str, dict[str, int]]] = {}
        for _mode_folder in glob.glob(os.path.join(_input, "image_UTKFACE", "*")):
            _mode: str = os.path.basename(_mode_folder)
            result[_mode] = {
                "gender": {"male": 0, "female": 0},
                "age_group": {"xx-24": 0, "25-34": 0, "35-49": 0, "50-xx": 0},
            }
            for _file in glob.glob(os.path.join(_mode_folder, "*.jpg")):
                labels = os.path.basename(_file).split("_")
                result[_mode]["gender"][
                    "male" if int(labels[1]) == 0 else "female"
                ] += 1
                result[_mode]["age_group"][Users.convert_age_group(int(labels[0]))] += 1
        print(result)

    @classmethod
    def classify_utf_face(cls, _input: str, mode: str) -> None:
        # output folder path
        _out_dir: str = os.path.join(_input, cls.CACHE_DIR, mode)
        _images: list[str] = glob.glob(
            os.path.join(_input, "image_UTKFACE", mode, "*.jpg")
        )
        cls.__reset_counter()
        for _image in _images:
            labels = os.path.basename(_image).split("_")
            # classify by age
            cls.__copy_image_to(
                _image,
                os.path.join(_out_dir, "age", Users.convert_age_group(int(labels[0]))),
                mode,
                True,
            )
            # classify by gender
            cls.__copy_image_to(
                _image,
                os.path.join(
                    _out_dir, "gender", "male" if int(labels[1]) == 0 else "female"
                ),
                mode,
                True,
            )
            # print current progress
            cls.__update_counter(len(_images))
