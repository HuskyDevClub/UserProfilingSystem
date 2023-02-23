import os
import shutil

from utils.user import User, Users

from .model import ImageModels


class Classifier:
    CACHE_DIR: str = "image_sorted"

    @staticmethod
    def __copy_image_to(src: str, dst: str):
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(src, dst)

    @classmethod
    def classify(cls, _input: str):
        # if a previous cache folder already exists, then remove it
        _out_dir: str = os.path.join(_input, cls.CACHE_DIR)
        if os.path.exists(_out_dir):
            shutil.rmtree(_out_dir)
        os.mkdir(_out_dir)
        # create folder for all the target attributes
        for key in ImageModels.ALL_TARGET_ATTRIBUTES:
            attribute_out_dir: str = os.path.join(_input, cls.CACHE_DIR, key)
            os.mkdir(attribute_out_dir)
        # load users database
        userProfileData: dict[str, User] = Users.load_database(
            os.path.join(_input, "profile", "profile.csv")
        )
        currentIndex: int = 0
        for _user in userProfileData.values():
            current_image_path: str = os.path.join(
                _input, "image", "{}.jpg".format(_user.get_id())
            )
            # classify by age
            cls.__copy_image_to(
                current_image_path,
                os.path.join(_out_dir, "age", _user.get_age_group()),
            )
            # classify by gender
            cls.__copy_image_to(
                current_image_path,
                os.path.join(_out_dir, "gender", _user.get_gender()),
            )
            # classify by ocean
            for _ocean in ImageModels.OCEAN:
                cls.__copy_image_to(
                    current_image_path,
                    os.path.join(
                        _out_dir,
                        _ocean,
                        str(round(_user.get_ocean(_ocean) * 2)),
                    ),
                )
            # print current progress
            currentIndex += 1
            print(
                "Image classified: {0}/{1}".format(currentIndex, len(userProfileData)),
                end="\r",
                flush=True,
            )
