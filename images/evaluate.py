import csv
import os
from typing import Any, Optional
import numpy
from tensorflow.keras.models import load_model
from utils.user import User, Users
from .images import Images
from .model import ImageModel


class EvaluateImageModel:
    GENDER_MODEL = load_model(ImageModel.GENDER_MODEL_WAS_SAVED_TO)
    AGE_MODEL = load_model(ImageModel.AGE_MODEL_WAS_SAVED_TO)
    OCEAN_MODELS = {
        key: load_model(ImageModel.OCEAN_MODEL_WAS_SAVED_TO.format(key))
        for key in ImageModel.OCEAN
    }

    # predict the result
    @classmethod
    def predict(cls, _input: str, _user_ids: list[str]) -> list[User]:
        _in: numpy.ndarray = numpy.asarray(
            [
                numpy.asarray(
                    Images.obtain_classified_face(
                        os.path.join(_input, "image", "{}.jpg".format(_id))
                    )
                )
                for _id in _user_ids
            ]
        )
        gender_predictions = cls.GENDER_MODEL.predict(_in)
        gender_ranges = ["male", "female"]
        age_predictions = cls.AGE_MODEL.predict(_in)
        age_ranges = ["xx-24", "25-34", "35-49", "50-xx"]
        OCEAN_predictions = {}
        for key in cls.OCEAN_MODELS:
            OCEAN_predictions[key] = cls.OCEAN_MODELS[key].predict(_in)
        return [
            User(
                _user_ids[i],
                Users.convert_to_age(age_ranges[numpy.argmax(age_predictions[i])]),
                gender_ranges[numpy.argmax(gender_predictions[i])],
                numpy.argmax(OCEAN_predictions["extrovert"][i]),
                numpy.argmax(OCEAN_predictions["neurotic"][i]),
                numpy.argmax(OCEAN_predictions["agreeable"][i]),
                numpy.argmax(OCEAN_predictions["conscientious"][i]),
                numpy.argmax(OCEAN_predictions["open"][i]),
            )
            for i in range(len(_user_ids))
        ]

    # predict the result
    @classmethod
    def process(cls, _input: str, _output: str) -> None:
        # all the information types
        keys: Optional[tuple[str, ...]] = None

        ids: list[str] = []

        # read the csv file
        with open(os.path.join(_input, "profile", "profile.csv"), newline="") as f:
            spamreader = csv.reader(f, delimiter=" ", quotechar="|")

            # process data from csv file
            for row in spamreader:
                _temp: list[str] = row[0].split(",")
                if keys is not None:
                    userInfo: dict[str, Any] = {}
                    # assert len(keys) == len(_temp)
                    for i, _value in enumerate(keys):
                        userInfo[_value] = _temp[i]
                    ids.append(userInfo["userid"])
                else:
                    keys = tuple(_temp)

        for _user in cls.predict(_input, ids):
            _user.save(_output)
