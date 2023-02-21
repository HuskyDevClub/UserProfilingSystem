import csv
import gc
import os
from typing import Any, Optional
import numpy

# from images.train_dt import TrainImageDecisionTree
from tensorflow.keras import models  # type: ignore
from utils.user import User, Users
from .images import Images
from .model import ImageModels


class EvaluateImageModel:

    """
    OCEAN_DT_MODELS = {
        key: load_model(os.path.join(TrainImageDecisionTree.SAVED_TO, key))
        for key in ImageModels.OCEAN
    }
    """

    @staticmethod
    def __get_gender_model():
        return models.load_model(ImageModels.GENDER_MODEL_WAS_SAVED_TO)

    @staticmethod
    def __get_age_model():
        return models.load_model(ImageModels.AGE_MODEL_WAS_SAVED_TO)

    @staticmethod
    def __get_ocean_model(_type: str):
        return models.load_model(ImageModels.OCEAN_MODEL_WAS_SAVED_TO.format(_type))

    # predict the result
    @classmethod
    def predict(cls, _input: str, _user_ids: list[str]) -> list[User]:
        _in: numpy.ndarray = numpy.asarray(
            [
                numpy.asarray(
                    Images.obtain_classified_face(
                        os.path.join(_input, "image", "{}.jpg".format(_id)), True
                    )
                )
                for _id in _user_ids
            ]
        )
        gender_predictions: numpy.ndarray = cls.__get_gender_model().predict(_in)
        age_predictions: numpy.ndarray = cls.__get_age_model().predict(_in)
        gc.collect()
        OCEAN_predictions: dict[str, numpy.ndarray] = {}
        for key in ImageModels.OCEAN:
            OCEAN_predictions[key] = cls.__get_ocean_model(key).predict(_in)
            gc.collect()
        results: list[User] = []
        for i in range(len(_user_ids)):
            results.append(
                User(
                    _user_ids[i],
                    Users.convert_to_age(
                        ImageModels.AGE_RANGES[numpy.argmax(age_predictions[i])]
                    ),
                    ImageModels.GENDER_RANGES[numpy.argmax(gender_predictions[i])],
                    round(int(numpy.argmax(OCEAN_predictions["extrovert"][i])) / 2, 1),
                    round(int(numpy.argmax(OCEAN_predictions["neurotic"][i])) / 2, 1),
                    round(int(numpy.argmax(OCEAN_predictions["agreeable"][i])) / 2, 1),
                    round(
                        int(numpy.argmax(OCEAN_predictions["conscientious"][i])) / 2, 1
                    ),
                    round(int(numpy.argmax(OCEAN_predictions["open"][i])) / 2, 1),
                )
            )
            assert (
                results[i].get_age_group()
                == ImageModels.AGE_RANGES[numpy.argmax(age_predictions[i])]
            )
        return results

    # predict the result
    @classmethod
    def process(cls, _input: str, _output: str) -> None:
        # database for storing user information, key is user id
        database: dict[str, User] = Users.load_database(
            os.path.join(_input, "profile", "profile.csv")
        )
        for _user in cls.predict(_input, list(database.keys())):
            _user.save(_output)
