import gc
import os

import numpy
import pandas  # type: ignore

# from images.train_dt import TrainImageDecisionTree
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
    def __start_predicting(
        category: str, idealIn: numpy.ndarray, greatestSquareIn: numpy.ndarray
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        ideal_model = ImageModels.try_load_model(category, "ideal")
        ideal_predictions: numpy.ndarray = (
            ideal_model.predict(idealIn)
            if ideal_model is not None
            else numpy.asarray([])
        )
        greatest_square_predictions: numpy.ndarray = ImageModels.get_model(
            category, "greatest_square", 2, True
        )[0].predict(greatestSquareIn)
        return ideal_predictions, greatest_square_predictions

    # predict the result
    @classmethod
    def predict(cls, _input: str, _user_ids: list[str]) -> list[User]:
        # load images into memory
        pixels_ideal_in_temp: list[numpy.ndarray] = []
        user_has_ideal_image_temp: list[str] = []
        pixel_greatest_square_in_temp: list[numpy.ndarray] = []
        for _id in _user_ids:
            imgGreatestSquare, imgIdeal = Images.obtain_images_for_prediction(
                os.path.join(_input, "image", "{}.jpg".format(_id))
            )
            pixel_greatest_square_in_temp.append(imgGreatestSquare)
            if imgIdeal is not None:
                pixels_ideal_in_temp.append(imgIdeal)
                user_has_ideal_image_temp.append(_id)
        user_has_ideal_image: tuple[str, ...] = tuple(user_has_ideal_image_temp)
        pixels_ideal: numpy.ndarray = numpy.asarray(pixels_ideal_in_temp)
        pixel_greatest_square: numpy.ndarray = numpy.asarray(
            pixel_greatest_square_in_temp
        )
        del (
            user_has_ideal_image_temp,
            pixels_ideal_in_temp,
            pixel_greatest_square_in_temp,
        )
        gc.collect()
        predictions: dict[str, numpy.ndarray] = {}
        for key in ImageModels.ALL_TARGET_ATTRIBUTES:
            (
                predictions[key + "_ideal"],
                predictions[key + "_greatest_square"],
            ) = cls.__start_predicting(key, pixels_ideal, pixel_greatest_square)
            gc.collect()
        results: list[User] = []
        currentUserIdealImageIndex: int = 0
        for i in range(len(_user_ids)):
            userId: str = _user_ids[i]
            _prob: dict[str, numpy.ndarray] = {}
            for key in ImageModels.ALL_TARGET_ATTRIBUTES:
                _prob[key] = predictions[key + "_greatest_square"][i]
            if userId == user_has_ideal_image[currentUserIdealImageIndex]:
                for key in ImageModels.ALL_TARGET_ATTRIBUTES:
                    if len(ideal_predictions := predictions[key + "_ideal"]) > 0:
                        _prob[key] = ideal_predictions[currentUserIdealImageIndex]
                currentUserIdealImageIndex += 1
            results.append(
                User(
                    userId,
                    Users.convert_to_age(
                        ImageModels.AGE_RANGES[numpy.argmax(_prob["age"])]
                    ),
                    ImageModels.GENDER_RANGES[numpy.argmax(_prob["gender"])],
                    round(
                        float(_prob["extrovert"])
                        / ImageModels.OCEAN_SCORE_AMPLIFY_SCALE,
                        2,
                    ),
                    round(
                        float(_prob["neurotic"])
                        / ImageModels.OCEAN_SCORE_AMPLIFY_SCALE,
                        2,
                    ),
                    round(
                        float(_prob["agreeable"])
                        / ImageModels.OCEAN_SCORE_AMPLIFY_SCALE,
                        2,
                    ),
                    round(
                        float(_prob["conscientious"])
                        / ImageModels.OCEAN_SCORE_AMPLIFY_SCALE,
                        2,
                    ),
                    round(
                        float(_prob["open"]) / ImageModels.OCEAN_SCORE_AMPLIFY_SCALE, 2
                    ),
                )
            )
        assert currentUserIdealImageIndex == len(user_has_ideal_image)
        return results

    # predict the result
    @classmethod
    def process(cls, _input: str, _output: str, _o_type: str) -> None:
        # database for storing user information, key is user id
        database: dict[str, User] = Users.load_database(
            os.path.join(_input, "profile", "profile.csv")
        )
        _results: list[User] = cls.predict(_input, list(database.keys()))
        if _o_type == "xml":
            for _user in _results:
                _user.save(_output)
        else:
            _df: pandas.DataFrame | None = None
            for _user in _results:
                _df = (
                    pandas.concat([_df, _user.get_data_frame()], ignore_index=True)
                    if _df is not None
                    else _user.get_data_frame()
                )
            assert _df is not None
            _df.to_csv(os.path.join(_output, "image_out.csv"))
