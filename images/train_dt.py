import os
from typing import Optional

import numpy
import pandas  # type: ignore
import tensorflow_decision_forests as tfdf  # type: ignore

from utils.user import User, Users

from .images import Images
from .model import ImageModels


class TrainImageDecisionTree:
    SAVED_TO: str = os.path.join(ImageModels.MODEL_WAS_SAVE_TO_DIR, "dt")

    @classmethod
    def train(cls, _input: str, _max: Optional[int] = None) -> None:
        # database for storing user information, key is user id
        database: dict[str, User] = Users.load_database(
            os.path.join(_input, "profile", "profile.csv")
        )
        # Load dataset
        inputs: dict[str, list] = {
            "scale": [],  # 0 for square, 1 for rectangle but width is longer, and 2 is vise versa
            "faces": [],  # number of faces found, 0 for no face, 1 for faces found
            "gender": [],  # gender that is predicted
            "age": [],  # age group that is predicted
            "size": [],  # 0 is smaller than 10 kb and 1 is else
        }
        # init dictionary
        targets: dict[str, list[int]] = {key: [] for key in ImageModels.OCEAN}

        SIZE_FRESH_HOLD: int = 10 * 1024

        current_index: int = 0
        # load all data
        for _key, value in database.items():
            # image path
            _path: str = os.path.join(_input, "image", "{}.jpg".format(_key))
            # ensure path exists
            if not os.path.exists(_path):
                raise FileNotFoundError("Cannot find image", _path)
            # processing image
            _image = Images.load(_path)
            _imageND: numpy.ndarray = numpy.asarray(Images.load(_path))
            width, height, color = _imageND.shape
            inputs["scale"].append(0 if width == height else 1 if width > height else 2)
            inputs["faces"].append(0 if len(Images.find_faces(_image)) <= 0 else 1 if len(Images.find_faces(_image)) == 1 else 2)
            inputs["gender"].append(0 if value.get_gender() == "male" else 1)
            inputs["age"].append(value.get_age_group_index())
            inputs["size"].append(0 if os.path.getsize(_path) > SIZE_FRESH_HOLD else 1)
            targets["open"].append(
                round(value.get_open() * ImageModels.OCEAN_SCORE_AMPLIFY_SCALE)
            )
            targets["conscientious"].append(
                round(value.get_conscientious() * ImageModels.OCEAN_SCORE_AMPLIFY_SCALE)
            )
            targets["extrovert"].append(
                round(value.get_extrovert() * ImageModels.OCEAN_SCORE_AMPLIFY_SCALE)
            )
            targets["agreeable"].append(
                round(value.get_agreeable() * ImageModels.OCEAN_SCORE_AMPLIFY_SCALE)
            )
            targets["neurotic"].append(
                round(value.get_neurotic() * ImageModels.OCEAN_SCORE_AMPLIFY_SCALE)
            )
            if _max is not None and current_index >= _max:
                break
            current_index += 1
            print(
                "Image loaded: {0}/{1}".format(current_index, len(database)),
                end="\r",
                flush=True,
            )

        """
        start training
        """
        # train ocean
        for key in ImageModels.OCEAN:
            # Convert the pandas dataframe into a TensorFlow dataset
            input_df: pandas.DataFrame = pandas.DataFrame(data=inputs, dtype=numpy.int8)
            input_df[key] = targets[key]
            # Reading input data
            split_point = len(targets[key]) * 4 // 5
            train_df = input_df.iloc[:split_point]
            test_df = input_df.iloc[split_point:]
            # Convert the pandas dataframe into a TensorFlow dataset
            train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label=key)
            test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label=key)
            # Generate the model
            model = tfdf.keras.RandomForestModel()
            # Train the model
            model.fit(train_ds)
            model.compile(metrics=["accuracy"])
            # Evaluate the model
            model.evaluate(test_ds, return_dict=True)
            # Save the model
            model.save(os.path.join(cls.SAVED_TO, key))
