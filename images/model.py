import os
from typing import Final
from tensorflow.keras import layers, models, losses  # type: ignore

from .images import Images


class ImageModels:
    GENDER_MODEL_WAS_SAVED_TO: Final[str] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "gender_model.h5"
    )
    AGE_MODEL_WAS_SAVED_TO: Final[str] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "age_model.h5"
    )
    OCEAN_MODEL_WAS_SAVED_TO: Final[str] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ocean_{}_model.h5"
    )
    OCEAN: tuple[str, ...] = (
        "open",
        "conscientious",
        "extrovert",
        "agreeable",
        "neurotic",
    )

    # try get model, if model does not exist, then create a new one
    @classmethod
    def get(cls, _path: str):
        # load model
        if os.path.exists(_path):
            # if model already exists, the continue to train
            model = models.load_model(_path)
        else:
            # build the model
            if _path == cls.GENDER_MODEL_WAS_SAVED_TO:
                model = ImageModels.get_gender_model()
            elif _path == cls.AGE_MODEL_WAS_SAVED_TO:
                model = ImageModels.get_age_model()
            else:
                model = ImageModels.get_ocean_model()
        model.summary()
        return model

    # credit:
    # https://www.tensorflow.org/tutorials/images
    @staticmethod
    def __get_model(output: layers.Dense):
        # create model
        model = models.Sequential()
        model.add(
            layers.Rescaling(
                1.0 / 255, input_shape=(Images.SIZE[0], Images.SIZE[1], 3)
            ),
        )
        # Data augmentation
        model.add(layers.RandomFlip("horizontal"))
        model.add(layers.RandomRotation(0.2))
        model.add(layers.RandomRotation(0.2))
        # hidden layers
        model.add(layers.Conv2D(16, 3, padding="same", activation="relu"))
        model.add(layers.MaxPooling2D())
        model.add(layers.Conv2D(32, 3, padding="same", activation="relu"))
        model.add(layers.MaxPooling2D())
        model.add(layers.Conv2D(64, 3, padding="same", activation="relu"))
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.2))
        model.add(layers.Flatten())
        # output layers
        model.add(layers.Dense(128, activation="relu"))
        model.add(output)
        # compile model
        model.compile(
            optimizer="adam",
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        return model

    @classmethod
    def get_gender_model(cls):
        return cls.__get_model(layers.Dense(2, activation="sigmoid"))

    @classmethod
    def get_age_model(cls):
        return cls.__get_model(layers.Dense(4, activation="softmax"))

    @classmethod
    def get_ocean_model(cls):
        return cls.__get_model(layers.Dense(6, activation="softmax"))
