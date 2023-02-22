import os
from typing import Final
from tensorflow.keras import layers, models, losses  # type: ignore

from .images import Images


class ImageModels:
    # the folder that all the models will be saved to
    MODEL_WAS_SAVE_TO_DIR: Final[str] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models"
    )
    if not os.path.exists(MODEL_WAS_SAVE_TO_DIR):
        os.mkdir(MODEL_WAS_SAVE_TO_DIR)
    # the path to cnn models
    GENDER_MODEL_WAS_SAVED_TO: Final[str] = os.path.join(
        MODEL_WAS_SAVE_TO_DIR, "gender_model.h5"
    )
    AGE_MODEL_WAS_SAVED_TO: Final[str] = os.path.join(
        MODEL_WAS_SAVE_TO_DIR, "age_model.h5"
    )
    OCEAN_MODEL_WAS_SAVED_TO: Final[str] = os.path.join(
        MODEL_WAS_SAVE_TO_DIR, "ocean_{}_model.h5"
    )
    OCEAN: Final[tuple[str, ...]] = (
        "open",
        "conscientious",
        "extrovert",
        "agreeable",
        "neurotic",
    )
    GENDER_RANGES: Final[tuple[str, ...]] = ("male", "female")
    AGE_RANGES: Final[tuple[str, ...]] = ("xx-24", "25-34", "35-49", "50-xx")

    # try get model, if model does not exist, then create a new one
    @classmethod
    def get(cls, _path: str):
        # load model
        if os.path.exists(_path):
            # if model already exists, the continue to train
            print("An existing model is found and will be loaded!")
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
        # input layer
        model.add(
            layers.RandomFlip(
                "horizontal", input_shape=(Images.SIZE[0], Images.SIZE[1], 3)
            )
        )
        model.add(layers.RandomRotation(0.2))
        model.add(layers.Rescaling(1.0 / 255))
        # hidden layer 1
        model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
        model.add(layers.MaxPooling2D(3, 3))
        # hidden layer 2
        model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(layers.MaxPooling2D(3, 3))
        # hidden layer 3
        model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(layers.MaxPooling2D(3, 3))
        model.add(layers.Dropout(0.05))
        # hidden layer 4
        model.add(layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
        model.add(layers.MaxPooling2D(3, 3))
        model.add(layers.Dropout(0.05))
        # flatten
        model.add(layers.Flatten())
        # output layers
        model.add(layers.Dense(128, activation="relu"))
        model.add(output)
        # compile model
        model.compile(
            optimizer="sgd",
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        return model

    @classmethod
    def get_gender_model(cls):
        return cls.__get_model(
            layers.Dense(len(cls.GENDER_RANGES), activation="sigmoid")
        )

    @classmethod
    def get_age_model(cls):
        return cls.__get_model(layers.Dense(len(cls.AGE_RANGES), activation="softmax"))

    @classmethod
    def get_ocean_model(cls):
        return cls.__get_model(layers.Dense(11, activation="softmax"))
