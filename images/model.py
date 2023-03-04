import os
from typing import Final

from tensorflow.keras import layers, losses, models  # type: ignore

from .images import Images


class ImageModels:
    # the folder that all the models will be saved to
    MODEL_WAS_SAVE_TO_DIR: Final[str] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models"
    )
    if not os.path.exists(MODEL_WAS_SAVE_TO_DIR):
        os.mkdir(MODEL_WAS_SAVE_TO_DIR)
    # attributes
    OCEAN: Final[tuple[str, ...]] = (
        "open",
        "conscientious",
        "extrovert",
        "agreeable",
        "neurotic",
    )
    ALL_TARGET_ATTRIBUTES: Final[tuple[str, ...]] = tuple(
        ["age", "gender"] + list(OCEAN)
    )
    # classes
    GENDER_RANGES: Final[tuple[str, ...]] = tuple(sorted(["male", "female"]))
    AGE_RANGES: Final[tuple[str, ...]] = ("xx-24", "25-34", "35-49", "50-xx")

    # credit:
    # https://www.tensorflow.org/tutorials/images
    @staticmethod
    def __get_model(
        output: layers.Dense,
        _loss=losses.SparseCategoricalCrossentropy(),
        metrics: list[str] = ["accuracy"],
    ):
        # create model
        model = models.Sequential()
        # input layer
        model.add(
            layers.RandomFlip(
                "horizontal", input_shape=(Images.SIZE[0], Images.SIZE[1], 3)
            )
        )
        # normalize
        model.add(layers.RandomRotation(0.1))
        model.add(layers.Rescaling(1.0 / 255))
        # hidden layer 1
        model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(layers.MaxPooling2D())
        # hidden layer 2
        model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(layers.MaxPooling2D())
        # hidden layer 3
        model.add(layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
        model.add(layers.MaxPooling2D())
        # hidden layer 4
        model.add(layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
        model.add(layers.MaxPooling2D())
        # hidden layer 5
        model.add(layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.2))
        # flatten
        model.add(layers.Flatten())
        # output layers
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(128, activation="relu"))
        model.add(output)
        # compile model
        model.compile(optimizer="adam", loss=_loss, metrics=metrics)
        return model

    @classmethod
    def get_model(
        cls, category: str, mode: str, classNum: int, assumeExist: bool = False
    ):
        print("**************************************************")
        _path: str = os.path.join(
            cls.MODEL_WAS_SAVE_TO_DIR, "{0}_{1}_model.h5".format(category, mode)
        )
        if os.path.exists(_path):
            # if model already exists, the continue to train
            model = models.load_model(_path)
            print("An existing model is found and loaded!")
        elif not assumeExist:
            # generate a new model
            model = (
                cls.__get_model(layers.Dense(classNum), "mse", ["mse"])
                if classNum <= 1
                else cls.__get_model(
                    layers.Dense(
                        classNum, activation="sigmoid" if classNum <= 2 else "softmax"
                    )
                )
            )
            print("An new model is created!")
        else:
            raise FileNotFoundError("Cannot find model at path:", _path)
        print("**************************************************")
        model.summary()
        return model, _path

    @classmethod
    def try_load_model(cls, category: str, mode: str):
        _path: str = os.path.join(
            cls.MODEL_WAS_SAVE_TO_DIR, "{0}_{1}_model.h5".format(category, mode)
        )
        return models.load_model(_path) if os.path.exists(_path) else None
