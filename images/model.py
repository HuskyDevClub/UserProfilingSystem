import os
from typing import Final

from keras.initializers import RandomNormal  # type: ignore
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
    # the path to cnn models
    MODEL_WAS_SAVED_TO: Final[dict[str, str]] = {
        "gender": os.path.join(MODEL_WAS_SAVE_TO_DIR, "gender_model.h5"),
        "age": os.path.join(MODEL_WAS_SAVE_TO_DIR, "age_model.h5"),
    }
    for _ocean_attribute in OCEAN:
        MODEL_WAS_SAVED_TO[_ocean_attribute] = os.path.join(
            MODEL_WAS_SAVE_TO_DIR, "{}_model.h5".format(_ocean_attribute)
        )
    # classes
    GENDER_RANGES: Final[tuple[str, ...]] = ("male", "female")
    AGE_RANGES: Final[tuple[str, ...]] = ("xx-24", "25-34", "35-49", "50-xx")

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
        model.add(layers.RandomZoom(0.1))
        model.add(layers.Rescaling(1.0 / 255))
        # hidden layer 1
        model.add(
            layers.Conv2D(
                64,
                (3, 3),
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.05),
                bias_initializer=RandomNormal(mean=0.0, stddev=0.05),
                activation="relu",
            )
        )
        model.add(layers.MaxPooling2D())
        # hidden layer 2
        model.add(
            layers.Conv2D(
                64,
                (3, 3),
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.05),
                bias_initializer=RandomNormal(mean=0.0, stddev=0.05),
                activation="relu",
            )
        )
        model.add(layers.MaxPooling2D(2, 2))
        # hidden layer 3
        model.add(
            layers.Conv2D(
                128,
                (3, 3),
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.05),
                bias_initializer=RandomNormal(mean=0.0, stddev=0.05),
                activation="relu",
            )
        )
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Dropout(0.2))
        # hidden layer 4
        model.add(
            layers.Conv2D(
                128,
                (3, 3),
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.05),
                bias_initializer=RandomNormal(mean=0.0, stddev=0.05),
                activation="relu",
            )
        )
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Dropout(0.2))
        # hidden layer 5
        model.add(
            layers.Conv2D(
                256,
                (3, 3),
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.05),
                bias_initializer=RandomNormal(mean=0.0, stddev=0.05),
                activation="relu",
            )
        )
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Dropout(0.2))
        # hidden layer 6
        model.add(
            layers.Conv2D(
                256,
                (3, 3),
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.05),
                bias_initializer=RandomNormal(mean=0.0, stddev=0.05),
                activation="relu",
            )
        )
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Dropout(0.2))
        # flatten
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
    def get_model(cls, classNum: int):
        return cls.__get_model(
            layers.Dense(classNum, activation="sigmoid" if classNum <= 2 else "softmax")
        )

    @classmethod
    def try_load_model(cls, category: str, classNum: int):
        if os.path.exists(cls.MODEL_WAS_SAVED_TO[category]):
            # if model already exists, the continue to train
            print("An existing model is found and will be loaded!")
            model = models.load_model(cls.MODEL_WAS_SAVED_TO[category])
        else:
            # generate a new model
            model = cls.get_model(classNum)
        model.summary()
        return model
