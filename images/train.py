import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

from utils.user import User, Users

from .images import Images
from .model import ImageModel


class TrainImageModel:
    savefig: bool = False
    epochs: int = 30

    @classmethod
    def __train(
        cls,
        pixels: numpy.ndarray,
        targets: numpy.ndarray,
        model_save_to: str,
    ):
        # load model
        model = ImageModel.get(model_save_to)
        # Split dataset for training and test
        x_train, x_test, y_train, y_test = train_test_split(
            pixels, targets, random_state=100
        )
        print("Start training! The trained model will be saved to:", model_save_to)
        # Model Checkpoint
        check_pointer = ModelCheckpoint(
            model_save_to,
            monitor="loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
        )
        callback_list = [check_pointer]
        # Fit the model
        result = model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=cls.epochs,
            callbacks=[callback_list],
        )
        # Final evaluation of the model
        scores = model.evaluate(x_test, y_test, verbose=0)
        print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))
        # show validation loss curve
        if cls.savefig is True:
            plt.plot(result.history["val_loss"], label="val_loss")
            plt.xlabel("epochs")
            plt.title("validation loss curve for " + model_save_to)
            plt.savefig(model_save_to.replace(".h5", ".png"))

    @classmethod
    def train(
        cls,
        _input: str,
        ignore: list[str] = [],
        _max: Optional[int] = None,
    ):
        # database for storing user information, key is user id
        database: dict[str, User] = Users.load_database(
            os.path.join(_input, "profile", "profile.csv")
        )

        # Load dataset
        pixels: list = []
        # init dictionary
        targets: dict[str, list[int]] = {
            "gender": [],
            "age": [],
        }
        for key in ImageModel.OCEAN:
            targets[key] = []
        # load all data
        for _key, value in database.items():
            imageName: str = "{}.jpg".format(_key)
            _path: str = os.path.join(_input, "image", imageName)
            if not os.path.exists(_path):
                raise FileNotFoundError("Cannot find image", _path)
            elif (_face := Images.obtain_classified_face(_path)) is not None:
                pixels.append(numpy.asarray(_face))
                targets["gender"].append(
                    numpy.array(0 if value.get_gender() == "male" else 1)
                )
                targets["age"].append(value.get_age_group_index())
                targets["open"].append(round(value.get_open()))
                targets["conscientious"].append(round(value.get_conscientious()))
                targets["extrovert"].append(round(value.get_extrovert()))
                targets["agreeable"].append(round(value.get_agreeable()))
                targets["neurotic"].append(round(value.get_neurotic()))
            if _max is not None and len(pixels) >= _max:
                break

        """
        start training
        """
        # train gender
        if "gender" not in ignore:
            cls.__train(
                numpy.asarray(pixels),
                numpy.asarray(targets["gender"], numpy.uint16),
                ImageModel.GENDER_MODEL_WAS_SAVED_TO,
            )
        # train age
        if "age" not in ignore:
            cls.__train(
                numpy.asarray(pixels),
                numpy.asarray(targets["age"], numpy.uint16),
                ImageModel.AGE_MODEL_WAS_SAVED_TO,
            )
        # train ocean
        for key in ImageModel.OCEAN:
            if key not in ignore:
                # train age
                cls.__train(
                    numpy.asarray(pixels),
                    numpy.asarray(targets[key], numpy.uint16),
                    ImageModel.OCEAN_MODEL_WAS_SAVED_TO.format(key),
                )
