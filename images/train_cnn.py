import gc
import os
from typing import Optional

import matplotlib.pyplot as plt  # type: ignore
import numpy
from sklearn.model_selection import train_test_split  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping  # type: ignore

from utils.user import User, Users

from .images import Images
from .model import ImageModels


class TrainCnnImageModel:
    savefig: bool = False
    epochs: int = 10

    @classmethod
    def __train(
        cls,
        pixels: numpy.ndarray,
        targets: numpy.ndarray,
        model_save_to: str,
    ):
        # load model
        model = ImageModels.get(model_save_to)
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
        # Model Early Stopping Rules
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=max(cls.epochs // 3, min(5, cls.epochs))
        )
        # Fit the model
        result = model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=cls.epochs,
            callbacks=[check_pointer, early_stopping],
        )
        # Final evaluation of the model
        scores = model.evaluate(x_test, y_test, verbose=0)
        print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))
        # show validation loss curve
        if cls.savefig is True:
            plt.clf()
            plt.plot(result.history["accuracy"], label="accuracy")
            plt.plot(result.history["val_accuracy"], label="val_accuracy")
            plt.xlabel("epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend(loc="lower right")
            plt.title("validation loss curve for " + os.path.basename(model_save_to))
            plt.savefig(model_save_to.replace(".h5", ".png"))
        # clear memory
        del model
        gc.collect()

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
        for key in ImageModels.OCEAN:
            targets[key] = []
        current_index: int = 0
        # load all data
        for _key, value in database.items():
            # image path
            _path: str = os.path.join(_input, "image", "{}.jpg".format(_key))
            # ensure path exists
            if not os.path.exists(_path):
                raise FileNotFoundError("Cannot find image", _path)
            # processing images
            _images: list = Images.obtain_training_images(_path)
            for _image in _images:
                pixels.append(numpy.asarray(_image))
            for _ in range(len(_images)):
                targets["gender"].append(0 if value.get_gender() == "male" else 1)
                targets["age"].append(value.get_age_group_index())
                targets["open"].append(round(value.get_open() * 2))
                targets["conscientious"].append(round(value.get_conscientious() * 2))
                targets["extrovert"].append(round(value.get_extrovert() * 2))
                targets["agreeable"].append(round(value.get_agreeable() * 2))
                targets["neurotic"].append(round(value.get_neurotic() * 2))
            current_index += 1
            print(
                "Image loaded: {0}/{1}".format(current_index, len(database)),
                end="\r",
                flush=True,
            )
            if _max is not None and current_index >= _max:
                break

        pixelsNdarray: numpy.ndarray = numpy.asarray(pixels)
        del pixels
        gc.collect()

        """
        start training
        """
        # train gender
        if "gender" not in ignore:
            cls.__train(
                pixelsNdarray,
                numpy.asarray(targets["gender"], numpy.uint16),
                ImageModels.GENDER_MODEL_WAS_SAVED_TO,
            )
            del targets["gender"]
        # train age
        if "age" not in ignore:
            cls.__train(
                pixelsNdarray,
                numpy.asarray(targets["age"], numpy.uint16),
                ImageModels.AGE_MODEL_WAS_SAVED_TO,
            )
            del targets["age"]
        # train ocean
        if "ocean" not in ignore:
            for key in ImageModels.OCEAN:
                if key not in ignore:
                    # train age
                    cls.__train(
                        pixelsNdarray,
                        numpy.asarray(targets[key], numpy.uint16),
                        ImageModels.OCEAN_MODEL_WAS_SAVED_TO.format(key),
                    )
