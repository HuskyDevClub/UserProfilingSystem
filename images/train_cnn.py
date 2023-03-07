import gc
import os
from typing import Sequence

import matplotlib.pyplot as plt  # type: ignore
from tensorflow.data import AUTOTUNE  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
from tensorflow.keras.utils import image_dataset_from_directory  # type: ignore
from utils.user import Users  # type: ignore

from .classifier import Classifier
from .images import Images
from .model import ImageModels


class TrainCnnImageModel:
    savefig: bool = False
    epochs: int = 30
    monitor: str | None = None
    stop_early: bool = False

    @classmethod
    def __train(cls, _input: str, _category: str, mode: str):
        if _category == "gender" or _category == "age":
            # load data
            full_dataset = image_dataset_from_directory(
                os.path.join(_input, Classifier.CACHE_DIR, mode, _category),
                image_size=Images.SIZE,
            )
            # load model
            _model, _path = ImageModels.get_model(
                _category, mode, len(full_dataset.class_names)
            )
            # default monitor for classify task
            _monitor = "val_accuracy" if cls.monitor is None else cls.monitor
        else:
            _path = os.path.join(_input, Classifier.CACHE_DIR, mode, _category)
            _files = []
            for root, dirs, files in os.walk(os.path.join(_path, "image")):
                _files = files
                break
            database = Users.load_database(
                os.path.join(_input, "profile", "profile.csv")
            )
            # load data
            full_dataset = image_dataset_from_directory(
                _path,
                image_size=Images.SIZE,
                labels=[
                    round(
                        database[os.path.splitext(_path)[0]].get_ocean(_category)
                        * ImageModels.OCEAN_SCORE_AMPLIFY_SCALE
                    )
                    for _path in _files
                ],
                label_mode="int",
            )
            # load model
            _model, _path = ImageModels.get_model(_category, mode, 1)
            # default monitor for linear regression task
            _monitor = "val_loss" if cls.monitor is None else cls.monitor
        # split data
        DATASET_SIZE: int = full_dataset.cardinality().numpy()
        val_size = DATASET_SIZE // 4
        train_size = DATASET_SIZE - val_size
        # training data
        train_dataset = full_dataset.take(train_size)
        # validation data
        val_dataset = full_dataset.skip(train_size)
        # prefetch data
        full_dataset.cache().prefetch(buffer_size=AUTOTUNE)
        # Model Checkpoint
        check_pointer = ModelCheckpoint(
            _path,
            monitor=_monitor,
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
        )
        # Model Early Stopping Rules
        early_stopping = EarlyStopping(
            monitor=_monitor, patience=max(cls.epochs // 3, min(5, cls.epochs))
        )
        _callbacks = [check_pointer]
        if cls.stop_early is True:
            _callbacks.append(early_stopping)
        # Fit the model
        result = _model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=cls.epochs,
            callbacks=_callbacks,
        )
        # show validation loss curve
        if cls.savefig is True:
            plt.clf()
            if _category == "gender" or _category == "age":
                plt.plot(result.history["accuracy"], label="accuracy")
                plt.plot(result.history["val_accuracy"], label="val_accuracy")
            plt.plot(result.history["loss"], label="loss")
            plt.plot(result.history["val_loss"], label="val_loss")
            plt.xlabel("epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend(loc="lower right")
            plt.title("validation loss curve for {0}_{1}".format(_category, mode))
            plt.savefig(_path.replace(".h5", ".png"))
        # clear memory
        del _model
        gc.collect()

    @classmethod
    def train(cls, _input: str, ignore: Sequence[str] = [], mode: str = "default"):
        """
        start training
        """
        # train age
        for key in ImageModels.ALL_TARGET_ATTRIBUTES:
            if key not in ignore:
                cls.__train(_input, key, mode)
