import gc
import os

import matplotlib.pyplot as plt  # type: ignore
from tensorflow.data import AUTOTUNE  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
from tensorflow.keras.utils import image_dataset_from_directory  # type: ignore

from .classifier import Classifier
from .images import Images
from .model import ImageModels


class TrainCnnImageModel:
    savefig: bool = False
    epochs: int = 10

    @classmethod
    def __train(cls, _input: str, _category: str):
        # load training data
        train_ds = image_dataset_from_directory(
            os.path.join(_input, Classifier.CACHE_DIR, _category),
            validation_split=0.4,
            subset="training",
            seed=123,
            image_size=Images.SIZE,
        )
        # load validation data
        val_ds = image_dataset_from_directory(
            os.path.join(_input, Classifier.CACHE_DIR, _category),
            validation_split=0.4,
            subset="validation",
            seed=123,
            image_size=Images.SIZE,
        )
        # load model
        _model = ImageModels.try_load_model(_category, len(train_ds.class_names))
        # prefetch data
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        # Model Checkpoint
        check_pointer = ModelCheckpoint(
            ImageModels.MODEL_WAS_SAVED_TO[_category],
            monitor="val_loss",
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
        result = _model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cls.epochs,
            callbacks=[check_pointer, early_stopping],
        )
        # show validation loss curve
        if cls.savefig is True:
            plt.clf()
            plt.plot(result.history["accuracy"], label="accuracy")
            plt.plot(result.history["val_accuracy"], label="val_accuracy")
            plt.xlabel("epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend(loc="lower right")
            plt.title(
                "validation loss curve for "
                + os.path.basename(ImageModels.MODEL_WAS_SAVED_TO[_category])
            )
            plt.savefig(
                ImageModels.MODEL_WAS_SAVED_TO[_category].replace(".h5", ".png")
            )
        # clear memory
        del _model
        gc.collect()

    @classmethod
    def train(cls, _input: str, ignore: list[str] = []):
        """
        start training
        """
        # train age
        for key in ImageModels.ALL_TARGET_ATTRIBUTES:
            if key not in ignore:
                cls.__train(_input, key)
