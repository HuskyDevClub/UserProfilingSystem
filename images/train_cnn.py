import gc
import os
from typing import Sequence

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
    def __train(cls, _input: str, _category: str, mode: str):
        # load data
        full_dataset = image_dataset_from_directory(
            os.path.join(_input, Classifier.CACHE_DIR, mode, _category),
            image_size=Images.SIZE,
        )
        DATASET_SIZE: int = full_dataset.cardinality().numpy()
        val_size = int(0.2 * DATASET_SIZE)
        train_size = DATASET_SIZE - val_size
        # get training data
        train_dataset = full_dataset.take(train_size)
        val_dataset = full_dataset.skip(train_size)
        # load model
        _model = ImageModels.try_load_model(_category, len(full_dataset.class_names))
        # prefetch data
        full_dataset.cache().prefetch(buffer_size=AUTOTUNE)
        # Model Checkpoint
        check_pointer = ModelCheckpoint(
            ImageModels.MODEL_WAS_SAVED_TO[_category],
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
        )
        # Model Early Stopping Rules
        early_stopping = EarlyStopping(
            monitor="accuracy", patience=max(cls.epochs // 3, min(5, cls.epochs))
        )
        # Fit the model
        result = _model.fit(
            train_dataset,
            validation_data=val_dataset,
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
    def train(cls, _input: str, ignore: Sequence[str] = [], mode: str = "default"):
        """
        start training
        """
        # train age
        for key in ImageModels.ALL_TARGET_ATTRIBUTES:
            if key not in ignore:
                cls.__train(_input, key, mode)
