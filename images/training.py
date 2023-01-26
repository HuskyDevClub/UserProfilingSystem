import os
import sys

import numpy
from model import get_gender_model, get_age_model
from parameters import GENDER_MODEL_WAS_SAVED_TO, AGE_MODEL_WAS_SAVED_TO
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from images import Images

sys.path.append("..")
from utils.user import User, Users

sys.path.pop()

import argparse

# using argparse to parse the argument from command line
parser: argparse.ArgumentParser = argparse.ArgumentParser()
parser.add_argument("-m", help="what to train")
parser.add_argument("-e", help="epochs")
args: argparse.Namespace = parser.parse_args()

mode: str = args.m
_epochs: int = int(args.e)

# input path
ABS_PATH: str = os.path.join("..", "training")
# output path
if mode == "g":
    _out_path = GENDER_MODEL_WAS_SAVED_TO
elif mode == "a":
    _out_path = AGE_MODEL_WAS_SAVED_TO
else:
    raise Exception("Unknown mode!")

# database for storing user information, key is user id
database: dict[str, User] = Users.load_database(
    os.path.join(ABS_PATH, "profile", "profile.csv")
)

# Load dataset
pixels = []
targets = []
for _key in database:
    imageName: str = "{}.jpg".format(_key)
    _path: str = os.path.join(ABS_PATH, "image", imageName)
    if not os.path.exists(_path):
        raise FileNotFoundError("Cannot find image", _path)
    elif (_face := Images.obtain_classified_face(_path)) is not None:
        pixels.append(numpy.asarray(_face))
        if mode == "g":
            targets.append(
                numpy.array(0 if database[_key].get_gender() == "male" else 1)
            )
        else:
            targets.append(database[_key].get_age_group_index())

pixels = numpy.asarray(pixels)
targets = numpy.asarray(targets, numpy.uint16)

# Split dataset for training and test
x_train, x_test, y_train, y_test = train_test_split(pixels, targets, random_state=100)

if os.path.exists(_out_path):
    # if model already exists, the continue to train
    model = load_model(_out_path)
else:
    # build the model
    if mode == "g":
        model = get_gender_model()
    else:
        model = get_age_model()
    model.summary()

# Model Checkpoint
check_pointer = ModelCheckpoint(
    _out_path,
    monitor="loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
)
callback_list = [check_pointer]

# Fit the model
save = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=_epochs,
    callbacks=[callback_list],
)

# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))
