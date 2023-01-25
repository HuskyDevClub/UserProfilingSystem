import csv
import os
import sys
from typing import Any, Optional

import numpy
from model import get_model
from parameters import MODEL_WAS_SAVED_TO
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

from images import Images

sys.path.append("..")
from utils.user import User, Users

sys.path.pop()

# database for storing user information, key is user id
database: dict[str, User] = {}
# all the information types
keys: Optional[tuple[str, ...]] = None

# input path
ABS_PATH: str = os.path.join("..", "training")

profile_csv_location = os.path.join(ABS_PATH, "profile", "profile.csv")

print("Start reading data from location:", profile_csv_location)

with open(profile_csv_location, newline="") as f:
    # read the csv file
    spamreader = csv.reader(f, delimiter=" ", quotechar="|")
    for row in spamreader:
        _temp: list[str] = row[0].split(",")
        if keys is not None:
            userInfo: dict[str, Any] = {}
            assert len(keys) == len(_temp)
            for i, _value in enumerate(keys):
                userInfo[_value] = _temp[i]
            theUser = Users.from_dict(userInfo)
            database[theUser.get_id()] = theUser
        else:
            keys = tuple(_temp)

print("Finish reading data from location:", profile_csv_location)

# Load dataset
pixels = []
genders = []

for _key in database:
    pixels.append(
        numpy.asarray(
            Images.load(os.path.join(ABS_PATH, "image", "{}.jpg".format(_key)))
        )
    )
    genders.append(numpy.array(0 if database[_key].get_gender() == "male" else 1))

pixels = numpy.asarray(pixels)
genders = numpy.asarray(genders, numpy.uint64)

# Split dataset for training and test
x_train, x_test, y_train, y_test = train_test_split(pixels, genders, random_state=100)

# build the model
model = get_model()
model.summary()

# Model Checkpoint
check_pointer = ModelCheckpoint(
    MODEL_WAS_SAVED_TO,
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
    epochs=10,
    callbacks=[callback_list],
)

# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))
