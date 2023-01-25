import glob

import numpy
from parameters import MODEL_WAS_SAVED_TO
from tensorflow import keras

from images import Images

reconstructed_model = keras.models.load_model(MODEL_WAS_SAVED_TO)

_in = [
    numpy.asarray(Images.load(_path)) for _path in glob.glob("../training/image/*.jpg")
]

# _in.append(numpy.asarray(Images.load("../training/image/0a7f4d8ee46c4f7245038bdf45cd505f.jpg")))

_in = numpy.asarray(_in)

yhat = reconstructed_model.predict(_in)


print(yhat)
