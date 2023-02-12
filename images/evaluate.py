import numpy
from tensorflow import keras

from .images import Images
from .model import ImageModel

reconstructed_gender_model = keras.models.load_model(
    ImageModel.GENDER_MODEL_WAS_SAVED_TO
)
reconstructed_age_model = keras.models.load_model(ImageModel.AGE_MODEL_WAS_SAVED_TO)

_in = [
    # numpy.asarray(Images.load(_path)) for _path in glob.glob("../training/image/*.jpg")
]

_in.append(
    numpy.asarray(
        Images.obtain_classified_face(
            "../training/image/0a0b4d5d8fa6e298e2c3d531a5633534.jpg"
        )
    )
)

_in = numpy.asarray(_in)

gender_predictions = reconstructed_gender_model.predict(_in)
gender_ranges = ["male", "female"]

age_predictions = reconstructed_age_model.predict(_in)
age_ranges = ["xx-24", "25-34", "35-49", "50-xx"]

for i in range(len(_in)):
    print(gender_ranges[numpy.argmax(gender_predictions[i])])
    print(age_ranges[numpy.argmax(age_predictions[i])])
