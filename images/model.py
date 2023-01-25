from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Model
from tensorflow.keras.layers import Activation, Input
from tensorflow.keras.regularizers import l2

from images import Images

# credit:
# https://medium.com/@skillcate/gender-detection-model-using-cnn-a-complete-guide-279706e94fdb
def get_model():
    # create model
    _input = Input(shape=(Images.SIZE[0], Images.SIZE[1], 1))
    conv1 = Conv2D(
        32, (3, 3), padding="same", strides=(1, 1), kernel_regularizer=l2(0.001)
    )(_input)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Activation("relu")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(
        64, (3, 3), padding="same", strides=(1, 1), kernel_regularizer=l2(0.001)
    )(pool1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Activation("relu")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(
        128, (3, 3), padding="same", strides=(1, 1), kernel_regularizer=l2(0.001)
    )(pool2)
    conv3 = Dropout(0.1)(conv3)
    conv3 = Activation("relu")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(
        256, (3, 3), padding="same", strides=(1, 1), kernel_regularizer=l2(0.001)
    )(pool3)
    conv4 = Dropout(0.1)(conv4)
    conv4 = Activation("relu")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    flatten = Flatten()(pool4)
    dense_1 = Dense(128, activation="relu")(flatten)
    drop_1 = Dropout(0.2)(dense_1)
    output = Dense(2, activation="sigmoid")(drop_1)
    # Model compile
    model = Model(inputs=_input, outputs=output)
    model.compile(
        optimizer="adam", loss=["sparse_categorical_crossentropy"], metrics=["accuracy"]
    )

    return model
