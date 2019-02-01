import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam, SGD
from keras.applications import InceptionV3


def custom_cnn(image_size=None, weights_path=None, learning_rate=1e-4):

    input_shape = image_size + (1, )
    input_layer = Input(shape=input_shape)

    network = Convolution2D(filters=32, kernel_size=3, strides=3, activation='relu')(input_layer)
    network = MaxPooling2D(pool_size=(2, 2), strides=2)(network)
    network = Dropout(rate=0.1)(network)
    network = Convolution2D(filters=64, kernel_size=3, strides=3, activation='relu')(input_layer)
    network = MaxPooling2D(pool_size=(2, 2), strides=2)(network)
    network = Dropout(rate=0.1)(network)
    network = Convolution2D(filters=128, kernel_size=3, strides=3, activation='relu')(input_layer)
    network = MaxPooling2D(pool_size=(2, 2), strides=2)(network)
    network = Dropout(rate=0.1)(network) 
    network = Flatten()(network)
    network = Dense(units=1024, activation='relu')(network)
    network = Dropout(rate=0.2)(network)
    network = Dense(units=3, activation='softmax')(network)

    model = Model(inputs=input_layer, outputs=network)
    model.compile(optimizer=SGD(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    if weights_path:
        model.load_weights(weights_path)

    return model

def alexnet(image_size=None, weights_path=None, learning_rate=1e-4):
    input_shape = image_size + (1, )
    input_layer = Input(shape=input_shape)

    network = Convolution2D(filters=96, kernel_size=11, strides=4, activation='relu', padding="same")(input_layer)
    network = MaxPooling2D(pool_size=3, strides=2, padding="same")(network)
    network = BatchNormalization()(network)
    network = Dropout(rate=0.2)(network)
    network = Convolution2D(filters=256, kernel_size=5, activation='relu', padding="same")(network)
    network = MaxPooling2D(pool_size=3, strides=2, padding="same")(network)
    network = BatchNormalization()(network)
    network = Dropout(rate=0.2)(network)
    network = Convolution2D(filters=384, kernel_size=3, activation='relu', padding="same")(network)
    network = Convolution2D(filters=384, kernel_size=3, activation='relu', padding="same")(network)
    network = Convolution2D(filters=256, kernel_size=3, activation='relu', padding="same")(network)
    network = MaxPooling2D(pool_size=3, strides=2, padding="same")(network)
    network = BatchNormalization()(network)
    network = Dropout(rate=0.2)(network)
    network = Flatten()(network)
    network = Dense(units=4096, activation='tanh')(network)
    network = Dropout(rate=0.5)(network)
    network = Dense(units=4096, activation='tanh')(network)
    network = Dropout(rate=0.5)(network)
    network = Dense(units=3, activation='softmax')(network)

    model = Model(inputs=input_layer, outputs=network)
    model.compile(optimizer=SGD(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    return model