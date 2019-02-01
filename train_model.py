import numpy as np
from img_processing import bbox, scale_factor
from keras.callbacks import TensorBoard
from time import time
from models import custom_cnn, alexnet
from keras.callbacks import Callback

class prediction_history(Callback):
    def __init__(self):
        self.history = []
    def on_epoch_end(self, epoch, logs={}):
        self.history.append(model.predict(X))

predictions = prediction_history()

WIDTH = int(bbox['width']*scale_factor/100)
HEIGHT = int(bbox['height']*scale_factor/100)

LR = 1e-5
EPOCHS = 50
BATCH_SIZE = 5
MODEL_NAME = 'pyAutoSim-{}-{}-{}-epochs.model'.format(LR, 'Alexnet', EPOCHS)

training_data = np.load('balanced_data.npy')

X = np.array([i[0] for i in training_data]).reshape(-1, WIDTH, HEIGHT, 1)
Y = np.array([i[1] for i in training_data])

print('Input image shape:', X[0].shape)

model = alexnet((WIDTH, HEIGHT), learning_rate=LR)
model.fit(X, Y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, shuffle=True, callbacks=[predictions])
print('training successful')

model.summary()
print(predictions.history[:1])


model.save(MODEL_NAME)

