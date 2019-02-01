from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from img_processing import bbox


scale_factor = 100

WIDTH = int(bbox['width']*scale_factor/100)
HEIGHT = int(bbox['height']*scale_factor/100)

LR = 1e-3
EPOCHS = 8
BATCH_SIZE = 10
MODEL_NAME = 'py-auto-sim-{}-{}-{}-epochs.model'.format(LR, 'Inception-v4', EPOCHS)

img_width, img_height = 256, 256
train_data_dir = "tf_files/codoon_photos"
validation_data_dir = "tf_files/codoon_photos"
nb_train_samples = 4125
nb_validation_samples = 466 
batch_size = 16
epochs = 50

model = applications.ResNet50(include_top=False, weights='imagenet', input_shape=(WIDTH, HEIGHT, 1))

x = model.output
predictions = Dense(2, activation="softmax")(x)

model_final = Model(inputs=model.input, outputs=predictions)

model_final.compile(loss = "categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

train_datagen = ImageDataGenerator(
  rescale = 1./255,
  horizontal_flip = True,
  fill_mode = "nearest",
  zoom_range = 0.3,
  width_shift_range = 0.3,
  height_shift_range=0.3,
  rotation_range=30)

test_datagen = ImageDataGenerator(
  rescale = 1./255,
  horizontal_flip = True,
  fill_mode = "nearest",
  zoom_range = 0.3,
  width_shift_range = 0.3,
  height_shift_range=0.3,
  rotation_range=30)

train_generator = train_datagen.flow_from_directory(
  train_data_dir,
  target_size = (img_height, img_width),
  batch_size = batch_size,
  class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
  validation_data_dir,
  target_size = (img_height, img_width),
  class_mode = "categorical")

checkpoint = ModelCheckpoint("resnet50_retrain.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

model_final.fit_generator(
  train_generator,
  samples_per_epoch = nb_train_samples,
  epochs = epochs,
  validation_data = validation_generator,
  nb_val_samples = nb_validation_samples,
  callbacks = [checkpoint, early])
