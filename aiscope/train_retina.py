import warnings
import keras
from detdata import DetGen
from detdata.augmenters import crazy_augmenter, dummy
from keras_retinanet.bin.train import create_models
from keras_retinanet.models.resnet import download_imagenet, resnet_retinanet as retinanet
from keras_retinanet.preprocessing.detgen import DetDataGenerator
from aiscope.callbacks import LossHistory
import logging

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

train_generator = DetGen('/home/i008/malaria_data/dataset_train.mxrecords',
                         '/home/i008/malaria_data/dataset_train.csv',
                         '/home/i008/malaria_data/dataset_train.mxindex', batch_size=4)

val_generator = DetGen('/home/i008/malaria_data/dataset_valid.mxrecords',
                       '/home/i008/malaria_data/dataset_valid.csv',
                       '/home/i008/malaria_data/dataset_valid.mxindex', batch_size=4)

train_generator = DetDataGenerator(train_generator, augmenter=crazy_augmenter)
val_generator = DetDataGenerator(val_generator, augmenter=dummy)

val_generator.image_max_side = 750
val_generator.image_min_side = 750
train_generator.image_max_side = 750
train_generator.image_min_side = 750

weights = download_imagenet('resnet50')

model_checkpoint = keras.callbacks.ModelCheckpoint(
    '/home/i008/googledrive/trained_models/mod-{epoch:02d}_loss-{loss:.4f}.h5',
    monitor='loss',
    verbose=2,
    save_best_only=False,
    save_weights_only=False,
    mode='auto',
    period=1)

callbacks = []

callbacks.append(keras.callbacks.ReduceLROnPlateau(
    monitor='loss',
    factor=0.1,
    patience=2,
    verbose=1,
    mode='auto',
    epsilon=0.0001,
    cooldown=0,
    min_lr=0
))

callbacks.append(LossHistory())
callbacks.append(model_checkpoint)

model, training_model, prediction_model = create_models(
    backbone_retinanet=retinanet,
    backbone='resnet50',
    num_classes=train_generator.num_classes(),
    weights=weights,
    multi_gpu=0,
    freeze_backbone=True
)

training_model.fit_generator(
    generator=train_generator,
    verbose=2,
    steps_per_epoch=2000,
    epochs=40,
    callbacks=callbacks,
    validation_data=val_generator,
    validation_steps=200
)
