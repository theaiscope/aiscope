import keras
from deepsense import neptune

ctx = neptune.Context()


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        ctx.channel_send("loss", logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        ctx.channel_send("val_loss", logs.get('val_loss'))
