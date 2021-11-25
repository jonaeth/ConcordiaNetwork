from abc import ABC

# TODO Modestas: implement all necessary callbacks,
#  think about creating a utils folder for this and torch_losses and similar files
class Callback(ABC):
    def __init__(self):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass