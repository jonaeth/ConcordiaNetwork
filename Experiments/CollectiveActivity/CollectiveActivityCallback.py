from Concordia.Callback import Callback
from Experiments.CollectiveActivity.logging import *


class CollectiveActivityCallback(Callback):
    def __init__(self, log_path):
        super(Callback).__init__()
        self.log_path = log_path

    def on_epoch_end(self, epoch, logs=None):
        train_info = {
            'epoch': epoch,
            'loss': logs['loss'],
            'activities_acc': logs['activities_acc'],
            'actions_acc': logs['actions_acc']
        }
        show_epoch_info(logs['evaluation_step'], self.log_path, train_info)