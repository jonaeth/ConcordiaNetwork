from Concordia.Callback import Callback


class CollectiveActivityCallback(Callback):
    def __init__(self):
        super(Callback).__init__()

    def on_epoch_end(self, epoch, logs=None):
        total_loss = logs['total_loss']
        activities_acc = logs['activities_acc']

        train_info = {
            'epoch': epoch,
            'loss': logs['total_loss'],
            'activities_acc': logs['activities_acc'],
            'actions_acc': logs['actions_acc'],
            'log_sigma_NN': logs['log_sigma_NN'],
            'log_sigma_PSL': logs['log_sigma_PSL']
        }