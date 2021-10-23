def show_config(cfg):
    print_log(cfg.log_path, '=====================Config=====================')
    for k,v in cfg.__dict__.items():
        print_log(cfg.log_path, k,': ',v)
    print_log(cfg.log_path, '======================End=======================')


def print_log(file_path,*args):
    print(*args)
    if file_path is not None:
        with open(file_path, 'a') as f:
            print(*args,file=f)


def show_epoch_info(phase, log_path, info):
    print_log(log_path, '')
    if phase == 'Test':
        print_log(log_path, '====> %s at epoch #%d' % (phase, info['epoch']))
    else:
        print_log(log_path, '%s at epoch #%d' % (phase, info['epoch']))

    print_log(log_path,
              'Group Activity Accuracy: %.2f%%, Individual Actions Accuracy: %.2f%%, Loss: %.5f, cross_entropy_loss_individual_actions: %.5f, '
              'kl_divergnece_loss_psl_nn_predictions: %.5f, soft_cross_entropy_loss_psl_nn_predictions: %.5f, Using %.1f seconds' % (
                  info['activities_acc'], info['actions_acc'], info['loss'],
                  info['cross_entropy_loss_individual_actions'],
                  info['kl_divergnece_loss_psl_nn_predictions'], info['soft_cross_entropy_loss_psl_nn_predictions'],
                  info['time']))