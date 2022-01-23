import torch


def l2_regularization_term(model, l2_lambda):
    loss_reg = 0
    for name, params in model.named_parameters():
        loss_reg += torch.square(torch.norm(params, 2)).sum()
    return l2_lambda * loss_reg

def RMSE_LOSS(y_pred, y_true):
    loss_fn = torch.nn.MSELoss()
    return torch.sqrt(loss_fn(y_pred, y_true.reshape(-1)))

def custom_RMSE_LOSS(y_pred, y_true):
    return RMSE_LOSS(y_pred[0], y_true)


def L2_RMSE_LOSS(model, y_pred, y_true, l2_lambda):
    mse_loss = RMSE_LOSS(y_pred, y_true)
    if not model.training:
        return mse_loss
    return mse_loss + l2_regularization_term(model, l2_lambda)