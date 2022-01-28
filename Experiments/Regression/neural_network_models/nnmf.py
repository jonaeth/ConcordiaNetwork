import torch
import torch.nn
from tqdm import tqdm
import tempfile


class NNMF_PSL(torch.nn.Module):
    def __init__(self,  num_users, num_items, embedding_dimension, embedding_dimension_prime, n_dense_units,
                 n_hidden_layers, distribution_size):
        super(NNMF_PSL, self).__init__()
        self.user_embedding_layer = torch.nn.Embedding(num_users, embedding_dimension)
        self.item_embedding_layer = torch.nn.Embedding(num_items, embedding_dimension)

        self.latent_user_embedding_layer = torch.nn.Embedding(num_users, embedding_dimension_prime)
        self.latent_item_embedding_layer = torch.nn.Embedding(num_items, embedding_dimension_prime)

        MLP_input_layer = [torch.nn.Linear(embedding_dimension * 2 + embedding_dimension_prime, n_dense_units)]
        MLP_layers = MLP_input_layer + [torch.nn.Linear(n_dense_units, n_dense_units) for _ in range(n_hidden_layers - 1)]

        self.output_distribution_layer = torch.nn.Linear(n_dense_units, distribution_size)
        self.MLP_layers = torch.nn.ModuleList(MLP_layers)
        self.output_layer = torch.nn.Linear(distribution_size, 1)
        self._init_weights()
        self.early_stop_iterations = 0
        self.smallest_validation_loss = 99999
        self._init_weights()
        self.temp_dir_for_model = tempfile.gettempdir()

    def _init_weights(self):
        torch.nn.init.trunc_normal_(self.user_embedding_layer.weight, mean=0, std=0.1)
        torch.nn.init.trunc_normal_(self.item_embedding_layer.weight, mean=0, std=0.1)

        torch.nn.init.trunc_normal_(self.latent_user_embedding_layer.weight, mean=0, std=0.1)
        torch.nn.init.trunc_normal_(self.latent_item_embedding_layer.weight, mean=0, std=0.1)

        for layer in self.MLP_layers:
            torch.nn.init.xavier_uniform_(layer.weight, 4)
        torch.nn.init.xavier_uniform_(self.output_layer.weight, 4)
        torch.nn.init.xavier_uniform_(self.output_distribution_layer.weight, 4)

    def compile(self, optimizer, optimizer_args, loss_function, loss_args):
        embeddin_parameters = torch.nn.ParameterList()
        mlp_parameters = torch.nn.ParameterList()

        for name, param in self.named_parameters():
            if name == 'user_embedding_layer.weight' or name == 'item_embedding_layer.weight' or name == 'latent_user_embedding_layer.weight' or name == 'latent_item_embedding_layer.weight':
                embeddin_parameters.append(param)
            else:
                mlp_parameters.append(param)
        self.optimizer_latent = optimizer(embeddin_parameters, **optimizer_args)
        self.optimizer_mlp = optimizer(mlp_parameters, **optimizer_args)
        self.loss_function = loss_function
        self.loss_args = loss_args

    def forward(self, data_instance):
        users, items = data_instance[:, 0], data_instance[:, 1]
        MLP_input = self._get_latent_features_vector(users, items)
        MLP_output = self._forward_MLP(MLP_input)
        distribution_output, distribution_probabilities = self._forward_distribution_layer(MLP_output)
        output = self.output_layer(distribution_probabilities)
        return output.squeeze(), distribution_output

    def _get_latent_features_vector(self, users, items):
        factorized_layer = self.latent_user_embedding_layer(users.to(torch.long)) * self.latent_item_embedding_layer(items.to(torch.long))
        user_embeddings = self.user_embedding_layer(users.to(torch.long))
        item_embeddings = self.item_embedding_layer(items.to(torch.long))
        latent_features = torch.cat([user_embeddings, item_embeddings, factorized_layer], dim=-1)
        return latent_features

    def _forward_distribution_layer(self, NN_output):
        distribution_output = self.output_distribution_layer(NN_output)
        distribution_probabilities = torch.relu(distribution_output)
        return distribution_output, distribution_probabilities

    def _forward_MLP(self, MLP_input):
        layer_output = torch.sigmoid(self.MLP_layers[0](MLP_input))
        for hidden_layer in self.MLP_layers[1:]:
            layer_output = torch.sigmoid(hidden_layer(layer_output))
        return layer_output

    def fit(self, training_data, validation_data, test_data, epochs):
        metrics_history = []
        val_rmse_loss, val_mse_loss, val_mae_loss = self.compute_validation_loss(validation_data)
        for epoch in range(1, epochs+1):
            with tqdm(training_data, unit="batch") as tqdm_iterator:
                self.train()
                for iteration, (input_batch, target_batch, logic_distribution_target) in enumerate(tqdm_iterator):
                    tqdm_iterator.set_description(f"Epoch {epoch}")
                    target_predictions, logic_distribution_predictions = self.forward(input_batch)
                    loss = self.loss_function(self, target_predictions.float(), target_batch.float(),
                                              logic_distribution_predictions.float(), logic_distribution_target.float(),
                                              **self.loss_args)
                    self._optimize(loss)
                    tqdm_iterator.set_postfix(loss=loss.item(), VAL_RMSE_LOSS=val_rmse_loss, VAL_MAE_LOSS=val_mae_loss,
                                              VAL_MSE_LOSS=val_mse_loss)
                val_rmse_loss, val_mse_loss, val_mae_loss = self.compute_validation_loss(validation_data)
                if self.early_stop(val_rmse_loss) or epoch == epochs:
                    print('Final performance')
                    self.load_model(f'{self.temp_dir_for_model}/best_model.pkl')
                    test_rmse_loss, test_mse_loss, test_mae_loss = self.compute_validation_loss(test_data)
                    break
                tqdm_iterator.set_postfix(loss=loss.item(), VAL_RMSE_LOSS=val_rmse_loss, VAL_MAE_LOSS=val_mae_loss, VAL_MSE_LOSS=val_mse_loss)
                metrics_history.append({'rmse': val_rmse_loss, 'mae': val_mae_loss})
        print(f'Test RMSE: {test_rmse_loss}, TEST MAE: {test_mae_loss}, VAL_LOSS: {self.smallest_validation_loss}')
        return self.smallest_validation_loss


    def fit_different_speeds(self, training_data, validation_data, epochs):
        val_rmse_loss, val_mse_loss, val_mae_loss = self.compute_validation_loss(validation_data)
        for epoch in range(1, epochs+1):
            self.train()
            with tqdm(training_data, unit="batch") as tqdm_iterator:
                for iteration, (input_batch, target_batch, logic_distribution_target) in enumerate(training_data):
                    tqdm_iterator.set_description(f"Epoch {epoch}")
                    target_predictions, logic_distribution_predictions = self.forward(input_batch)
                    loss = self.loss_function(self, target_predictions.float(), target_batch.float(),
                                              logic_distribution_predictions.float(), logic_distribution_target.float(),
                                              **self.loss_args)
                    self._optimize(loss)
                    tqdm_iterator.set_postfix(loss=loss.item(), VAL_RMSE_LOSS=val_rmse_loss, VAL_MAE_LOSS=val_mae_loss, VAL_MSE_LOSS=val_mse_loss)


    def _optimize(self, loss):
        loss.backward()
        self.optimizer_mlp.step()
        self.optimizer_latent.step()
        self.optimizer_mlp.zero_grad()
        self.optimizer_latent.zero_grad()

    def save_model(self, path_to_save_model):
        torch.save(self.state_dict(), path_to_save_model)

    def load_model(self, path_to_saved_model):
        self.load_state_dict(torch.load(path_to_saved_model))

    def early_stop(self, curr_loss):
        if self.smallest_validation_loss > curr_loss:
            self.smallest_validation_loss = curr_loss
            self.save_model(f'{self.temp_dir_for_model}/best_model.pkl')
            self.early_stop_iterations = 0
        else:
            self.early_stop_iterations += 1
        return self.early_stop_iterations > 50