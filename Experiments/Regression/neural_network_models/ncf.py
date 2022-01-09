import torch
import torch.nn as nn
import tempfile


class NCF(nn.Module):
    def __init__(self, **cfg):
        super(NCF, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors;
        num_layers: the number of layers in MLP model;
        dropout: dropout rate between fully connected layers;
        model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
        GMF_model: pre-trained GMF weights;
        MLP_model: pre-trained MLP weights.
        """
        self.dropout = cfg['dropout']
        self.model = cfg['model']

        num_users = cfg['num_users']
        num_items = cfg['num_items']
        self.embed_user_GMF = nn.Embedding(num_users, cfg['embedding_dimension'])
        self.embed_item_GMF = nn.Embedding(num_items, cfg['embedding_dimension'])
        self.embed_user_MLP = nn.Embedding(
                num_users, cfg['embedding_dimension'] * (2 ** (cfg['n_hidden_layers'] - 1)))
        self.embed_item_MLP = nn.Embedding(
                num_items, cfg['embedding_dimension'] * (2 ** (cfg['n_hidden_layers'] - 1)))

        MLP_modules = []
        for i in range(cfg['n_hidden_layers']):
            input_size = cfg['embedding_dimension'] * (2 ** (cfg['n_hidden_layers'] - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size//2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)
        predict_size = cfg['embedding_dimension']
        self.predict_layer = nn.Linear(cfg['distribution_size'], 1)

        self._init_weight_()

        self.early_stop_iterations = 0
        self.smallest_validation_loss = 99999
        self.temp_dir_for_model = tempfile.gettempdir()
        self.output_distribution_layer = torch.nn.Linear(predict_size, cfg['distribution_size'])


    def _init_weight_(self):
        """ We leave the weights initialization here. """
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight,
                                 a=1, nonlinearity='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, data_instance):
        users, items = data_instance[:, 0], data_instance[:, 1]
        embed_user_MLP = self.embed_user_MLP(users.int())
        embed_item_MLP = self.embed_item_MLP(items.int())
        interaction = torch.cat([embed_user_MLP, embed_item_MLP], dim=-1)
        output_MLP = self.MLP_layers(interaction)
        distribution_output, distribution_probabilities = self._forward_distribution_layer(output_MLP)
        prediction = self.predict_layer(distribution_probabilities)
        return prediction.squeeze(), distribution_output

    def _forward_distribution_layer(self, NN_output):
        distribution_output = self.output_distribution_layer(NN_output)
        distribution_probabilities = torch.relu(distribution_output) #dim=1
        return distribution_output, distribution_probabilities

    def compile(self, optimizer, optimizer_args, loss_function, loss_args):
        self.optimizer = optimizer(self.parameters(), **optimizer_args)
        self.loss_function = loss_function
        self.loss_args = loss_args

    def save_model(self, path_to_save_model):
        torch.save(self.state_dict(), path_to_save_model)

    def load_model(self, path_to_saved_model):
        self.load_state_dict(torch.load(path_to_saved_model))