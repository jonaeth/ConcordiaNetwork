import torch.utils.data
from base_data_loader import BaseDataLoader
from text_folder_basic import TxtFolder_MLP

class BasicDataLoader:
    def __init__(self, file_path, vocab, window_size, batch_size, valid_token=None):
        dataset = TxtFolder_MLP(file_name=file_path, vocab=vocab, window_size=window_size, valid_token=valid_token)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,#=int(self.opt.nThreads),
            drop_last=True
        )
        self.dataset = dataset
        self.mlp_data = data_loader

    def name(self):
        return 'MLPDataLoader'

    def load_data(self):
        return self.mlp_data

    def __len__(self):
        return len(self.dataset)
