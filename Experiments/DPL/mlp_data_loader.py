import torch.utils.data
from base_data_loader import BaseDataLoader
from text_folder_basic import TxtFolder_MLP

class BasicDataLoader(BaseDataLoader):
    
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        dataset = TxtFolder_MLP(file_name = opt.file_path, vocab = opt.vocab, window_size=opt.windowSize, valid_token=opt.valid_token)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=int(self.opt.nThreads),
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