from mlp_data_loader import BasicDataLoader
from rnn_data_loader import RnnDataLoader


def CreateDataLoader(classifier_type, file_path, vocab, window_size, batch_size, valid_token=None):
    if classifier_type == 'mlp':
        data_loader = BasicDataLoader(file_path, vocab, window_size, valid_token, batch_size)
    else:
        data_loader = RnnDataLoader(file_path, vocab, batch_size)

    print(data_loader.name())
    return data_loader