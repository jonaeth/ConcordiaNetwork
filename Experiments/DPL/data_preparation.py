def set_initial_seed(seed, args):
    torch.manual_seed(seed)
    print(" use cuda: %d \n" % args.cuda)
    if args.cuda:
        torch.cuda.manual_seed(seed)

def get_word2vec(path_to_embeddings):
    if not path_to_embeddings:
        return None
    with open(path_to_embeddings, "rb") as fp:
        wordvec = pickle.load(fp)
    return wordvec


def get_vocabulary_wrapper(vocab_path):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


def load_pickle_data(file_path):
    with open(file_path, "rb") as fp:
        data = pickle.load(fp)
    return data


def get_data_balancing_weights(target):
    # make the data balance
    num_pos = float(sum(target[:, 0] <= target[:, 1]))
    num_neg = float(sum(target[:, 0] > target[:, 1]))
    mask_pos = (target[:, 0] <= target[:, 1]).cpu().float()
    mask_neg = (target[:, 0] > target[:, 1]).cpu().float()
    weight = mask_pos * (num_pos + num_neg) / num_pos
    weight += mask_neg * (num_pos + num_neg) / num_neg
    if args.cuda:
        weight = weight.cuda()

    return weight