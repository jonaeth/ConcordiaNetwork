import argparse
import torch

def load_arguments():
    # Training settings with the pytorch
    parser = argparse.ArgumentParser(description='entities disambuity with PyTorch (rnn settings)')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=6, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--word_embedding', type=str,
                        default='Experiments/DPL/data/pubmed_parsed/embedding_vec_gene.pkl',
                        help='initial word embedding file')
    parser.add_argument('--classifier_type', type=str, default="rnn",
                        help='the classifier type')
    parser.add_argument('--windowSize', type=int, default=5,
                        help='the window size')
    parser.add_argument('--dataroot', type=str,
                        default='Experiments/DPL/data/pubmed_parsed',
                        help='the data root')
    parser.add_argument('--train_data', type=str, default='chunk_train_41_gene.pkl',
                        help='train data')
    parser.add_argument('--val_data', type=str, default='validation_gene_1_soft.pkl',
                        help='val data')
    parser.add_argument('--test_data', type=str, default='test_gene_1_soft.pkl',
                        help='test data')
    parser.add_argument('--vocab_path', type=str,
                        default="Experiments/DPL/data/pubmed_parsed/vocab_gene.pkl",
                        help='the vocab path')
    parser.add_argument('--embed_size', type=int, default=200,
                        help='the initial word embedding size')
    parser.add_argument('--fix_embed', type=bool, default=False,
                        help='whether fix the embedding or not')
    parser.add_argument('--nThreads', type=int, default=1,
                        help='number of thread for the data reading')
    parser.add_argument('--entity_type', type=str, default='gene',
                        help='the current entity type we are trained on')
    parser.add_argument('--initial_model', type=str, default='',
                        help='the current entity type we are trained on')
    parser.add_argument('--save_path', type=str,
                        default='Experiments/DPL/model/model.pkl',
                        help='the current entity type we are trained on')
    parser.add_argument('--visulization_html', type=str, default='./result/mlp_vis.html',
                        help='the html that can write')
    parser.add_argument('--combine', type=str, default='concatenate',
                        help='how to combine the wordvector: mean, max, product, concatenate, maxpool')
    parser.add_argument('--confidence_html', type=str, default='./result/confidence.html',
                        help='display the confidence for each prediction')
    parser.add_argument('--gene_key', type=str,
                        default='Experiments/DPL/data/gene_key.pkl',
                        help='display the confidence for each prediction')
    parser.add_argument('--window_size', type=int, default=5,
                        help='lstm or rnn')
    parser.add_argument('--class_label', type=int, default=2,
                        help='output label number')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='hidden_size of rnn')
    parser.add_argument('--num_layer', type=int, default=1,
                        help='output label number')
    parser.add_argument('--cell', type=str, default="lstm",
                        help='lstm or rnn')
    parser.add_argument('--max_confidence_instance', type=int, default=200,
                        help='maximum positive or negative example')

    # adjust the threshold for evaluation
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='adjust the threshold for evaluation')

    # different optimization method
    parser.add_argument('--hard_em', help='whether use hard em', type=bool, default=False)
    parser.add_argument('--stochastic', help='whether use the incremental EM or not', type=bool, default=False)
    parser.add_argument('--multiple_M', help='whether train multiple M step', type=int, default=2)
    parser.add_argument('--learn_graph', help='whether learn the graph parameters or not', action='store_true',
                        default=False)
    parser.add_argument('--learn_rate_graph', help='learning rate of graph parameters', type=float, default=0.01)
    parser.add_argument('--graph_regularizer', help='regularizer of graph parameters', type=float, default=1e-4)

    # E step or M step, split the task into different machine
    parser.add_argument('--stage', help='E step or M step', type=str, default="E")
    parser.add_argument('--prediction_file', help='prediction file', type=str, default="")

    parser.add_argument('--tune_threshold', help='whether to tune threshold', type=bool, default=True)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
