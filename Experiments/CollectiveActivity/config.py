import time
import os


class Config(object):
    """
    class to save config parameter
    """
    def __init__(self, dataset_name):
        # Global
        self.image_size = 480, 720  # input image size
        self.batch_size = 1  # train batch size
        self.test_batch_size = 1  # test batch size
        self.num_boxes = 13  # max number of bounding boxes in each frame

        self.imitation_param_learning_method = 'decaying'  # ['constant', 'decaying']
        self.decay_rate = 0.05
        # self.imitation_param -= self.decay_rate
        self.initial_imitation_rate = 0.2

        # Gpu
        self.use_gpu = False
        self.use_multi_gpu = False
        self.device_list = "0,1"  # id list of gpus used for training # TODO

        # Dataset
        assert (dataset_name in ['collective'])
        self.dataset_name = dataset_name

        self.data_path = 'Experiments/CollectiveActivity/data/collective/ActivityDataset'  # data path for the collective dataset
        self.test_seqs = [5, 6, 7, 8, 9, 10, 11, 15, 16, 25, 28, 29, 52, 53, 59, 60, 64, 66, 68, 70, 71]
        self.train_seqs = [s for s in range(1, 73) if s not in self.test_seqs]

        # START: Original code by Zijian and Xinran
        # Backbone
        self.backbone = 'inv3'
        # self.backbone='mobilenet'
        # END: Original code by Zijian and Xinran

        self.crop_size = 5, 5  # crop size of roi align
        self.train_backbone = True  # if freeze the feature extraction part of network, True for stage 1, False for stage 2
        self.out_size = 57,87  # output feature map size of backbone
        self.emb_features = 1056  # output feature map channel of backbone

        # Activity Action
        self.num_actions = 9  # number of action categories
        self.num_activities = 8  # number of activity categories
        self.actions_loss_weight = 1.0  # weight used to balance action loss and activity loss
        self.actions_weights = None

        # Sample
        self.num_frames = 4
        self.num_before = 5
        self.num_after = 4

        # GCN
        self.num_features_boxes = 1024
        self.num_features_relation = 256
        self.num_graph = 16  # number of graphs
        self.num_features_gcn = self.num_features_boxes
        self.gcn_layers = 1  # number of GCN layers
        self.tau_sqrt = False
        self.pos_threshold = 0.2  # distance mask threshold in position relation

        # START: Original code by Zijian and Xinran
        self.appearance_calc = "DotProduct"  # based on our experiments, we suggest to use NCC or SAD to represent similarity relation graph instead of DotProduct
        # END: Original code by Zijian and Xinran

        # Training Parameters
        self.train_random_seed = 0
        self.train_learning_rate = 1e-5  # initial learning rate
        self.lr_plan = {41: 1e-4, 81: 5e-5, 121: 1e-5}  # change learning rate in these epochs
        self.train_dropout_prob = 0.5  # dropout probability
        self.weight_decay = 1e-2  # l2 weight decay

        self.max_epoch = 50  # max training epoch
        self.test_interval_epoch = 1

        # Exp
        self.training_stage = 1  # specify stage1 or stage2
        self.stage1_model_path = 'result/stage1_inv3_90.91%.pth'  # path of the base model, need to be set in stage2
        self.stage2_model_path = 'result/stage2_epoch84_90.65%.pth'  # path of the gcn model, need to be set in stage3
        self.test_before_train = False
        self.exp_note = 'Collective_train_' + self.backbone
        self.exp_name = None

        # Our experiments
        self.use_psl = False
        self.train_psl = False
        self.start_psl_at_epoch = 5

        self.remove_walking = False
        self.include_walking = True

        self.use_modified_same_predicate = False

        # START: Original code by Zijian and Xinran
        self.num_actions = 8 if self.include_walking else 7
        self.num_activities = 7 if self.include_walking else 6

        self.load_learned_psl = False
        self.psl_model_path = './'

        self.rule_25_doing_groundings = 'truth'
        self.use_predicates_doingTruth = False
        self.left_hand_side_local = True

        if self.include_walking:
            self.ACTIONS = ['NA', 'Crossing', 'Waiting', 'Queueing', 'Walking', 'Talking', 'Dancing', 'Jogging']
        else:
            self.ACTIONS = ['NA', 'Crossing', 'Waiting', 'Queueing', 'Talking', 'Dancing', 'Jogging']

    def set_other_config_options(self):
        if self.include_walking:
            self.ACTIONS = ['NA', 'Crossing', 'Waiting', 'Queueing', 'Walking', 'Talking', 'Dancing', 'Jogging']
        else:
            self.ACTIONS = ['NA', 'Crossing', 'Waiting', 'Queueing', 'Talking', 'Dancing', 'Jogging']

        # START: Original code by Zijian and Xinran
        self.num_actions = 8 if self.include_walking else 7
        self.num_activities = 7 if self.include_walking else 6

    def init_config(self, experiment_result_folder, need_new_folder=True):
        if self.exp_name is None:
            time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.exp_name = '/%s_stage%d/%s/' % (self.exp_note, self.training_stage, time_str)

        self.result_path = '%s/%s' % (experiment_result_folder, self.exp_name)
        self.log_path = '%s/%s/log.txt' % (experiment_result_folder, self.exp_name)

        if need_new_folder:
            os.makedirs(self.result_path)