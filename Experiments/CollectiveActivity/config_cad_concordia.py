import sys
sys.path.append(".")
from Experiments.CollectiveActivity.config import Config

cfg=Config('collective')
cfg.init_config('Experiments/CollectiveActivity/result/')

# cfg.train_backbone=True

# cfg.image_size=480, 720
# cfg.out_size=57,87
# cfg.num_boxes=13

# cfg.num_frames=4

# cfg.batch_size=1
# cfg.test_batch_size=1
# cfg.train_learning_rate=1e-5
# cfg.train_dropout_prob=0.5
# cfg.weight_decay=1e-2
cfg.lr_plan={}
# cfg.test_interval_epoch=1


# cfg.initial_imitation_rate=0.2
# cfg.imitation_param_learning_method = 'decaying'
# cfg.max_epoch=5
# cfg.use_gpu=True
# cfg.exp_note='Collective_train_' + cfg.backbone

# cfg.train_psl = True


# cfg.start_psl_at_epoch = 1

cfg.use_r4 = True

