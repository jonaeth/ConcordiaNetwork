import sys
sys.path.append(".")
from Experiments.CollectiveActivity.config import Config

cfg=Config('collective')
cfg.init_config('Experiments/CollectiveActivity/result/')
cfg.training_stage=1
cfg.train_backbone=True

cfg.image_size=480, 720
cfg.out_size=57,87
cfg.num_boxes=13

# END: Original code by Zijian and Xinran

cfg.num_frames=4
cfg.use_multi_gpu = False

cfg.batch_size=1
cfg.test_batch_size=1
cfg.train_learning_rate=1e-5
cfg.train_dropout_prob=0.5
cfg.weight_decay=1e-2
cfg.lr_plan={}
cfg.test_interval_epoch=1
cfg.backbone = 'inv3'

#cfg.initial_imitation_rate = 0.4
#cfg.imitation_param_learning_method = 'constant'


cfg.initial_imitation_rate=0.2
cfg.decay_rate = 0.05
cfg.imitation_param_learning_method = 'decaying'
cfg.max_epoch=5
cfg.use_gpu=True
# START: Original code by Zijian and Xinran
cfg.exp_note='Collective_train_' + cfg.backbone
# END: Original code by Zijian and Xinran

cfg.use_psl = True
cfg.train_psl = True


cfg.remove_walking=False
cfg.include_walking=True

cfg.test_before_train = False
cfg.start_psl_at_epoch = 1
#cfg.use_

cfg.use_r4 = True

# START: Original code by Zijian and Xinran
cfg.num_actions = 8 if cfg.include_walking else 7
cfg.num_activities = 7 if cfg.include_walking else 6

cfg.rule_25_doing_groundings = 'truth'
cfg.left_hand_side_local = True


if cfg.include_walking:
    cfg.ACTIONS = ['NA', 'Crossing', 'Waiting', 'Queueing', 'Walking', 'Talking', 'Dancing', 'Jogging']
else:
    cfg.ACTIONS = ['NA', 'Crossing', 'Waiting', 'Queueing', 'Talking', 'Dancing', 'Jogging']