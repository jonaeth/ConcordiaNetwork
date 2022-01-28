import torch

config_concordia = {
    ###################################################################
    # Concordia
    ###################################################################
    'use_teacher_inference_online': True,
    'train_teacher': False,
    'teacher_training_starting_epoch': 4,
    'gpu_device': torch.device('cuda', 0),
    'image_vector_length': 1280,
    'regression': False,
    ###################################################################
    # Teacher
    ###################################################################
    'teacher_model_path': 'Experiments/EntityLinking/teacher/model',
    'ground_predicates_path': 'Experiments/EntityLinking/teacher/train',
    'psl_options': {
        'log4j.threshold': 'OFF'
    },
    'cli_options': [],
    'jvm_options': ['-Xms4096M', '-Xmx12000M'],
    'teacher_student_distributions_comparison': [True, False],
    'markov_blanket_file': 'markov_blanket.psl',
    'predicates_folder': 'Experiments/EntityLinking/teacher/train',
    'supervision': 'unsupervised'  # 'semi-supervised', 'supervised'
}