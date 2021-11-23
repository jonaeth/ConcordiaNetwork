import torch

config_concordia = {
    ###################################################################
    # Concordia
    ###################################################################
    'use_teacher_inference_online': True,
    'train_teacher': False,
    'teacher_training_starting_epoch': 4,
    'gpu_device': torch.device('cpu', 0),
    'image_vector_length': 1280,
    ###################################################################
    # Teacher
    ###################################################################
    'teacher_model_path': 'Experiments/CollectiveActivity/data/teacher/model',
    'ground_predicates_path': 'Experiments/CollectiveActivity/data/teacher/train',
    'psl_options': {
        'log4j.threshold': 'OFF',
        'votedperceptron.numsteps': '2'
    },
    'cli_options': [],
    'jvm_options': ['-Xms4096M', '-Xmx12000M'],
    'teacher_student_distributions_comparison': [True, False]
}