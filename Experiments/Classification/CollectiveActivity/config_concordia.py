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
    'teacher_model_path': 'Experiments/CollectiveActivity/teacher/model',
    'ground_predicates_path': 'Experiments/CollectiveActivity/teacher/train',
    'psl_options': {
        'log4j.threshold': 'OFF',
        'votedperceptron.numsteps': '2'
    },
    'cli_options': [],
    'jvm_options': ['-Xms4096M', '-Xmx12000M'],
    'teacher_student_distributions_comparison': [True, False],
    'predicates_folder': 'Experiments/CollectiveActivity/teacher/train'
}