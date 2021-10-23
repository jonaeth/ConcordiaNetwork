concordia_config = {
    'train_teacher': True,
    'teacher_training_starting_epoch': 4,
    'trained_teacher_path': None,
    'teacher_model_path': 'Experiments/CollectiveActivity/data/teacher/model',
    'ground_predicates_path': 'Experiments/CollectiveActivity/data/teacher/train',
    'psl_options': {
                'log4j.threshold': 'OFF',
                'votedperceptron.numsteps': '2'
            },
    'cli_options': None,
    'jvm_options': ['-Xms4096M', '-Xmx12000M']
}