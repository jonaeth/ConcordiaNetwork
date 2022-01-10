import time
time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

neural_network_config = {
    'num_users': 1,
    'num_items': 1,
    'embedding_dimension': 64,
    'n_hidden_layers': 3,
    'model': 'MLP',
    'dropout': 0.3,
    'distribution_size': 10
}


optimiser_config = {
    'lr': 0.0005,
    'batch_size': 1,
    'l2_lambda': 0.1,
}

concordia_config = {
    'gpu_device': False,
    'train_online': False,
    'regression': True,
    'teacher_student_distributions_comparison': [True],
    'log_path': f'Experiments/RecommendationsMovieLens/logs/log_{time_str}.txt'
}