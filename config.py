supervision = 'supervised' # supervised/unsupervised/semi-supervised
use_teacher = True # If set to False, the network will behave like a standard neural network.
paths = {
    'psl_model': 'models/teacher/model.psl',
    'test_predicates': 'data/teacher/test_predicates.psl',
    'train_predicates': 'data/teacher/train_predicates.psl',
    'train_data': 'data/student/train/',
    'test_data': 'data/student/test/'
}
