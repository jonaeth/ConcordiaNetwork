from abc import ABC
from torch.optim import Adam


class Student(ABC):
    def __init__(self, neural_network, optimizer=None):
        self.model = neural_network
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = Adam(self.model.parameters(), lr=0.001)

    def build_model(self, predicate_file, rules_file):
        pass

    def __str__(self):
        pass

    def write_model_to_file(self, file_name):
        pass

    def fit(self, loss):
        pass

    def predict(self):
        pass


class ClassificationStudent(Student):
    def __init__(self, neural_network, optimizer):
        super().__init__(neural_network, optimizer)

    def fit(self, loss):
        pass

    def predict(self):
        pass


class RegressionStudent(Student):
    def __init__(self):
        super().__init__()

    def fit(self, loss):
        pass

    def predict(self):
        pass

