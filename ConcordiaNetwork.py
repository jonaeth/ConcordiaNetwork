import types
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader


class ConcordiaNetwork:
    def __init__(self, student, teacher, predicate_builder, loss_function, teacher_offline):
        self.loss_function = loss_function
        self.student = student
        self.teacher = teacher
        self.predicate_builder = predicate_builder
        self.teacher_offline = teacher_offline

    def fit(self, inputs, targets=None):
        self.student.model.train()
        if self.teacher_offline:
            self._fit_teacher_offline(inputs, targets)
        else:
            self._fit_teacher_online(inputs, targets)

    def _fit_teacher_online(self, inputs, targets=None):
        data_loader = self._create_tensor_dataloader(inputs, targets)
        for input, target in data_loader:
            student_prediction = self.student.predict(input)
            self.predicate_builder.build_predicates(input, student_prediction)
            self.teacher.fit()
            teacher_prediction = self.teacher.predict()
            loss = self.compute_loss(student_prediction, teacher_prediction, target)
            self.student.fit(loss)

    def _fit_teacher_offline(self, inputs, targets=None):
        self.teacher.fit()
        teacher_predictions = self.teacher.predict()
        data_loader = self._create_tensor_dataloader(teacher_predictions, batch_size=12)
        for input, target, teacher_prediction in data_loader:
            student_prediction = self.student.predict(input)
            loss = self.compute_loss(student_prediction, teacher_prediction, target)
            self.student.fit(loss)

    def predict(self):
        pass

    def compute_loss(self, student_predictions, teacher_predictions, target_values):
        return self.loss_function(student_predictions, teacher_predictions) + 0

    def _create_tensor_dataloader_for_neural_network(self, input_data, target_data, batch_size):
        input_data = torch.tensor(input_data)
        target_data = torch.tensor(target_data)
        dataset = TensorDataset(input_data, target_data)
        sampler = RandomSampler(dataset)

        loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return loader

    def _create_tensor_dataloader(self, teacher_predictions, batch_size):
        teacher_predictions = torch.tensor(teacher_predictions)
        teacher_predictions = TensorDataset(teacher_predictions)
        return DataLoader(teacher_predictions, batch_size=batch_size)


class ConcordiaDataLoader:
    def __init__(self, x, y=None, teacher_predictions=None):
        self.x = x
        self.y = y
        self.teacher_predictions = teacher_predictions

    def _create_generator(self):
        if type(self.x):
            pass

    def __next__(self):
        pass

'''
def compute_losses(self, predictions, targets):
    predicted_action_scores, predicted_activitie_scores = predictions[0], predictions[1]
    target_action_scores, target_group_activity_scores = targets[0], targets[1]
    target_action_scores = torch.cat(target_action_scores, dim=0).reshape(-1, )
    target_group_activity_scores = target_group_activity_scores.reshape(-1, )
    individual_actions_loss = F.cross_entropy(predicted_action_scores, target_action_scores, weight=None)
    group_activity_loss = F.cross_entropy(predicted_activitie_scores, target_group_activity_scores)
    return individual_actions_loss, group_activity_loss
'''