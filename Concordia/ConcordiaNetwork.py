import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from tqdm import tqdm


class ConcordiaNetwork:
    def __init__(self, student, teacher, predicate_builder, student_teacher_loss, student_target_loss, teacher_offline):
        self.student_teacher_loss = student_teacher_loss
        self.student_target_loss = student_target_loss
        self.student = student
        self.teacher = teacher
        self.predicate_builder = predicate_builder
        self.teacher_offline = teacher_offline
        self.device = torch.device('cuda', 0) #TODO replace this with config settings

    def fit(self, input_data_loader, targets=None, **kwargs):
        self.student.model.train()
        if self.teacher_offline:
            self._fit_teacher_offline(input_data_loader)
        else:
            self._fit_teacher_online(input_data_loader)

    def _fit_teacher_online(self, input_data_loader, epochs=10, callback=None):
        for epoch in range(1, epochs+1):
            for input, target in tqdm(input_data_loader):
                student_prediction = self.student.predict(input)
                self.predicate_builder.build_predicates(input, student_prediction, target)
                self.teacher.set_ground_predicates(self.predicate_builder.path_to_save_predicates)
                self.teacher.fit()
                teacher_prediction = self.teacher.predict()
                loss = self.compute_loss(student_prediction, teacher_prediction, target)
                self.student.fit(loss)
                callback()

    def _fit_teacher_offline(self, input_loader):
        pass

    def predict(self):
        pass

    def compute_loss(self, student_predictions, teacher_predictions, target_values):
        return self.student_teacher_loss(student_predictions, teacher_predictions) + self.student_target_loss(student_predictions, target_values) #TODO add the balancing params


