from torch.optim import Adam


class Student:
    def __init__(self, neural_model, student_loss_function, optimizer=None):
        self.model = neural_model
        self.loss_fn = self._set_student_loss(student_loss_function)
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = Adam(self.model.parameters())

    def __str__(self):
        print(self.model)

    def _set_student_loss(self, student_loss_function):
        if student_loss_function is None:
            return lambda x, y: 0
        else:
            return student_loss_function

    # TODO implement function
    def write_model_to_file(self, file_name):
        pass

    def fit(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, neural_input):
        predictions = self.model(neural_input)
        return list(predictions)  # Transformed into a list as it can be a series of predictions for several tasks


