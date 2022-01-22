import torch
from tqdm import tqdm
from Concordia.utils.torch_losses import kl_divergence
from Concordia.MixtureOfExperts import MixtureOfExperts
from Concordia.Logger import Logger


class ConcordiaNetwork:
    def __init__(self, student, teacher=None, **config):
        self.student = student
        self.teacher = teacher
        self.device = config['gpu_device'] if config['gpu_device'] else torch.device('cpu')
        self.config = config
        #self.mixture_of_experts_model = MixtureOfExperts(config['image_vector_length']) #TODO Figure this out!
        self.train_online = config['train_online']
        self.regression = config['regression']
        self.logger = Logger(config, config['log_path'])
        self._epoch = 0

    def fit(self, train_data_loader, val_data_loader, epochs=10, callbacks=[], metrics={}):
        for epoch in range(1, epochs+1):
            self._epoch = epoch
            if self.train_online:
                batches_metrics = self._fit_online(train_data_loader, metrics)
            else:
                batches_metrics = self._fit_precomputed_teacher_predictions(train_data_loader, metrics)
            epoch_log = self.logger.build_epoch_log(batches_metrics, 'Training', self._epoch)
            self.logger.print_epoch_log(epoch_log)
            self._run_callbacks(callbacks, 'epoch_end', epoch_log)
            self._evaluate_student(val_data_loader, callbacks, metrics)

    def fit_semisupervised(self, unlabled_train_data_loader, labeled_train_data_loader, val_data_loader, epochs, metrics={}):
        for epoch in range(1, epochs + 1):
            self.fit_supervised(labeled_train_data_loader, val_data_loader, 1, metrics=metrics)
            self.fit_unsupervised(unlabled_train_data_loader, val_data_loader, 1, metrics=metrics)

    def fit_unsupervised(self, unlabled_train_data_loader, val_data_loader, epochs, callbacks=[], metrics={}):
        batches_metrics = []
        for epoch in range(1, epochs + 1):
            with tqdm(unlabled_train_data_loader) as t:
                for training_input, teacher_prediction, target in t:
                    student_prediction = self.student.predict(self._to_device(training_input))
                    loss = self._get_teacher_student_loss(self._to_device(teacher_prediction), student_prediction)
                    self.student.fit(loss)
                    batches_metrics.append({'loss': self._detach_variables(loss)})
                    t.set_postfix(self.logger.build_epoch_log(batches_metrics, 'Training-usnupervised', self._epoch))
            self._evaluate_student(val_data_loader, callbacks, metrics)


    def fit_supervised(self, labeled_training_data, val_data_loader, epochs, callbacks=[], metrics={}):
        batches_metrics = []
        for epoch in range(1, epochs + 1):
            with tqdm(labeled_training_data) as t:
                for training_input, teacher_prediction, target in t:
                    student_prediction = self.student.predict(self._to_device(training_input))
                    loss = self.compute_loss(student_prediction,
                                             self._to_device(teacher_prediction),
                                             self._to_device(target))
                    self.student.fit(loss)

                    batches_metrics.append({'loss': self._detach_variables(loss)})

                    t.set_postfix(self.logger.build_epoch_log(batches_metrics, 'Training-supervised', self._epoch))
            self._evaluate_student(val_data_loader, callbacks, metrics)

    def fit_student_alone(self, labeled_training_data, val_data_loader, epochs, callbacks=[], metrics={}):
        batches_metrics = []
        for epoch in range(1, epochs + 1):
            with tqdm(labeled_training_data) as t:
                for training_input, target in t:
                    student_prediction = self.student.predict(self._to_device(training_input))
                    loss = self.student.loss_fn(student_prediction, self._to_device(target).to(torch.float32))
                    self.student.fit(loss)

                    batches_metrics.append(self._get_batch_metrics(self._detach_variables(student_prediction),
                                                                   target,
                                                                   metrics,
                                                                   loss.item()))

                    t.set_postfix(self.logger.build_epoch_log(batches_metrics, 'Training-supervised', self._epoch))
            self._evaluate_student(val_data_loader, callbacks, metrics)

    def _fit_precomputed_teacher_predictions(self, train_data_loader, metrics=None):
        batches_metrics = []
        with tqdm(train_data_loader) as t:
            for training_input, teacher_prediction, target in t:
                student_prediction = self.student.predict(self._to_device(training_input))
                loss = self.compute_loss(student_prediction,
                                         self._to_device(teacher_prediction),
                                         self._to_device(target))
                self.student.fit(loss)
                batches_metrics.append(self._get_batch_metrics(self._detach_variables(student_prediction),
                                                               target,
                                                               metrics,
                                                               loss.item()))
                t.set_postfix(self.logger.build_epoch_log(batches_metrics, 'Training', self._epoch))
        return batches_metrics

    def _fit_online(self, train_data_loader, metrics=None):
        batches_metrics = []
        for training_input, target in tqdm(train_data_loader):
            student_prediction = self.student.predict(self._to_device(training_input)).to(torch.float)
            teacher_input = self._detach_variables(training_input + student_prediction)
            teacher_prediction = self.teacher.predict(teacher_input,
                                                      self._detach_variables(target)).to(torch.float)

            loss = self.compute_loss(student_prediction,
                                     self._to_device(teacher_prediction),
                                     self._to_device(target.to(torch.float)))
            self.student.fit(loss)
            batches_metrics.append(self._get_batch_metrics(self._detach_variables(student_prediction),
                                                           target,
                                                           metrics,
                                                           loss.item()))
        return batches_metrics

    def predict(self):
        pass

    def compute_loss(self, student_predictions, teacher_predictions, target_values):
        if self.regression:
            student_predictions_val = student_predictions[0].to(torch.float32)
            student_predictions_distribution = [student_predictions[1].to(torch.float32)]
        else:
            student_predictions_val = student_predictions
            student_predictions_distribution = student_predictions
        return 0.5 * self._get_teacher_student_loss(teacher_predictions, student_predictions_distribution) \
               + 0.5 * self._to_device(self.student.loss_fn(student_predictions_val, target_values.to(torch.float32)))


    def _get_teacher_student_loss(self, teacher_predictions, student_predictions):
        kl_divergence_loss = 0
        for task_index, do_comparison in enumerate(self.config['teacher_student_distributions_comparison']):
            if do_comparison:
                kl_divergence_loss += kl_divergence(student_predictions[task_index].to(torch.float32),
                                                    teacher_predictions[task_index].to(torch.float32))
        return kl_divergence_loss

    def _detach_variables(self, variables):
        return [variable.detach().cpu() for variable in variables]

    def _to_device(self, variables):
        if variables is None:
            return torch.tensor(0).to(device=self.device)
        if type(variables) != list and type(variables) != tuple:
            return variables.to(device=self.device)
        return [variable.to(device=self.device) for variable in variables]

    def _get_batch_metrics(self, student_predictions, target, metrics, loss):
        custom_metrics_batch = self._evaluate_custom_metrics(student_predictions, target, metrics)
        custom_metrics_batch['loss'] = loss
        return custom_metrics_batch

    def _evaluate_student(self, val_data_loader, callbacks=None, metrics=None):
        batches_metrics = []
        t = tqdm(val_data_loader)
        for validation_input, target in t:
            student_prediction = self.student.predict(self._to_device(validation_input))
            if self.regression:
                loss = self.student.loss_fn(student_prediction[0], self._to_device(target))
            else:
                loss = self.student.loss_fn(student_prediction, self._to_device(target))
            batches_metrics.append(self._get_batch_metrics(student_prediction, self._to_device(target), metrics, loss.item()))
            t.set_postfix(self.logger.build_epoch_log(batches_metrics, 'Validation', self._epoch))

        epoch_log = self.logger.build_epoch_log(batches_metrics, 'Test', self._epoch)
        epoch_log['Epoch'] = epoch_log
        self.logger.print_epoch_log(epoch_log)

        self._run_callbacks(callbacks, 'epoch_end', epoch_log)

    def _run_callbacks(self, callbacks, training_loop_status, logs):
        for callback in callbacks:
            if training_loop_status == 'epoch_end':
                callback.on_epoch_end(self._epoch, logs)

    def _evaluate_custom_metrics(self, student_predictions, targets, metrics):
        metric_results = {}
        if not metrics or targets is None:
            return metric_results
        for metric_name, metric_fn in metrics.items():
            val = metric_fn(student_predictions, targets)
            metric_results[metric_name] = val.detach().numpy()
        return metric_results
