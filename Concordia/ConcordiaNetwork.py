import torch
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from Concordia.utils.torch_losses import kl_divergence
from Concordia.MixtureOfExperts import MixtureOfExperts


class ConcordiaNetwork:
    def __init__(self, student, teacher, **config):
        self.student = student
        self.teacher = teacher
        self.device = config['gpu_device'] if config['gpu_device'] else torch.device('cpu')
        self.config = config
        self.mixture_of_experts_model = MixtureOfExperts(config['image_vector_length'])

    def fit(self, train_data_loader, val_data_loader, epochs=10, callbacks=None, metrics=None):

        for epoch in range(1, epochs+1):
            batches_metrics = []
            for training_input, target in tqdm(train_data_loader):
                student_prediction = self.student.predict(self._to_device(training_input))
                teacher_prediction = self.teacher.predict(self._detach_variables(training_input),
                                                          self._detach_variables(student_prediction),
                                                          self._detach_variables(target))

                loss = self.compute_loss(student_prediction,
                                         self._to_device(teacher_prediction),
                                         self._to_device(target))
                self.student.fit(loss)
                # TODO: Consider this to run on GPU
                batches_metrics.append(self._get_batch_metrics(self._detach_variables(student_prediction),
                                                               target,
                                                               metrics,
                                                               loss.item()))

            epoch_log = self._build_epoch_log(batches_metrics, 'Training')
            self._run_callbacks(callbacks, 'epoch_end', epoch_log, epoch)
            self._evaluate_student(val_data_loader, epoch, callbacks, metrics)

    def predict(self):
        pass

    def compute_loss(self, student_predictions, teacher_predictions, target_values):
        return 0.5 * self._get_teacher_student_loss(teacher_predictions, student_predictions) \
               + 0.5 * self.student.loss_fn(student_predictions, target_values)

    def _get_teacher_student_loss(self, teacher_predictions, student_predictions):
        kl_divergence_loss = 0
        for task_index, do_comparison in enumerate(self.config['teacher_student_distributions_comparison']):
            if do_comparison:
                kl_divergence_loss += kl_divergence(student_predictions[task_index],
                                                    teacher_predictions[task_index])
        return kl_divergence_loss

    def _detach_variables(self, variables):
        return [variable.detach().cpu() for variable in variables]

    def _to_device(self, variables):
        if type(variables) != list:
            return variables.to(device=self.device)
        return [variable.to(device=self.device) for variable in variables]

    def _get_batch_metrics(self, student_predictions, target, metrics, loss):
        custom_metrics_batch = self._evaluate_custom_metrics(student_predictions, target, metrics)
        custom_metrics_batch['loss'] = loss
        return custom_metrics_batch

    def _evaluate_student(self, val_data_loader, epoch, callbacks=None, metrics=None):
        batches_metrics = []
        for validation_input, target in tqdm(val_data_loader):
            student_prediction = self.student.predict(validation_input)
            loss = self.student.loss_fn(student_prediction, target)
            batches_metrics.append(self._get_batch_metrics(student_prediction, target, metrics, loss.item()))
        epoch_log = self._build_epoch_log(batches_metrics, 'Test')
        self._run_callbacks(callbacks, 'epoch_end', epoch_log, epoch)

    # TODO move this to another class potentially, Logging class?
    def _build_epoch_log(self, batches_metrics, evaluation_step):
        wide_batch_metrics = defaultdict(list)
        for batch_metrics in batches_metrics:
            for metric_name, metric_val in batch_metrics.items():
                wide_batch_metrics[metric_name].append(metric_val)
        logs = {}
        for metric_name, metric_values_per_batch in wide_batch_metrics.items():
            logs[metric_name] = np.mean(metric_values_per_batch)
        logs['evaluation_step'] = evaluation_step
        return logs

    def _run_callbacks(self, callbacks, training_loop_status, logs, epoch):
        for callback in callbacks:
            if training_loop_status == 'epoch_end':
                callback.on_epoch_end(epoch, logs)

    def _evaluate_custom_metrics(self, student_predictions, targets, metrics):
        metric_results = {}
        for metric_name, metric_fn in metrics.items():
            val = metric_fn(student_predictions, targets)
            metric_results[metric_name] = val
        return metric_results
