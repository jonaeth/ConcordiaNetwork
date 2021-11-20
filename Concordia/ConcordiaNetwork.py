import torch
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from Concordia.torch_losses import kl_divergence


class ConcordiaNetwork:
    def __init__(self, student, teacher, predicate_builder, **config):
        self.student = student
        self.teacher = teacher
        self.predicate_builder = predicate_builder
        self.device = config['gpu_device'] if config['gpu_device'] else torch.device('cpu')
        self.config = config

    def fit(self, input_data_loader, val_data_loader, epochs=10, callbacks=None, metrics=None):
        for epoch in range(1, epochs+1):
            batch_metrics = []
            for input, target in tqdm(input_data_loader):
                student_prediction = self.student.predict(input)
                teacher_prediction = self._get_teacher_predictions_online(input, student_prediction, target)
                loss = self.compute_loss(student_prediction, teacher_prediction, target)
                self.student.fit(loss)
                # TODO think how to move the bottom lines somewhere else
                custom_metrics_batch = self._evaluate_custom_metrics(student_prediction, target, metrics)
                custom_metrics_batch['loss'] = loss.item()
                batch_metrics.append(custom_metrics_batch)
            epoch_log = self._build_epoch_log(batch_metrics, 'Training')
            self._run_callbacks(callbacks, 'epoch_end', epoch_log, epoch)
            self._evaluate_the_student(val_data_loader, epoch, callbacks, metrics)


    def _get_teacher_predictions_online(self, input, student_predictions, target):
        self.predicate_builder.build_predicates(input, student_predictions, target)
        self.teacher.set_ground_predicates(self.predicate_builder.path_to_save_predicates)
        self.teacher.fit()
        teacher_prediction = self.teacher.predict()
        return teacher_prediction

    def _evaluate_the_student(self, val_data_loader, epoch, callbacks=None, metrics=None):
        metrics_per_batch = []
        for input, target in tqdm(val_data_loader):
            student_prediction = self.student.predict(input)
            loss = self.student.loss_fn(student_prediction, target)
            custom_metrics_batch = self._evaluate_custom_metrics(student_prediction, target, metrics)
            custom_metrics_batch['loss'] = loss.item()
            metrics_per_batch.append(custom_metrics_batch)
        epoch_log = self._build_epoch_log(metrics_per_batch, 'Test')
        self._run_callbacks(callbacks, 'epoch_end', epoch_log, epoch)

    def _fit_teacher_offline(self, input_loader):
        pass

    # TODO move this to another class potentially, Logging class?
    def _build_epoch_log(self, metrics_per_batch, evaluation_step):
        metrics_per_batch_wide = defaultdict(list)
        for metrics_in_batch in metrics_per_batch:
            for metric_name, metric_val in metrics_in_batch.items():
                metrics_per_batch_wide[metric_name].append(metric_val)
        logs = {}
        for metric_name, metric_values_per_batch in metrics_per_batch_wide.items():
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

    def predict(self):
        pass

    def _get_teacher_student_loss(self, teacher_predictions, student_predictions):
        kl_divergence_loss = 0
        for task_index, do_comparison in enumerate(self.config['teacher_student_distributions_comparison']):
            if do_comparison:
                kl_divergence_loss += kl_divergence(student_predictions[task_index],
                                                    teacher_predictions[task_index])
        return kl_divergence_loss

    def compute_loss(self, student_predictions, teacher_predictions, target_values):
        return self._get_teacher_student_loss(student_predictions, teacher_predictions) \
               + self.student.loss_fn(student_predictions, target_values)


