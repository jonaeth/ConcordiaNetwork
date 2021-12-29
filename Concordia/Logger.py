from collections import defaultdict
import numpy as np


class Logger:
    def __init__(self, cfg, log_save_path):
        self.log_save_path = log_save_path
        self.print_config(cfg)

    def _write_log(self, *args):
        with open(self.log_save_path, 'a') as f:
            print(*args, file=f)

    def print_config(self, cfg):
        for key, value in cfg.items():
            self._write_log(key, ': ', value)

    def print_epoch_log(self, log_dict):
        log_line = ["Epoch Log:\n"]
        for key, value in log_dict.items():
            log_line.append(f"{key.replace('_', ' ').upper()}: {value}")
        self._write_log(" ".join(log_line))

    def build_epoch_log(self, batches_metrics, evaluation_step, epoch):
        wide_batch_metrics = defaultdict(list)
        for batch_metrics in batches_metrics:
            for metric_name, metric_val in batch_metrics.items():
                wide_batch_metrics[metric_name].append(metric_val)
        logs = {}
        for metric_name, metric_values_per_batch in wide_batch_metrics.items():
            logs[metric_name] = np.mean(metric_values_per_batch)
        logs['evaluation_step'] = evaluation_step
        logs['epoch_nr'] = epoch
        return logs

