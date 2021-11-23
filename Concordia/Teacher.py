from abc import ABC
import os
from pslpython.model import Model
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule
from distutils.util import strtobool
import torch


class Teacher(ABC):
    def __init__(self, knowledge_base_factory, predicates=None, predicates_to_infer=None, **config):
        self.knowledge_base_factory = knowledge_base_factory
        self.predicates_to_infer = predicates_to_infer
        if predicates:
            self.predicates = predicates
        else:
            self.predicates = []
        self.config = config

    def build_model(self):
        pass

    def __str__(self):
        pass

    def write_model_to_file(self, file_name):
        pass

    def fit(self):
        pass

    def predict(self, teacher_input, student_predictions, target):
        pass


class PSLTeacher(Teacher):
    def __init__(self,
                 knowledge_base_factory,
                 predicates=None,
                 predicates_to_infer=None,
                 **config):
        super().__init__(knowledge_base_factory=knowledge_base_factory,
                         predicates=predicates,
                         predicates_to_infer=predicates_to_infer, **config)
        if 'cli_options' in config:
            self.cli_options = config['cli_options']
        else:
            self.cli_options = []
        if 'psl_options' in config:
            self.psl_options = config['psl_options']
        else:
            self.psl_options = {
                'log4j.threshold': 'OFF',
                'votedperceptron.numsteps': '2'
            }
        if 'jvm_options' in config:
            self.jvm_options = config['jvm_options']
        else:
            self.jvm_options = ['-Xms4096M', '-Xmx12000M']

        self.model_path = config['teacher_model_path']
        self.model = self._build_model()

    def _build_model(self):
        model = Model('Teacher')
        self._add_predicates(model, f"{self.model_path}/predicates.psl")
        self._add_rules(model, f"{self.model_path}/model.psl")
        return model

    def _add_predicates(self, model, predicate_file):
        with open(predicate_file, 'r') as p_file:
            for line in p_file.readlines():
                predicate = line.split('\t')
                predicate_name = predicate[0]
                self.predicates.append(predicate_name)
                is_closed = strtobool(predicate[1])
                arity = int(predicate[2])
                model.add_predicate(Predicate(predicate_name, closed=is_closed, size=arity))

    def _add_rules(self, model, rules_file):
        with open(rules_file, 'r') as r_file:
            for line in r_file.readlines():
                rule = line.split('\t')
                implication = rule[0]
                weight = float(rule[1])
                if weight == -1:
                    weighted = False
                    model.add_rule(Rule(implication, weighted=weighted))
                else:
                    model.add_rule(Rule(implication, weight=weight))

    def __str__(self):
        rules = '\n'.join(str(rule) for rule in self.model.get_rules())
        return rules

    def write_model_to_file(self, file_name):
        with open(file_name, 'w') as f:
            for rule in self.model.get_rules():
                f.write(str(rule) + '\n')

    def fit(self):
        self.model.learn(additional_cli_optons=self.cli_options,
                         psl_config=self.psl_options,
                         jvm_options=self.jvm_options)

    def predict(self, teacher_input, student_predictions, target):
        self.knowledge_base_factory.build_predicates(teacher_input, student_predictions, target)
        self._set_ground_predicates(self.knowledge_base_factory.path_to_save_predicates)

        if self.config['train_teacher']:
            self.fit()

        results = self.model.infer(additional_cli_optons=self.cli_options, psl_config=self.psl_options)
        predictions = []
        for predicate in self.predicates_to_infer:
            if not predicate:
                predictions.append(torch.Tensor([]))
            else:
                psl_predictions = results[self.model.get_predicate(predicate)]\
                                                    .sort_values(by=[0, 1])\
                                                    .pivot(index=0, columns=1, values='truth')\
                                                    .values
                predictions.append(torch.Tensor(psl_predictions))
        return predictions

    def _set_ground_predicates(self, predicate_folder):
        grounded_predicates = []
        observations_folder = f'{predicate_folder}/observations/'
        targets_folder = f'{predicate_folder}/targets/'
        truths_folder = f'{predicate_folder}/truths/'
        observed_predicates = [predicate.replace('.psl', '') for predicate in os.listdir(observations_folder)]
        target_predicates = [predicate.replace('.psl', '') for predicate in os.listdir(targets_folder)]
        truth_predicates = [predicate.replace('.psl', '') for predicate in os.listdir(truths_folder)]
        for predicate in self.predicates:
            if predicate not in grounded_predicates:
                self.model.get_predicate(predicate).clear_data()
            if predicate in observed_predicates:
                self.model.get_predicate(predicate).add_data_file(Partition.OBSERVATIONS,
                                                                  f'{observations_folder}/{predicate}.psl')
            if predicate in target_predicates:
                self.model.get_predicate(predicate).add_data_file(Partition.TARGETS,
                                                                  f'{targets_folder}/{predicate}.psl')
            if predicate in truth_predicates:
                self.model.get_predicate(predicate).add_data_file(Partition.TRUTH,
                                                                  f'{truths_folder}/{predicate}.psl')
            grounded_predicates.append(predicate)




class MLNTeacher(Teacher):
    pass
