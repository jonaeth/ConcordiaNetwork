from abc import ABC
from collections import defaultdict
import os
import numpy as np
from pslpython.model import Model
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule

class Teacher(ABC):

    def __init__(self, model_name='model', model=None, predicates=None):
        self.model_name = model_name
        self.model = model
        if predicates:
            self.predicates = predicates
        else:
            self.predicates = []

    def build_model(self, predicate_file, rules_file):
        pass

    def __str__(self):
        pass

    def write_model_to_file(self, file_name):
        pass

    def fit(self):
        pass

    def predict(self):
        pass


class PSLTeacher(Teacher):

    def __init__(self,
                 model_name='model',
                 model=None,
                 predicates=None,
                 cli_options=None,
                 psl_options=None):
        super().__init__(model_name=model_name, model=model, predicates=predicates)
        if cli_options:
            self.cli_options = cli_options
        else:
            self.cli_options = []  # TODO Discuss need
        if psl_options:
            self.psl_options = psl_options
        else:
            self.psl_options = {
                'log4j.threshold': 'OFF',  # TODO Discuss good default
                'votedperceptron.numsteps': '2'
            }

    def build_model(self, predicate_file, rules_file):

        self.model = Model(self.model_name)
        self._add_predicates(predicate_file)
        self._add_rules(rules_file)

    def __str__(self):
        rules = '\n'.join(str(rule) for rule in self.model.get_rules())
        return rules

    def write_model_to_file(self, file_name):
        with open(file_name, 'w') as f:
            for rule in self.model.get_rules():
                f.write(str(rule) + '\n')

    def fit(self):
        # We have to find a good option to store difference between observations and truths
        for predicate in self.predicates:
            self.model.get_predicate(predicate).clear_data().add_data_file(Partition.OBSERVATIONS, '')
        self.model.learn(additional_cli_optons=self.cli_options, psl_config=self.psl_options)

    def predict(self):
        # Why result'S'?
        results = self.model.infer(additional_cli_optons=self.cli_options, psl_config=self.psl_options)
        # You return the model separately (not really needed, but then you only return results for one predicate
        return results

    def _add_predicates(self, predicate_file):
        with open(predicate_file, 'r') as p_file:
            for line in p_file.readlines():
                predicate = line.split('\t')
                predicate_name = predicate[0]
                self.predicates.append(predicate_name)
                is_closed = bool(predicate[1])
                arity = int(predicate[3])
                self.model.add_predicate(Predicate(predicate_name, closed=is_closed, size=arity))

    def _add_rules(self, rules_file):
        with open(rules_file, 'r') as r_file:
            for line in r_file.readlines():
                rule = line.split('\t')
                implication = rule[0]
                weight = float(rule[1])
                if weight == -1:
                    weighted = False
                    self.model.add_rule(Rule(implication, weighted=weighted))
                else:
                    self.model.add_rule(Rule(implication, weight=weight))


class MLNTeacher(Teacher):
    pass
