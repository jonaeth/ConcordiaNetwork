from collections import defaultdict
from pslpython.partition import Partition


class Facts:
    def __init__(self, psl_model):
        self.facts_base = defaultdict(dict)
        for predicate_name, predicate_data in psl_model.get_predicates().items():
            self.facts_base[predicate_name.lower()] = {'_'.join(row[:-1].astype(str)): row[-1]\
                                               for row in predicate_data.data()[Partition.OBSERVATIONS].values}

    def get_truth_value_of_predicate(self, predicate_name, *args):
        if "_".join(args) not in self.facts_base[predicate_name]:
            return None
        return float(self.facts_base[predicate_name]["_".join(args)])

    def get_truth_values_of_grounded_atom(self, grounded_atom):
        predicate_name = grounded_atom.split('(')[0]
        arguments = grounded_atom.split('(')[1][:-1].split(', ')
        return self.get_truth_value_of_predicate(predicate_name, *arguments)

