from collections import defaultdict
import numpy as np


class Facts:
    def __init__(self, facts_file, fold, mode, frac=None):
        grounded_predicates_files = self.read_grounded_predicate_files(facts_file,
                                                                       str(fold) if fold else fold,
                                                                       str(mode),
                                                                       str(frac)
                                                                       )
        self.facts_base = defaultdict(dict)
        for predicate, arity, predicate_data_file in grounded_predicates_files:
            self.facts_base[predicate] = self.build_predicate_truth_value_mapping(predicate_data_file, arity)

    def build_predicate_truth_value_mapping(self, groundings_path, arity):
        """
        Reads the file of a particular predicate which contains observed truth values
        of a specific grounding, and build a mapping from a specific grounding to predicate truth value.
        There are 2 input file formats.
        Format 1 (e.g. Rating(user, item)):
        '
        3 \t 101 \t 0.85
        3 \t 102 \t 0.5
        .
        .
        .
        952 \t 105 \t 0.57
        '
        Format 2 (e.g. rated(user, item), where the truth value is 1):
        '
        3 101
        3 102
        .
        .
        .
        952 105
        '
        :param groundings_path:
        file path to the truth values of the predicate
        :param arity:
        arity of the predicate
        :return: a dictionary with grounding to truth value of the predicate
        'constant1_constant2': truth_value
        e.g.{'3_101': 0.85
        .
        .
        .
        '952_105': 0.57}
        """
        groundings = self.read_file(groundings_path)
        predicate_truth_value_mapping = {}
        for line in groundings:
            line = line.strip()
            line_split_values = line.split('\t')
            if arity == 1:
                arguments = line_split_values[0]
                # Format 2 style files (see comment above)
                if len(line_split_values) == 1:
                    predicate_truth_value_mapping[arguments] = 1
                # Format 2 style files (see comment above)
                else:
                    predicate_truth_value_mapping[arguments] = np.float32(line_split_values[1])
            if arity == 2:
                arguments = "_".join(line_split_values[:2])
                if len(line_split_values) == 2:
                    predicate_truth_value_mapping[arguments] = 1
                else:
                    predicate_truth_value_mapping[arguments] = np.float32(line_split_values[-1])

        return predicate_truth_value_mapping

    def read_grounded_predicate_files(self, src_path, fold, mode, frac):
        '''
        Reads predicate groundings file which for each predicate contains paths to its truth values for a specific
        grounding, and predicate's arity.
        The paths may contain three different wildcards (see params below):
        These wildcards are replaced with the provided arguments to this function.
        The file format:
        PREDICATE_NAME\arity: Path to the file containing truth values for specific groundings
        e.g.:
        SIM_PEARSON_ITEMS/2: Data/Yelp/Transformed/?frac?/?fold?/?mode?/sgd_rating_obs.txt
        These txt files contain, for a specific predicate, data of the form:
        e.g. Rating(user, item):
        '
        3 \t 101 \t 0.85
        3 \t 102 \t 0.5
        .
        .
        .
        952 \t 105 \t 0.57
        '
        :param src_path:
        :param fold: integer value representing fold number from kfold cross validation
        :param mode: integer value representing percentage of data available to the models
        :param frac: string value representing if we want to read PSL learn or eval split
        :return:
        A list of lists containing predicate name, its arity and path to the file containing truth values:
        [
            ['SGD_RATING', 2, Data/Yelp/Transformed/50/0/eval/sgd_rating_obs.txt],
            ['SIM_COS_ITEMS', 2, Data/Yelp/Transformed/50/0/eval/sim_cosine_items_obs.txt],
            .
            .
            .
        ]
        '''
        with open(src_path) as fp:
            lines = fp.readlines()
        return_lines = []
        for line in lines:
            if fold is not None:
                line = line.replace('?fold?', fold)
            line = line.replace('?mode?', mode)
            line = line.replace('?frac?', frac)

            predicate_arity, data_path = line.strip().split(': ')
            predicate, arity = predicate_arity.split('/')
            return_lines.append([predicate.lower(), int(arity), data_path])

        return return_lines

    def read_file(self, src_path):
        lines = []
        with open(src_path) as fp:
            lines += fp.readlines()

        return list(set(lines))

    def get_truth_value_of_predicate(self, predicate_name, *args):
        if "_".join(args) not in self.facts_base[predicate_name]:
            return None
        return float(self.facts_base[predicate_name]["_".join(args)])

    def get_truth_values_of_grounded_atom(self, grounded_atom):
        predicate_name = grounded_atom.split('(')[0]
        arguments = grounded_atom.split('(')[1][:-1].split(', ')
        return self.get_truth_value_of_predicate(predicate_name, *arguments)

