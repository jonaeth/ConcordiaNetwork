from collections import defaultdict
from PSLImplementation.DataBase.Rule import Rule


class HerbrandBase:
    def __init__(self, herbrand_base_file, predicate_of_interest):
        """
        1. Reads the Herbrand base file, which is computed by the PSL solver and stores the data in this class.
        The file format:
        Weight \t Squared? \t Rule
        weight \t Boolean if rule is squared or not \t rule in CNF format

        e.g.
        Weight Squared? Rule
        0.765073 false ( ~( SIM_MF_COSINE_ITEMS('1136', '1499') ) | ~( RATING('10', '1136') ) | RATING('10', '1499') )
        0.765073 false ( ~( SIM_MF_COSINE_ITEMS('549', '105') ) | ~( RATED('5', '549') ) | RATING('5', '105') )

        Remark: the weight is for the ungrounded rule and does not represent the truth value for the specific grounding.

        2. In addition, it creates a dictionary where the keys are each possible grounding of a predicate of interest
        and the values are lists of rules containing that specific ground predicate. This is used later to find the
        markov blanket of a ground predicate in constant time.
        E.g.:
        {
            "RATING('10', '1499')": [
                "SIM_MF_COSINE_ITEMS('1136', '1499') ) & RATING('10', '1136') => RATING('10', '1499')",
                "SIM_MF_COSINE_ITEMS('1499', '1136') ) & RATING('10', '1499') => RATING('10', '1136')",
                ...
            ]
            "RATING('10', '1136')" : ...
        }

        :param herbrand_base_file:
        """
        self.all_grounded_rules = []
        with open(herbrand_base_file, 'r') as fp:
            for full_formula in fp.readlines()[1:]:
                weight = float(full_formula.split('\t')[0])
                formula = full_formula.split('\t')[2].lower()
                self.all_grounded_rules.append(Rule(weight=weight, formula=formula))

        self.markov_blankets = self.__build_markov_blankets_of_ground_predicates(predicate_of_interest)

    def get_markov_blanket_of_ground_predicate(self, grounded_predicate):
        """
        Returns list of formulas with their weights which contain the provided grounded predicate (markov blanket)
        :param grounded_predicate: e.g. "PREDICATE1(3, 1)"
        :return: List of tuples with the following format:
        (formula_weight, "grounded formula")
        e.g.
        [
        (0.14, "PREDICATE1(123, 3) & PREDICATE2(5, 123) => PREDICATE3(5, 3)"),
        (1.88, "PREDICATE1(199, 8) & PREDICATE2(13, 199) & PREDICATE4(5) => PREDICATE1(123, 3)"),
        ...
        ]
        """
        return [self.all_grounded_rules[i] for i in self.markov_blankets[grounded_predicate]]

    def __build_markov_blankets_of_ground_predicates(self, target_predicate):
        """
        Builds a dictionary where keys are grounded predicates appearing in the Herbrand base and values are lists of
        indexes to the formulas appearing in the markov blanket of the grounded predicate.
        :param target_predicate: e.g. PREDICATE1
        :return: dictionary e.g:
        {
            "PREDICATE1(2, 3)": [1, 3, 4],
            "PREDICATE1(3, 4)": [4, 10, 123],
            "PREDICATE1(55, 99)": [431, 155, 0]
        }
        """
        markov_blankets = defaultdict(list)
        for i, rule in enumerate(self.all_grounded_rules):
            for atom in rule.get_atoms_by_predicate(target_predicate):
                markov_blankets[atom].append(i)
        return markov_blankets

