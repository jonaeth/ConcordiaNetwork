import numpy as np
from tqdm import tqdm
from scipy import integrate
from Concordia.PSLImplementation.DataBase.TruthAssignedRule import TruthAssignedRule
import gc

def assign_true_values_to_formulas(grounded_formulas, herbrand_interpretation, grounded_target_atom=None):
    """
    Assigns the observed truth values to each grounded atom inside a list of formulas. Unobserved target atom is
    replaced by a wildcard 'target_predicate'.
    :param grounded_formulas: List of grounded formulas with their weight.
    e.g.:
    [(1.05, "PREDICATE1(Alice,Bob,...) & PREDICATE2(Alice,Bob,...) >> PREDICATE3(Alice,Bob,...)"),
     (0.84, "PREDICATE1(Alice,Bob,...) & PREDICATE2(Alice,Bob,...) >> PREDICATE4(Alice,Bob,...)"),
     ...
     ]
    :param herbrand_interpretation: Herbrand interpretation object with observed truth values of predicates.
    :param grounded_target_atom: e.g. 'PREDICATE2(Alice, Bob, ...)'
    :return: List of tuples with weight of each formula and lists of truth values of head and body parts of the formula.
    format of a tuple entry:
    (weight, [truth values of body atoms], [truth values of head atoms])
    e.g.
    [
     (1.05, [1, 'target_predicate'], [0.4]),
     (0.84, [0.8, 'target_predicate'], [0.9])
    ]
    """
    return [TruthAssignedRule(rule, herbrand_interpretation, grounded_target_atom) for rule in grounded_formulas]


def hinge_loss(body_atoms, head_atoms):
    formula_potential = 1 - sum(head_atoms) - len(body_atoms) + sum(body_atoms)
    potential_val = max(0, formula_potential)
    return potential_val


def sample_potential_dist(target_predicate_value, assgined_truth_rules):
    """
    Computes the likelihood of the target predicate being true given observed truth values of the
    grounded atoms in a list of formulas.
    :param target_predicate_value: float value of a ground unobserved target predicate
    :param fully_grounded_formulas: list of tuples with of the following format:assigned
    [(weight, [truth values of formula body atoms], [truth values of formula head atoms])]
    e.g.
    [
    (0.994, ['1', '0.31', '0', 'target_predicate'], ['0.44']),
    (0.994, ['1', '0.51', '1', '0.94'], ['target_predicate']),
    ...
    ]
    :return: float representing the likelihood of the target_predicate having the passed in value given observations
    """
    potential = 0
    for truth_assigned_rule in assgined_truth_rules:
        body_atoms, head_atoms = truth_assigned_rule.assign_truth_value_to_target_predicate(target_predicate_value)
        potential += truth_assigned_rule.get_weight() * hinge_loss(body_atoms, head_atoms)
    return np.exp(-potential)


def get_pdf_estimate_of_targets_integration(herbrand_base,
                                            facts,
                                            target_predicate_arguments,
                                            target_predicate,
                                            integration_points=np.arange(0, 1.1, 0.1)):
    """
    Returns a probability distribution for each grounded target predicate. The probability distribution is computed
    using numerical integration method QUAD from the Fortran library in different intervals (integration_points).

    :param herbrand_base: Herbrand base object with all available groundings of the KB.
    :param facts: Herbrand interpretation object with observed truth values of predicates.
    :param rating_targets: Target grounded predicate list to calculate probability distributions.
    :param integration_points: list or generator with points between probability estimate will be computed.
    :return: List of lists with probability distribution computed for each grounded predicate.
    """
    estimates = []
    arguments_of_estimates = []
    for arguments in tqdm(target_predicate_arguments):
        grounded_target_predicate = f'{target_predicate}({", ".join([str(arg) for arg in arguments])})'
        grounded_rules = herbrand_base.get_markov_blanket_of_ground_predicate(grounded_target_predicate)
        evaluated_formulas = assign_true_values_to_formulas(grounded_rules, facts,
                                                            grounded_target_atom=grounded_target_predicate)
        potential_dist = []
        Z = 0
        for lower_bound, upper_bound in zip(integration_points[:-1], integration_points[1:]):
            p = integrate.quad(sample_potential_dist, lower_bound, upper_bound, args=(evaluated_formulas), epsabs=1e-1)[0]
            Z += p
            potential_dist.append(p)
        potential_dist = np.array(potential_dist) / Z
        estimates.append(potential_dist)
        arguments_of_estimates.append(arguments)

    return arguments_of_estimates, estimates


