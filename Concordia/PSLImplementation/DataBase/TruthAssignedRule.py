
class TruthAssignedRule:
    def __init__(self, rule, herbrand_interpretation, target_atom=None):
        self.unassigned_rule = rule
        self.target_atom = target_atom
        self.body_atoms, self.head_atoms = self._assign_true_values_to_rule(herbrand_interpretation)

    def __str__(self):
        print("Weight: " + str(self.unassigned_rule.weight) + "\tRule:" + " & ".join(self.body_atoms) + " >> " +
              " & ".join(self.head_atoms))

    def _assign_true_values_to_rule(self, herbrand_interpretation):
        body_atoms = []
        for atom in self.unassigned_rule.body_atoms:
            if atom == self.target_atom:
                body_atoms.append('target_predicate()')
            else:
                truth_value = herbrand_interpretation.get_truth_values_of_grounded_atom(atom)
                if truth_value is None:
                    truth_value = 0
                body_atoms.append(truth_value)
        head_atoms = []
        for atom in self.unassigned_rule.head_atoms:
            if atom == self.target_atom:
                head_atoms.append('target_predicate()')
            else:
                truth_value = herbrand_interpretation.get_truth_values_of_grounded_atom(atom)
                if truth_value is None:
                    truth_value = 0
                head_atoms.append(truth_value)

        return body_atoms, head_atoms

    def assign_truth_value_to_target_predicate(self, truth_value):
        new_body_atoms = [truth_value if atom == 'target_predicate()' else atom for atom in self.body_atoms]
        new_head_atoms = [truth_value if atom == 'target_predicate()' else atom for atom in self.head_atoms]
        return new_body_atoms, new_head_atoms

    def get_weight(self):
        return self.unassigned_rule.weight


