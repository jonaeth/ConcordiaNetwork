class Rule:
    def __init__(self, weight, formula=None, body_atoms=None, head_atoms=None):
        self.weight = weight
        if formula:
            if self._is_cnf(formula):
                self.body_atoms, self.head_atoms = self.create_rule_from_cnf_string(formula)
            else:
                self.body_atoms, self.head_atoms = self.create_rule_from_implication_string(formula)
        else:
            self.body_atoms = body_atoms
            self.head_atoms = head_atoms

    def __str__(self):
        print("Weight: " + str(self.weight) + "\tRule:" + " & ".join(self.body_atoms) + " >> " +
              " & ".join(self.head_atoms))

    def _is_cnf(self, formula):
        return '>>' not in formula

    def create_rule_from_implication_string(self, formula):
        body, head = formula.split(" >> ")
        body_atoms = [atom for atom in body.split(" & ")]
        head_atoms = [atom for atom in head.split(" & ")]
        return body_atoms, head_atoms

    def create_rule_from_cnf_string(self, formula):
        """
        Converts formula in conjuctive normal form (CNF) to implication form.
        NOTE: Currently it is expected that all atoms in implication form are not negated.
        :param cnf_formula: formula in CNF, e.g.:
        "( ~( PREDICATE1('3250', '528') ) | ~( PREDICATE3('100', '528') ) | PREDICATE2('3250', '528') )"
        :return: formula in propositional logic, e.g.:
        "PREDICATE1('3250', '528') & PREDICATE3('100', '528') => PREDICATE2('3250', '528')"
        """
        atoms = formula[2:-2].split(' | ')
        body_atoms = []
        head_atoms = []
        for atom in atoms:
            if atom[0] == '~':
                body_atoms.append(atom[3:-2].strip().replace("'", ''))
            else:
                head_atoms.append(atom.strip().replace("'", ''))
        return body_atoms, head_atoms

    def get_weight(self):
        return self.weight

    def get_atoms_by_predicate(self, predicate):
        return [atom for atom in self.body_atoms + self.head_atoms if predicate == atom.split('(')[0]]
