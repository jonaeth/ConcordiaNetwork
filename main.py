from Teacher import PSLTeacher

predicate_file = 'Experiments/VQA/data/teacher/model/predicates.psl'
rule_file = 'Experiments/VQA/data/teacher/model/model.psl'
train_predicate_folder = 'Experiments/VQA/data/teacher/train'

psl_model = PSLTeacher()
psl_model.build_model(predicate_file=predicate_file, rules_file=rule_file, predicate_folder=train_predicate_folder)
psl_model.fit()
print(psl_model.predict())
print(psl_model)
