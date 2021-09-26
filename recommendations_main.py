
from Teacher import PSLTeacher

predicate_file = 'Experiments/Recommendations/data/teacher/model/predicates.psl'
rule_file = 'Experiments/Recommendations/data/teacher/model/model.psl'
train_predicate_folder = 'Experiments/Recommendations/data/teacher/train'
test_predicate_folder = 'Experiments/Recommendations/data/teacher/test'


import os

#[os.rename(f'{test_predicate_folder}/observations/{i}', f'{test_predicate_folder}/observations/{i.replace("_obs.txt", ".psl")}') for i in os.listdir(f'{test_predicate_folder}/observations/')]
#[os.rename(f'{test_predicate_folder}/truths/{i}', f'{test_predicate_folder}/truths/{i.replace("_targets.txt", ".psl")}') for i in os.listdir(f'{test_predicate_folder}/truths/')]
#[os.rename(f'{test_predicate_folder}/targets/{i}', f'{test_predicate_folder}/targets/{i.replace("_truth.txt", ".psl")}') for i in os.listdir(f'{test_predicate_folder}/targets/')]


print('jj')

psl_model = PSLTeacher()
psl_model.build_model(predicate_file=predicate_file, rules_file=rule_file, predicate_folder=train_predicate_folder)
print('Learning rule weights')
psl_model.fit()
print('Making inference')
print(psl_model.model.get_predicate('Rating'))
print(psl_model.predict())
print(psl_model)
