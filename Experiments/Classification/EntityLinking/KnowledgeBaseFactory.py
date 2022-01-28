import os
from collections import defaultdict
import re


class KnowledgeBaseFactory:
    def __init__(self, path_to_save_predicates):
        self.path_to_save_predicates = path_to_save_predicates
        self._build_necessary_folders()
        self.transformed_data = None

    def _build_necessary_folders(self):
        os.makedirs(f"{self.path_to_save_predicates}/observations", exist_ok=True)
        os.makedirs(f"{self.path_to_save_predicates}/targets", exist_ok=True)
        os.makedirs(f"{self.path_to_save_predicates}/truths", exist_ok=True)

    def build_predicates(self, training_data, df_ratings_targets=None):
        predicates = defaultdict(list)
        targets = []
        truths = []
        sentence_ids = 0
        for trial_id, trial in enumerate(training_data.keys()):
            for i in range(len(training_data[trial]['inc'])):
                instance = training_data[trial]['inc'][i]
                if not len(instance['pos_neg_example']):
                    continue
                sentence_ids += 1
                targets += [f"{sentence_ids}\t0\t{i}\t{example_id}" for example_id in range(len(instance['pos_neg_example']))]
                truths += [f"{sentence_ids}\t0\t{i}\t{example_id}\t{instance['pos_neg_example'][example_id][2][1]}" for example_id in range(len(instance['pos_neg_example']))]

                for rv_name, rv_value in instance['graph'].base_assignment.items():
                    if len(re.findall('[0-9]+', rv_name)) == 1:
                        example_id = re.findall('[0-9]+', rv_name)[0]
                        predicate_name = rv_name[:rv_name.index(example_id)]
                        predicates[predicate_name].append(f"{sentence_ids}\t0\t{i}\t{example_id}\t{self.normalise_predicate_value(predicate_name, rv_value)}")
                    else:
                        example_ids = re.findall('[0-9]+', rv_name)
                        predicate_name = rv_name[:rv_name.index(example_ids[0])]
                        predicates[predicate_name].append(f"{sentence_ids}\t0\t{i}\t{example_ids[0]}\t{example_ids[1]}\t{self.normalise_predicate_value(predicate_name, rv_value)}")

            for i in range(len(training_data[trial]['exc'])):
                instance = training_data[trial]['exc'][i]
                if not len(instance['pos_neg_example']):
                    continue
                sentence_ids += 1
                targets += [f"{sentence_ids}\t1\t{i}\t{example_id}" for example_id in range(len(instance['pos_neg_example']))]
                truths += [f"{sentence_ids}\t1\t{i}\t{example_id}\t{instance['pos_neg_example'][example_id][2][1]}" for example_id in range(len(instance['pos_neg_example']))]

                for rv_name, rv_value in instance['graph'].base_assignment.items():
                    if len(re.findall('[0-9]+', rv_name)) == 1:
                        example_id = re.findall('[0-9]+', rv_name)[0]
                        predicate_name = rv_name[:rv_name.index(example_id)]
                        predicates[predicate_name].append(
                            f"{sentence_ids}\t1\t{i}\t{example_id}\t{self.normalise_predicate_value(predicate_name, rv_value)}")
                    else:
                        example_ids = re.findall('[0-9]+', rv_name)
                        predicate_name = rv_name[:rv_name.index(example_ids[0])]
                        predicates[predicate_name].append(
                            f"{sentence_ids}\t1\t{i}\t{example_ids[0]}\t{example_ids[1]}\t{self.normalise_predicate_value(predicate_name, rv_value)}")

        with open(f'Experiments/Classification/EntityLinking/teacher/train/targets/z.psl', 'w') as f:
            for row in targets:
                f.write(f'{row}\n')

        with open(f'Experiments/Classification/EntityLinking/teacher/train/truths/z.psl', 'w') as f:
            for row in truths:
                f.write(f'{row}\n')

        for predicate_name, groundings in predicates.items():
            with open(f'Experiments/Classification/EntityLinking/teacher/train/observations/{predicate_name}.psl', 'w') as f:
                for row in groundings:
                    f.write(f'{row}\n')
        return predicates

    def normalise_predicate_value(self, predicate_name, value):
        if predicate_name in ['token_length', 'str_length', 'mention_ctx_entropy', 'entity_entropy']:
            if value == 1:
                return 0.5
            if value == 2:
                return 1
        return value