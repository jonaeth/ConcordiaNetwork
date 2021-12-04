import os

from PSLImplementation import Facts
from PSLImplementation import get_pdf_estimate_of_targets_integration
from PSLImplementation.Groundings.HerbrandBase import HerbrandBase


def predict_psl_distribution(fold_nr, data_fraction, psl_prediction_folder, output_folder, path_to_predicate_file):
    os.makedirs(output_folder, exist_ok=True)
    grounded_rules_path = f'{psl_prediction_folder}/psl_rules_{data_fraction}_{fold_nr}.psl' #CORE_PSL_OUTPUT_FOLDER
    predicate_database_path = path_to_predicate_file#PATH_TO_PREDICATE_DATA_PATHS_FILE
    rating_targets_path = f'Data/MovieLens/Transformed/train/{data_fraction}/{fold_nr}/learn/rated_obs.psl'#f'{psl_prediction_folder}/psl_pred_{data_fraction}_{fold_nr}.psl'
    path_to_save_prob_density_files = f'{output_folder}/prob_denstiy_frac{data_fraction}_{fold_nr}.psl' #PSL_DISTRIBUTION_OUTPUT_FOLDER
    facts = Facts(predicate_database_path, str(fold_nr), 'learn', data_fraction)
    print('done')

    herbrand_base = HerbrandBase(grounded_rules_path, 'rating')

    with open(rating_targets_path) as fp:
        target_predicate_arguments = [tuple(line.strip().split()[:2]) for line in fp.readlines()]

    print('making inference')
    prob_estimations = get_pdf_estimate_of_targets_integration(herbrand_base, facts, target_predicate_arguments,
                                                               'rating', path_to_save_prob_density_files)
    print('inference is done')
    with open(path_to_save_prob_density_files, 'w') as fp:
       fp.writelines([f"{user_id}\t{item_id}\t{', '.join([str(i) for i in dist])}\n" for (user_id, item_id), dist in
                      zip(target_predicate_arguments, prob_estimations)])


#predict_psl_distribution(0, 5)