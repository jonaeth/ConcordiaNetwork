import pandas as pd

files_to_adjust = ['sgd_rating_obs.psl',
                   'rating_obs.psl',
                   'bpmf_rating_obs.psl',
                   'rating_truth.psl',
                   'item_pearson_rating_obs.psl']

def normalize_ratings(path_to_data):
    for file_to_adjust in files_to_adjust:
        path_learn = f'{path_to_data}/{file_to_adjust}'
        df = pd.read_csv(path_learn, names=['user_id', 'item_id', 'rating'], sep='\t')
        if len(df[df['rating'] > 1]) != 0:
            df['rating'] /= 5
            df['rating'] = df['rating'].clip(0, 1)
        df.to_csv(path_learn, header=None, index=False, sep='\t')

    for file_to_adjust in ['avg_item_rating_obs.psl', 'avg_user_rating_obs.psl']:
        path_learn = f'{path_to_data}/{file_to_adjust}'
        df = pd.read_csv(path_learn, names=['id', 'rating'], sep='\t')
        if len(df[df['rating'] > 1]) != 0:
            df['rating'] /= 5
            df['rating'] = df['rating'].clip(0, 1)
        df.to_csv(path_learn, header=None, index=False, sep='\t')
