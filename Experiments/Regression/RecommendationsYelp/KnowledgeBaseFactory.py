from sklearn.neighbors import NearestNeighbors
import operator
from Experiments.Regression.Utils.HyperTransformations.generate_left_predicates import *
from Experiments.Regression.Utils.HyperTransformations.fix_scale import normalize_ratings
from Experiments.Regression.Utils.HyperTransformations.DataTransformation import DataTransformation


class KnowledgeBaseFactory:
    def __init__(self, path_to_save_predicates):
        self.path_to_save_predicates = path_to_save_predicates
        self.transformed_data = None

    def build_predicates(self, training_data, df_ratings_targets):
        df_users = training_data[0]
        df_items = training_data[1]
        df_ratings_obs = training_data[2]

        self.transformed_data = DataTransformation(df_users, df_items, df_ratings_obs, False)

        df_rated_obs = pd.concat([df_ratings_obs, df_ratings_targets])[['user_id', 'item_id']]
        df_rated_obs.to_csv(f'{self.path_to_save_predicates}/observations/rated.psl', header=False, index=False,
                            sep='\t')
        df_ratings_obs[['user_id', 'item_id', 'rating']].to_csv(
            f'{self.path_to_save_predicates}/observations/rating.psl',
            header=False, index=False,
            sep='\t')

        df_ratings_targets[['user_id', 'item_id']].to_csv(f'{self.path_to_save_predicates}/targets/rating.psl',
                                                          header=False, index=False,
                                                          sep='\t')
        df_ratings_targets[['user_id', 'item_id', 'rating']].to_csv(f'{self.path_to_save_predicates}/truths/rating.psl',
                                                                    header=False,
                                                                    index=False, sep='\t')

        print('Writing predicate data')
        self._build_avg_item_obs()
        self._build_avg_user_obs()
        self._build_users_are_friends_obs()
        self._build_sim_mf_cosine_items_obs()
        self._build_sim_mf_cosine_users_obs()
        self._build_user_obs()
        self._build_item_obs()
        self._build_sim_adjust_cosine_items_obs()
        self._build_sim_content_items_jaccard_obs()
        self._build_sim_cosine_items()
        self._build_sim_cosine_users()
        compute_predicates(self.path_to_save_predicates)
        self._build_sim_mf_euclidean_items_obs()
        self._build_sim_mf_euclidean_users_obs()
        normalize_ratings(self.path_to_save_predicates)

    def _build_avg_item_obs(self):
        pd.DataFrame(self.transformed_data.df_ratings.groupby('item_id').rating.apply(np.mean).reset_index()) \
            .to_csv(f'{self.path_to_save_predicates}/observations/avg_item_rating.psl',
                    header=False, index=False, sep='\t')

    def _build_avg_user_obs(self):
        pd.DataFrame(self.transformed_data.df_ratings.groupby('user_id').rating.apply(np.mean).reset_index()) \
            .to_csv(f'{self.path_to_save_predicates}/observations/avg_user_rating.psl',
                    header=False, index=False, sep='\t')

    def _build_users_are_friends_obs(self):
        if 'friends_idx' not in self.transformed_data.df_users.columns:
            return
        friend_pair_list = []
        for user_id, friend_list_idx in self.transformed_data.df_users[['user_id', 'friends_idx']].values:
            friend_pair_list += [(user_id, friend_id) for friend_id in friend_list_idx]
        df = pd.DataFrame(friend_pair_list, columns=['user_id', 'friend_id'])
        df[df['user_id'] != df['friend_id']].to_csv(f'{self.path_to_save_predicates}/observations/users_are_friends.psl',
                                                    header=False,
                                                    index=False, sep='\t')

    def _build_sim_mf_cosine_items_obs(self):
        nbrs = NearestNeighbors(n_neighbors=51, metric='cosine', n_jobs=4).fit(
            self.transformed_data.item_latent_space.T)
        all_distances, all_neighbor_idxs = nbrs.kneighbors(self.transformed_data.item_latent_space.T)
        all_distances, all_neighbor_idxs = self.__remove_self_reference_in_neighbours(all_distances, all_neighbor_idxs)
        self.save_predicate_obs_list_to_file(all_neighbor_idxs,
                                             f'{self.path_to_save_predicates}/observations/sim_mf_cosine_items.psl')

    def _build_sim_mf_cosine_users_obs(self):
        nbrs = NearestNeighbors(n_neighbors=51, metric='cosine', n_jobs=4).fit(self.transformed_data.user_latent_space)
        all_distances, all_neighbor_idxs = nbrs.kneighbors(self.transformed_data.user_latent_space)
        all_distances, all_neighbor_idxs = self.__remove_self_reference_in_neighbours(all_distances, all_neighbor_idxs)
        self.save_predicate_obs_list_to_file(all_neighbor_idxs,
                                             f'{self.path_to_save_predicates}/observations/sim_mf_cosine_users.psl')

    def _build_user_obs(self):
        self.transformed_data.df_users[['user_id']].sort_values('user_id').to_csv(
            f'{self.path_to_save_predicates}/observations/user.psl', sep='\t',
            index=False, header=False)

    def _build_item_obs(self):
        self.transformed_data.df_items[['item_id']].sort_values('item_id').to_csv(
            f'{self.path_to_save_predicates}/observations/item.psl', sep='\t',
            index=False, header=False)

    def _build_sim_adjust_cosine_items_obs(self):
        rating_matrix_user_mean_subtacted = self.transformed_data.rating_matrix - \
                                            self.transformed_data.rating_matrix.mean(axis=1)[:, None]
        nbrs = NearestNeighbors(n_neighbors=51, metric='cosine', n_jobs=4).fit(rating_matrix_user_mean_subtacted)
        all_distances, all_neighbor_idxs = nbrs.kneighbors(rating_matrix_user_mean_subtacted)
        all_distances, all_neighbor_idxs = self.__remove_self_reference_in_neighbours(all_distances, all_neighbor_idxs)
        _, filtered_neighbour_idxs = self.filter_nearest_neighbours(all_distances, all_neighbor_idxs, 1, operator.lt)
        self.save_predicate_obs_list_to_file(filtered_neighbour_idxs,
                                             f'{self.path_to_save_predicates}/observations/sim_adjcos_items.psl')

    def _build_sim_content_items_jaccard_obs(self):
        nbrs = NearestNeighbors(n_neighbors=51, metric='jaccard', n_jobs=4).fit(
            np.array(self.transformed_data.df_items.categories_binary.values.tolist()))
        all_distances, all_neighbor_idxs = nbrs.kneighbors(
            np.array(self.transformed_data.df_items.categories_binary.values.tolist()))
        all_distances, all_neighbor_idxs = self.__remove_self_reference_in_neighbours(all_distances, all_neighbor_idxs)
        _, filtered_neighbour_idxs = self.filter_nearest_neighbours(all_distances, all_neighbor_idxs, 1, operator.eq)
        self.save_predicate_obs_list_to_file(filtered_neighbour_idxs,
                                             f'{self.path_to_save_predicates}/observations/sim_content_items_jaccard.psl')

    def _build_sim_cosine_items(self):
        nbrs = NearestNeighbors(n_neighbors=51, metric='cosine', n_jobs=4).fit(self.transformed_data.rating_matrix)
        all_distances, all_neighbor_idxs = nbrs.kneighbors(self.transformed_data.rating_matrix)
        all_distances, all_neighbor_idxs = self.__remove_self_reference_in_neighbours(all_distances, all_neighbor_idxs)
        _, filtered_neighbour_idxs = self.filter_nearest_neighbours(all_distances, all_neighbor_idxs, 1, operator.eq)
        self.save_predicate_obs_list_to_file(filtered_neighbour_idxs,
                                             f'{self.path_to_save_predicates}/observations/sim_cosine_items.psl')

    def _build_sim_cosine_users(self):
        nbrs = NearestNeighbors(n_neighbors=51, metric='cosine', n_jobs=4).fit(self.transformed_data.rating_matrix.T)
        all_distances, all_neighbor_idxs = nbrs.kneighbors(self.transformed_data.rating_matrix.T)
        all_distances, all_neighbor_idxs = self.__remove_self_reference_in_neighbours(all_distances, all_neighbor_idxs)
        _, filtered_neighbour_idxs = self.filter_nearest_neighbours(all_distances, all_neighbor_idxs, 1, operator.eq)
        self.save_predicate_obs_list_to_file(filtered_neighbour_idxs,
                                             f'{self.path_to_save_predicates}/observations/sim_cosine_users.psl')

    def _build_sim_mf_euclidean_items_obs(self):
        nbrs = NearestNeighbors(n_neighbors=51, metric='euclidean', n_jobs=4).fit(
            self.transformed_data.item_latent_space.T)
        all_distances, all_neighbor_idxs = nbrs.kneighbors(self.transformed_data.item_latent_space.T)
        all_distances, all_neighbor_idxs = self.__remove_self_reference_in_neighbours(all_distances, all_neighbor_idxs)
        self.save_predicate_obs_list_to_file(all_neighbor_idxs,
                                             f'{self.path_to_save_predicates}/observations/sim_mf_euclidean_items.psl')

    def _build_sim_mf_euclidean_users_obs(self):
        nbrs = NearestNeighbors(n_neighbors=51, metric='euclidean', n_jobs=4).fit(
            self.transformed_data.user_latent_space)
        all_distances, all_neighbor_idxs = nbrs.kneighbors(self.transformed_data.user_latent_space)
        all_distances, all_neighbor_idxs = self.__remove_self_reference_in_neighbours(all_distances, all_neighbor_idxs)
        self.save_predicate_obs_list_to_file(all_neighbor_idxs,
                                             f'{self.path_to_save_predicates}/observations/sim_mf_euclidean_users.psl')

    def filter_nearest_neighbours(self, all_distances, all_neighbor_idxs, filter_value, filter_operator):
        filtered_distances, filtered_neighbour_idxs = [], []
        for distances, neighbour_idxs in zip(all_distances, all_neighbor_idxs):
            filtered_distance_neighbour_pairs = [(distance, neighbour_idx) for distance, neighbour_idx in
                                                 zip(distances, neighbour_idxs) if
                                                 filter_operator(distance, filter_value)]
            if not filtered_distance_neighbour_pairs:
                continue
            new_distances, new_neighbours = zip(*filtered_distance_neighbour_pairs)
            filtered_distances.append(new_distances)
            filtered_neighbour_idxs.append(new_neighbours)
        return filtered_distances, filtered_neighbour_idxs

    def __remove_self_reference_in_neighbours(self, all_distances, all_neighbour_idxs):
        all_distances = [[distance for neighbour_idx, distance in zip(neighbour_idxs, distances) if neighbour_idx != i]
                         for i, (neighbour_idxs, distances) in enumerate(zip(all_neighbour_idxs, all_distances))]
        all_neighbour_idxs = [[neighbour_idx for neighbour_idx in neighbour_idxs if neighbour_idx != i] for
                              i, neighbour_idxs in enumerate(all_neighbour_idxs)]
        return all_distances, all_neighbour_idxs

    def save_predicate_obs_list_to_file(self, all_neighbor_idxs, path):
        nearest_neighbours_pair_list = []
        for item_idx, neighbour_idxs in enumerate(all_neighbor_idxs):
            nearest_neighbours_pair_list += [(item_idx, item2_idx) for item2_idx in neighbour_idxs if
                                             item_idx != item2_idx]
        pd.DataFrame(nearest_neighbours_pair_list).to_csv(path, header=False, index=False, sep='\t')
