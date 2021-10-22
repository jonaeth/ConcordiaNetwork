import numpy as np
import pandas as pd
from scipy.special import softmax


class PredicateBuilder:
    def __init__(self, path_to_save_predicates, collective_activity_config):
        self.path_to_save_predicates = path_to_save_predicates
        self.cfg = collective_activity_config

    def write_predicate_to_file(self, predicate_values, predicate_name, predicate_visibility):
        with open(f'{predicate_values}/{predicate_visibility}/{predicate_name}.psl') as fp:
            for predicate_tuple in predicate_values:
                fp.write('\t'.join(predicate_tuple))

    def build_predicates(self, inputs, student_predictions):
        batch_data = inputs
        batch_size=batch_data[0].shape[0]
        num_frames=batch_data[0].shape[1]
        actions_in = batch_data[2].reshape((batch_size, num_frames, self.cfg.num_boxes))
        bboxes_num = batch_data[4].reshape(batch_size, num_frames)

        actions_in_nopad_same_shape = []

        for b in range(batch_size):
            actions_of_batch = []
            for i, N in enumerate(bboxes_num[b]):
                actions_of_batch.append(actions_in[b][i, :N])
            actions_in_nopad_same_shape.append(actions_of_batch)

        bounding_boxes = batch_data[1].detach().cpu().numpy()
        actions_scores_original_size = self._convert_action_scores_to_original_shape(student_predictions, bboxes_num, batch_size)
        bboxes_nopad = self._remove_padding_from_bounding_boxes(bounding_boxes, bboxes_num, batch_size)
        ground_truths = [[np.identity(self.cfg.num_actions)[action.cpu().numpy()] for action in batch] for batch in actions_in_nopad_same_shape]

        self.build_and_save_psl_predicates(bboxes_nopad, actions_scores_original_size, ground_truths)


    def _convert_action_scores_to_original_shape(self, actions_scores, bboxes_num, batch_size):
        actions_scores_original_size = []
        action_scores_cpu = actions_scores.detach().cpu().numpy()
        cumm_counter = 0
        for b in range(batch_size):
            batch_action_scores = []
            for nr_of_boxes in bboxes_num[b].squeeze(0):
                batch_action_scores.append(action_scores_cpu[cumm_counter:cumm_counter+nr_of_boxes])
                cumm_counter += nr_of_boxes
            actions_scores_original_size.append(batch_action_scores)
        return actions_scores_original_size


    def _remove_padding_from_bounding_boxes(self, bounding_boxes, bboxes_num, batch_size):
        bboxes_nopad = []
        for b in range(batch_size):
            bboxes_nopad_batch = []
            for i, N in enumerate(bboxes_num[b]):
                bboxes_nopad_batch.append(bounding_boxes[b, i, :N, :])
            bboxes_nopad.append(bboxes_nopad_batch)
        return bboxes_nopad

    def get_data_df_column_names(self, cfg):
        if cfg.num_actions == 8:
            df_bbox_cord_column_names = [0, 1, 2, 3]
            df_bbox_label_column_names = [4, 5, 6, 7, 8, 9, 10, 11]
            df_bbox_truth_label_column_names = [12, 13, 14, 15, 16, 17, 18, 19]
            df_frame_id_column_name = 20
            df_bbox_target_column_name = 21
        else:
            df_bbox_cord_column_names = [0, 1, 2, 3]
            df_bbox_label_column_names = [4, 5, 6, 7, 8, 9, 10]
            df_bbox_truth_label_column_names = [11, 12, 13, 14, 15, 16, 17]
            df_frame_id_column_name = 18
            df_bbox_target_column_name = 19

        return df_bbox_cord_column_names, df_bbox_label_column_names, df_bbox_truth_label_column_names, df_frame_id_column_name, df_bbox_target_column_name

    def get_distance_between_boxes(self, box1, box2):
        box_1_center = np.array([(box1[0] + box1[1]) / 2, (box1[2] + box1[3]) / 2])
        box_2_center = np.array([(box2[0] + box2[1]) / 2, (box2[2] + box2[3]) / 2])
        return np.exp(-np.linalg.norm((box_1_center - box_2_center) / 8)) #this 8 is hardcoded for the rbf kernel


    """
        Closeness between bounding boxes predicate is computed between bboxes in the same frame.
    """
    def distances_between_bounding_boxes(self, df):
        distances = []
        for _, df_frame in df.groupby('frame_id'):
            bounding_boxes = df_frame[['concatinated_bb', 'box_id']].values
            for box_id_1, box1 in enumerate(bounding_boxes):
                for box_id_2, box2 in enumerate(bounding_boxes):
                    if box_id_1 != box_id_2:
                        distances.append((int(box1[1]), int(box2[1]), self.get_distance_between_boxes(box1[0], box2[0])))
        if len(distances) == 0:
            distances.append((-1, -1, 0))
        return distances

    def get_frame_label(self, bounding_box_labels):
        return np.mean(bounding_box_labels, axis=1)

    def build_close_predicate(self, df):
        predicate_values = self.distances_between_bounding_boxes(df)
        self.write_predicate_to_file(predicate_values, 'Close', 'observations')

    def build_local_predicates(self, bounding_box_label_pred):
        nr_of_actions = len(bounding_box_label_pred[0])
        predicate_values = []
        for box_id, box in enumerate(bounding_box_label_pred):
            for i in range(nr_of_actions):
                predicate_values.append((box_id, i, box[i]))
        self.write_predicate_to_file(predicate_values, 'Local', 'observations')


    def build_truth_doing_predicates(self, bounding_box_label_pred):
        nr_of_actions = len(bounding_box_label_pred[0])
        predicate_values = []
        for box_id, box in enumerate(bounding_box_label_pred):
            for i in range(nr_of_actions):
                predicate_values.append((box_id, i, box[i]))
        self.write_predicate_to_file(predicate_values, 'Doing', 'truths')
        self.write_predicate_to_file(predicate_values, 'DoingTruth', 'truths')


    def build_frame_global_label(self, df_data, cfg):
        nr_of_actions = len(df_data['concatinated_bb_labels'].iloc[0])
        predicate_values = []
        if cfg.rule_25_doing_groundings == 'truth':
            avg_box_label_pred = df_data[['concatinated_ground_truths', 'frame_id']].groupby('frame_id').apply(
                lambda x: x['concatinated_ground_truths'].values.mean())
        else:
            avg_box_label_pred = df_data[['concatinated_bb_labels', 'frame_id']].groupby('frame_id').apply(
                lambda x: x['concatinated_ground_truths'].values.mean())

        for j, frame_nr in enumerate(df_data['frame_id']):
            for i in range(nr_of_actions):
                predicate_values.append((j, i, avg_box_label_pred.loc[frame_nr][i]))
        self.write_predicate_to_file(predicate_values, 'GlobalAct', 'observations')


    def build_target_doing_predicates(self, df):
        nr_of_actions = len(df['concatinated_bb_labels'].iloc[0])
        predicate_values = []
        for box_id in range(len(df)):
            for i in range(nr_of_actions):
                predicate_values.append((box_id, i))
        self.write_predicate_to_file(predicate_values, 'Doing', 'targets')


    def build_sequence_predicate(self, df):
        predicate_values = []
        for box_id1 in range(len(df)):
            for box_id2 in range(box_id1 + 1, len(df)):
                if df.iloc[box_id1]['frame_id'] + 1 == df.iloc[box_id2]['frame_id']:
                    predicate_values.append((box_id1, box_id2, 1))
                    predicate_values.append((box_id2, box_id1, 1))
        self.write_predicate_to_file(predicate_values, 'Seq', 'observations')

    def build_sequence_close_predicate(self, df):
        predicate_values = []
        for box_id1 in range(len(df)):
            for box_id2 in range(box_id1 + 1, len(df)):
                if df.iloc[box_id1]['frame_id'] + 1 == df.iloc[box_id2]['frame_id']:
                    predicate_values.append((box_id1, box_id2,
                                             self.get_distance_between_boxes(df.iloc[box_id1]['concatinated_bb'],
                                                                        df.iloc[box_id1]['concatinated_bb'])))
        self.write_predicate_to_file(predicate_values, 'CloseSeq', 'observations')

    def build_same_obs_predicate(self, df):
        predicate_values = [(-1, -1, 0)]
        for box_id1, track_id_box1 in enumerate(df['track_id']):
            for box_id2, track_id_box2 in enumerate(df['track_id']):
                if box_id1 != box_id2:
                    if track_id_box1 != -1 and track_id_box1 == track_id_box2:
                        predicate_values.append((box_id1, box_id2, 1))
                    if track_id_box1 != -1 and track_id_box1 != track_id_box2:
                        predicate_values.append((box_id1, box_id2, 0))
                    if track_id_box1 == -1 and track_id_box1 != track_id_box2:
                        predicate_values.append((box_id1, box_id2, 0))
        self.write_predicate_to_file(predicate_values, 'Same', 'observations')

    def build_bounding_boxes_data_structure(self, bounding_boxes_coord, bounding_boxes_labels, ground_truths):
        concatinated_bb = np.concatenate([np.concatenate(batch) for batch in bounding_boxes_coord])
        concatinated_bb_targets = concatinated_bb[:, 4]
        concatinated_bb = concatinated_bb[:, :-1]
        concatinated_bb_labels = np.concatenate([np.concatenate(batch) for batch in bounding_boxes_labels])
        concatinated_ground_truths = np.concatenate([np.concatenate(batch) for batch in ground_truths])
        frame_indexes = []
        frame_cumu_idx = 0
        for batch in bounding_boxes_coord:
            for frame_idx, frame in enumerate(batch):
                for _ in frame:
                    frame_indexes.append(frame_cumu_idx)
                frame_cumu_idx += 1
            frame_cumu_idx += 10
        concatinated_bb_labels = softmax(concatinated_bb_labels, axis=1)


        return concatinated_bb, concatinated_bb_labels, concatinated_ground_truths, frame_indexes, concatinated_bb_targets

    def build_pandas_df_from_input_data(self, concatinated_bb, frame_indexes, concatinated_bb_targets,
                                        concatinated_bb_labels, concatinated_ground_truths):
        df = pd.DataFrame({"concatinated_bb": list(concatinated_bb)})
        df['frame_id'] = np.array(frame_indexes).reshape(-1, 1)
        df['track_id'] = np.array(concatinated_bb_targets).reshape(-1, 1)
        df['concatinated_bb_labels'] = list(concatinated_bb_labels)
        df['concatinated_ground_truths'] = list(concatinated_ground_truths)
        df = df.reset_index().rename({'index': 'box_id'}, axis=1)
        return df

    def build_and_save_psl_predicates(self, bounding_boxes_coord, bounding_boxes_student_predictions, ground_truths):
        concatinated_bb, concatinated_bb_labels, concatinated_ground_truths, frame_indexes, concatinated_bb_targets \
            = self.build_bounding_boxes_data_structure(bounding_boxes_coord, bounding_boxes_student_predictions, ground_truths)
        df = self.build_pandas_df_from_input_data(concatinated_bb, frame_indexes, concatinated_bb_targets,
                                                  concatinated_bb_labels, concatinated_ground_truths)
        self.build_close_predicate(df)
        self.build_local_predicates(concatinated_bb_labels)
        self.build_target_doing_predicates(df)
        self.build_truth_doing_predicates(concatinated_ground_truths)
        self.build_frame_global_label(df, self.cfg)
        self.build_same_obs_predicate(df)
        return df
