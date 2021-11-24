import numpy as np
from collections import defaultdict
import re
import pandas as pd


class Tracks:
    def __init__(self, id = 0,  ti = 0 , te = 0, bbs = None, locs = None):
        self.id = id #target id
        self.ti = ti #First frame target appears in
        self.te = te #Last frame target appears in
        if bbs:
            self.bbs = bbs #bounding boxes [frame, x, y, withd, height]
        else:
            self.bbs = []
        if locs:
            self.locs = locs # whatever
        else:
            self.locs = []

    def find_bounding_box_in_frame(self, frame_id):
        frame_id = int(frame_id)
        for bounding_box in self.bbs:
            bounding_box_frame_id = int(bounding_box[0])
            if bounding_box_frame_id == frame_id:
                return bounding_box
        return None


def read_tracks(dir):
    with open(dir,mode='r') as f:
        framesLine = f.readline() #dummyline
        targetsLine = f.readline()
        n_targets = [int(s) for s in re.findall(r'[-+]?\d*\.\d+|\d+', targetsLine)][0]
        tracks = []
        for i in range(1, n_targets+1):
            track = Tracks()
            targetLine = f.readline()
            targetInformation = [int(s) for s in re.findall(r'[-+]?\d*\.\d+|\d+', targetLine)]
            track.id = targetInformation[0]
            track.ti = targetInformation[1]
            track.te = targetInformation[2]
            loopLength = targetInformation[2]-targetInformation[1]
            dummy = f.readline() #dummy line
            print(dummy)
            for i in range(loopLength+1):
                line = f.readline()
                track.bbs.append([float(s) for s in re.findall(r'[-+]?\d*\.\d+|\d+', line)])
            dummy = f.readline()  # dummy line
            print(dummy)
            for i in range(loopLength+1):
                line = f.readline()
                track.locs.append([float(s) for s in re.findall(r'[-+]?\d*\.\d+|\d+', line)])
            tracks.append(track)
    return tracks


def get_closest_bounding_box(bounding_box, tracks, tracks_used_in_frame):
    min_distance = 99999
    track_id = -1
    for track in tracks:
        if track.id in tracks_used_in_frame:
            continue
        track_bounding_box = track.find_bounding_box_in_frame(bounding_box[0])
        if track_bounding_box is None:
            continue
        if abs(track_bounding_box[1] - bounding_box[1]) < 20 and abs(track_bounding_box[2] - bounding_box[2]) < 20:
            distance = np.sqrt(np.power(track_bounding_box[1] - bounding_box[1], 2) + np.power(track_bounding_box[2] - bounding_box[2], 2))
            if distance < min_distance:
                track_id = track.id
                min_distance = distance

    return track_id


def write_new_annotations(tracks_path, annotations_path):
    tracks = read_tracks(tracks_path)
    tracks_used_in_frame = defaultdict(list)
    with open(annotations_path + '/new_annotations.txt', 'w') as new_annotations_f:
        with open(annotations_path + '/annotations.txt', 'r') as annonations_f:
            for line in annonations_f.readlines():
                new_line = [entry.strip() for entry in line.split('\t')]
                line_arr = [float(entry) for entry in new_line]
                target_id = get_closest_bounding_box(line_arr, tracks, tracks_used_in_frame[line_arr[0]])
                tracks_used_in_frame[line_arr[0]].append(target_id)
                print(target_id)
                final_line = '\t'.join(new_line) + '\t' + str(target_id) + '\n'
                new_annotations_f.write(final_line)


def get_euclidean_distance(bb_1, bb_2):
    return ((bb_1[0] - bb_2[0]) ** 2 + (bb_1[1] - bb_2[1]) ** 2) ** 0.5


def remove_bounding_box_from_distance_list(best_bounding_boxes_ids, bounding_box_distance_list):
    new_bounding_box_distance_list = []
    for bounding_box_src, bounding_box_target, distance in bounding_box_distance_list:
        if best_bounding_boxes_ids[0] == bounding_box_src:
            continue
        if best_bounding_boxes_ids[1] == bounding_box_target:
            continue
        new_bounding_box_distance_list.append((bounding_box_src, bounding_box_target, distance))
    return new_bounding_box_distance_list


def compute_cartesian_product_distances(frame_1_df, frame_2_df):
    distances_between_bounding_boxes = []
    for bounding_boxes1_id, bounding_boxes1_cord in enumerate(frame_1_df[[1, 2]].values):
        for bounding_boxes2_id, bounding_boxes2_cord in enumerate(frame_2_df[[1, 2]].values):
            distances_between_bounding_boxes.append((bounding_boxes1_id, bounding_boxes2_id,
                                                 get_euclidean_distance(bounding_boxes1_cord, bounding_boxes2_cord)))
    return sorted(distances_between_bounding_boxes, key=lambda x: x[2])


def create_annotations_dataframe(all_the_annotations):
    df = pd.DataFrame(all_the_annotations)
    df = df.rename({0: 'frame_id', 7: 'track_id'}, axis=1)
    try:
        df['track_id'] = df['track_id'].astype(int)
    except:
        pass
    df[[1, 2]] = df[[1, 2]].astype(int)
    return df


def fill_missing_track_ids_with_new_objects(bbox_df, number_of_tracked_bounding_boxes):
    new_track_values = []
    for track_id in bbox_df['track_id']:
        if track_id == -1:
            number_of_tracked_bounding_boxes += 1
            new_track_values.append(number_of_tracked_bounding_boxes)
        else:
            new_track_values.append(track_id)
    bbox_df['track_id'] = new_track_values
    return bbox_df, number_of_tracked_bounding_boxes


def fill_missing_data_tracks(path_to_sequence, seq_id):
    with open(path_to_sequence, 'r') as new_annotations_f:
        all_the_annotations = [line.strip().split('\t') for line in new_annotations_f.readlines()]

    df = create_annotations_dataframe(all_the_annotations)
    original_df = df.copy()
    number_of_tracked_bounding_boxes = df['track_id'].max()
    df = df[df['track_id'] == -1]

    bounding_boxes_in_frames_dfs = [frame_df for _, frame_df in df.groupby('frame_id')]
    new_bouding_boxes_in_frames_df = [bounding_boxes_in_frames_dfs[0]]
    for ii in range(len(bounding_boxes_in_frames_dfs)-1):
        bb_in_fram1, bb_in_fram2 = new_bouding_boxes_in_frames_df[ii].copy(), bounding_boxes_in_frames_dfs[ii+1]
        bb_in_fram1 = bb_in_fram1.reset_index(drop=True)
        bb_in_fram2 = bb_in_fram2.reset_index(drop=True)

        if len(bb_in_fram1[bb_in_fram1['track_id'] == -1]):
            bbox_df, number_of_tracked_bounding_boxes = fill_missing_track_ids_with_new_objects(bb_in_fram1, number_of_tracked_bounding_boxes)
            new_bouding_boxes_in_frames_df[-1] = bb_in_fram1

        if int(bb_in_fram2.frame_id[0]) - int(bb_in_fram1.frame_id[0]) > 5:
            new_bouding_boxes_in_frames_df.append(bb_in_fram2)
            continue

        distances_between_bounding_boxes = compute_cartesian_product_distances(bb_in_fram1, bb_in_fram2)
        while distances_between_bounding_boxes:
            min_distance = distances_between_bounding_boxes[0]
            distances_between_bounding_boxes = remove_bounding_box_from_distance_list(min_distance, distances_between_bounding_boxes)
            bb_in_fram2.loc[min_distance[1], 'track_id'] = bb_in_fram1.iloc[min_distance[0]]['track_id']

        new_bouding_boxes_in_frames_df.append(bb_in_fram2)
    new_bouding_boxes_in_frames_df[-1] = fill_missing_track_ids_with_new_objects(new_bouding_boxes_in_frames_df[-1], number_of_tracked_bounding_boxes)[0]

    final_sequence_df = pd.concat([original_df, pd.concat(new_bouding_boxes_in_frames_df)]).drop_duplicates([1, 2], keep='last')\
        .sort_values('frame_id').reset_index(drop=True)
    final_sequence_df.to_csv(path_to_sequence.replace(f'seq{seq_id}/new_annotations.txt', f'fixed_bboxes_{seq_id}.txt'), index=False, header=False, sep='\t')


def write_all_annotations():
    for i in range(1, 73):
        new_i = f'0{i}' if i < 10 else i
        write_new_annotations(f'Experiments/CollectiveActivity/data/collective/ActivityDataset/tracks/track{i}.dat',
                              f'Experiments/CollectiveActivity/data/collective/ActivityDataset/seq{new_i}')


write_all_annotations()

for i in range(1, 73):
    new_i = f'0{i}' if i < 10 else i
    fill_missing_data_tracks(f'Experiments/CollectiveActivity/data/collective/ActivityDataset/seq{new_i}/new_annotations.txt', new_i)

