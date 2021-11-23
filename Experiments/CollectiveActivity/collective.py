import torch
from torch.utils import data
import torchvision.models as models
import torchvision.transforms as transforms
from collections import defaultdict
import random
from PIL import Image
import numpy as np

from collections import Counter
import pandas as pd
# START: Original code by Zijian and Xinran
FRAMES_NUM={1: 302, 2: 347, 3: 194, 4: 257, 5: 536, 6: 401, 7: 968, 8: 221, 9: 356, 10: 302, 11: 1813, 12: 1084,
            13: 851, 14: 723, 15: 464, 16: 1021, 17: 905, 18: 600, 19: 203, 20: 342, 21: 650, 22: 361, 23: 311,
            24: 321, 25: 617, 26: 734, 27: 1804, 28: 470, 29: 635, 30: 356, 31: 690, 32: 194, 33: 193, 34: 395,
            35: 707, 36: 914, 37: 1049, 38: 653, 39: 518, 40: 401, 41: 707, 42: 420, 43: 410, 44: 356, 45: 151,
            46: 174, 47: 218, 48: 47, 49: 223, 50: 365, 51: 362, 52: 781, 53: 401, 54: 486, 55: 695, 56: 462,
            57: 443, 58: 629, 59: 899, 60: 550, 61: 373, 62: 200, 63: 433, 64: 319, 65: 443, 66: 315, 67: 391,
            68: 945, 69: 1011, 70: 449, 71: 351, 72: 751, 73: 325, 74: 400}


FRAMES_SIZE={1: (480, 720), 2: (480, 720), 3: (480, 720), 4: (480, 720), 5: (480, 720), 6: (480, 720), 7: (480, 720),
             8: (480, 720), 9: (480, 720), 10: (480, 720), 11: (480, 720), 12: (480, 720), 13: (480, 720), 14: (480, 720),
             15: (450, 800), 16: (480, 720), 17: (480, 720), 18: (480, 720), 19: (480, 720), 20: (450, 800), 21: (450, 800),
             22: (450, 800), 23: (450, 800), 24: (450, 800), 25: (480, 720), 26: (480, 720), 27: (480, 720), 28: (480, 720),
             29: (480, 720), 30: (480, 720), 31: (480, 720), 32: (480, 720), 33: (480, 720), 34: (480, 720), 35: (480, 720),
             36: (480, 720), 37: (480, 720), 38: (480, 720), 39: (480, 720), 40: (480, 720), 41: (480, 720), 42: (480, 720),
             43: (480, 720), 44: (480, 720), 45: (480, 640), 46: (480, 640), 47: (480, 640), 48: (480, 640), 49: (480, 640),
             50: (480, 640), 51: (480, 640), 52: (480, 640), 53: (480, 640), 54: (480, 640), 55: (480, 640), 56: (480, 640),
             57: (480, 640), 58: (480, 640), 59: (480, 640), 60: (720, 1280), 61: (720, 1280), 62: (720, 1280), 63: (720, 1280),
             64: (720, 1280), 65: (720, 1280), 66: (720, 1280), 67: (720, 1280), 68: (720, 1280), 69: (720, 1280), 70: (720, 1280),
             71: (720, 1280), 72: (720, 1280), 73: (720, 1280), 74: (720, 1280)}


def adjust_actions_annotations(action_id):
    if action_id == 4:
        action_id = 0
    if action_id > 4:
        action_id -= 1

    return action_id

ACTIONS=['NA','Crossing','Waiting','Queueing','Walking','Talking','Dancing','Jogging']
ACTIVITIES=['Crossing','Waiting','Queueing','Walking','Talking','Dancing','Jogging']


def collective_read_annotations(cfg, path, sid):
    annotations={}
    if cfg.use_modified_same_predicate:
        path=path + '/seq%02d/fixed_bboxes.txt' % sid
    else:
        path = path + '/seq%02d/new_annotations.txt' % sid

    with open(path,mode='r') as f:
        frame_id=None
        group_activity=None
        actions=[]
        bboxes=[]
        for l in f.readlines():
            values=l[:-1].split('	')

            if int(values[0])!=frame_id:
                if frame_id!=None and frame_id%10==1 and frame_id+9<=FRAMES_NUM[sid]:
                    counter = Counter(actions).most_common(2)

                    if len(counter) and not (len(counter) == 1 and counter[0][0] == 0):
                        group_activity = counter[0][0] - 1 if counter[0][0] != 0 else counter[1][0] - 1
                        annotations[frame_id] = {
                            'frame_id': frame_id,
                            'group_activity': group_activity,
                            'actions': actions,
                            'bboxes': bboxes
                        }


                frame_id=int(values[0])
                group_activity=None
                actions=[]
                bboxes=[]
            if not cfg.include_walking:
                action = int(values[5]) - 1
                if action == 4 and cfg.remove_walking:
                    continue
                actions.append(adjust_actions_annotations(int(values[5])-1) if not cfg.include_walking else int(values[5])-1)
            else:
                actions.append(int(values[5])-1)

            x,y,w,h = (int(values[i])  for i  in range(1,5))
            H,W=FRAMES_SIZE[sid]
            target = values[-1]

            bboxes.append( (y/H,x/W,(y+h)/H,(x+w)/W, target) )

        if frame_id!=None and frame_id%10==1 and frame_id+9<=FRAMES_NUM[sid]:
            counter = Counter(actions).most_common(2)
            if len(counter) and not (len(counter) == 1 and counter[0][0] == 0):
                group_activity= counter[0][0]-1 if counter[0][0]!=0 else counter[1][0]-1
                annotations[frame_id]={
                    'frame_id':frame_id,
                    'group_activity':group_activity,
                    'actions':actions,
                    'bboxes':bboxes
                }

    return annotations


import itertools
from collections import Counter

def collective_read_dataset(cfg, path,seqs):
    data = {}
    data_concatenated = []
    for sid in seqs:
        data[sid] = collective_read_annotations(cfg, path,sid)
        data_concatenated += itertools.chain.from_iterable([frame['actions'] for frame in data[sid].values()])
    counter = Counter(data_concatenated)

    if cfg.include_walking:
        ACTIONS_ID = {0: 'NA', 1: 'Crossing', 2:'Waiting', 3:'Queueing', 4: 'Walking', 5:'Talking', 6:'Dancing', 7:'Jogging'}
        total_bboxes = sum(counter.values())
        for id, frequency in counter.items():
            print(f'{ACTIONS_ID[id]} ({id}): {frequency}, {frequency/total_bboxes}')
    else:
        ACTIONS_ID = {0: 'NA', 1: 'Crossing', 2: 'Waiting', 3: 'Queueing', 4: 'Talking', 5: 'Dancing', 6: 'Jogging'}
        total_bboxes = sum(counter.values())
        for id, frequency in counter.items():
            print(f'{ACTIONS_ID[id]} ({id}): {frequency}, {frequency / total_bboxes}')
    return data

def collective_all_frames(anns):
    return [(s,f)  for s in anns for f in anns[s] ]


class CollectiveDataset(data.Dataset):
    """
    Characterize collective dataset for pytorch
    """
    def __init__(self,anns,frames,images_path,image_size,feature_size,num_boxes=13,num_frames=10,is_training=True,is_finetune=False, is_gnn=False):
        self.anns=anns
        self.frames=frames
        self.images_path=images_path
        self.image_size=image_size
        self.feature_size=feature_size

        self.num_boxes=num_boxes
        self.num_frames=num_frames

        self.is_training=is_training
        self.is_finetune=is_finetune
        self.is_gnn = is_gnn

        self.frames = self.adjust_frames_to_work_with_batches()
        self.build_frame_dataset()

    def adjust_frames_to_work_with_batches(self):
        frames_df = pd.DataFrame(self.frames).groupby(0)
        frames_data = []
        for _, data_df in frames_df:
            data_idx_to_keep = len(data_df) - len(data_df) % self.num_frames
            frames_data.append(data_df.iloc[:data_idx_to_keep])
        return pd.concat(frames_data).values


    def build_frame_dataset(self):
        self.frame_dataset = defaultdict(list)
        self.frame_dataset_indexes = []
        if self.is_gnn:
            for j in range(len(self.frames)):
                if j + self.num_frames >= len(self.frames):
                    break
                for i in range(j, j+self.num_frames):
                    sequence_id, labled_frame_id = self.frames[i]
                    if len(self.frame_dataset[sequence_id]):
                        if len(self.frame_dataset[sequence_id][-1]) != self.num_frames:
                            self.frame_dataset[sequence_id][-1].append(labled_frame_id)
                        else:
                            self.frame_dataset[sequence_id].append([labled_frame_id])
                            self.frame_dataset_indexes.append((sequence_id, len(self.frame_dataset[sequence_id]) - 1))
                    else:
                        self.frame_dataset[sequence_id].append([labled_frame_id])
                        self.frame_dataset_indexes.append((sequence_id, len(self.frame_dataset[sequence_id]) - 1))
        else:
            for i in range(len(self.frames)):
                sequence_id, labled_frame_id = self.frames[i]
                if len(self.frame_dataset[sequence_id]):
                    if len(self.frame_dataset[sequence_id][-1]) != self.num_frames:
                        self.frame_dataset[sequence_id][-1].append(labled_frame_id)
                    else:
                        self.frame_dataset[sequence_id].append([labled_frame_id])
                        self.frame_dataset_indexes.append((sequence_id, len(self.frame_dataset[sequence_id]) - 1))
                else:
                    self.frame_dataset[sequence_id].append([labled_frame_id])
                    self.frame_dataset_indexes.append((sequence_id, len(self.frame_dataset[sequence_id]) - 1))


        pass




    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.frame_dataset_indexes)

    def __getitem__(self,index):
        """
        Generate one sample of the dataset
        """

        select_frames=self.frame_dataset_indexes[index]

        sample_inputs, sample_targets=self.load_samples_sequence(select_frames)

        return sample_inputs, sample_targets


    def load_samples_sequence(self,select_frames):
        """
        load samples sequence

        Returns:
            pytorch tensors
        """
        OH, OW=self.feature_size

        images, bboxes = [], []
        activities, actions = [], []
        bboxes_num=[]
        sid = select_frames[0]

        targets = []

        for i, fid in enumerate(self.frame_dataset[sid][select_frames[1]]):

            img = Image.open(self.images_path + '/seq%02d/frame%04d.jpg'%(sid,fid))

            img=transforms.functional.resize(img,self.image_size)
            img=np.array(img)

            # H,W,3 -> 3,H,W
            img=img.transpose(2,0,1)
            images.append(img)

            temp_boxes=[]
            for box in self.anns[sid][fid]['bboxes']:
                y1,x1,y2,x2,target=box
                w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH
                temp_boxes.append((w1,h1,w2,h2,target))

            temp_actions=self.anns[sid][fid]['actions'][:]
            bboxes_num.append(len(temp_boxes))

            while len(temp_boxes)!=self.num_boxes:
                temp_boxes.append((0,0,0,0, -1))
                temp_actions.append(-1)

            bboxes.append(temp_boxes)
            actions.append(temp_actions)

            activities.append(self.anns[sid][fid]['group_activity'])


        images = np.stack(images)
        activities = np.array(activities, dtype=np.int32)
        bboxes_num = np.array(bboxes_num, dtype=np.int32)
        bboxes=np.array(bboxes,dtype=np.float).reshape(-1,self.num_boxes,5)
        actions=np.array(actions,dtype=np.int32).reshape(-1,self.num_boxes)

        #convert to pytorch tensor
        images=torch.from_numpy(images).float()
        bboxes=torch.from_numpy(bboxes).float()
        actions=torch.from_numpy(actions).long()
        activities=torch.from_numpy(activities).long()
        bboxes_num=torch.from_numpy(bboxes_num).int()

        return (images, bboxes, bboxes_num), (actions, activities)


def return_dataset(cfg):
    if cfg.dataset_name == 'collective':
        print('-----' * 5)
        print('Sequence from 1-44')
        collective_read_dataset(cfg, cfg.data_path, list(range(1, 45)))
        print('Sequence from 44-72')
        collective_read_dataset(cfg, cfg.data_path, list(range(45, 73)))
        print('Training set')
        train_anns = collective_read_dataset(cfg, cfg.data_path, cfg.train_seqs)
        train_frames = collective_all_frames(train_anns)
        print('-----' * 5)
        print('Test set')
        test_anns = collective_read_dataset(cfg, cfg.data_path, cfg.test_seqs)
        test_frames = collective_all_frames(test_anns)
        print('-----' * 5)

        training_set = CollectiveDataset(train_anns, train_frames,
                                         cfg.data_path, cfg.image_size, cfg.out_size,
                                         num_frames=cfg.num_frames, is_training=True, is_finetune=False,
                                         is_gnn=cfg.training_stage == 2)

        validation_set = CollectiveDataset(test_anns, test_frames,
                                           cfg.data_path, cfg.image_size, cfg.out_size,
                                           num_frames=1, is_training=False, is_finetune=False,
                                           is_gnn=cfg.training_stage == 2)

    else:
        assert False

    print('Reading dataset finished...')
    print('%d train samples' % len(train_frames))
    print('%d test samples' % len(test_frames))
    return training_set, validation_set


