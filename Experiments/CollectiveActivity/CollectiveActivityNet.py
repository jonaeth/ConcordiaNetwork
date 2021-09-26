from Student import Student
import torch
import numpy as np
import torch.nn.functional as F

class CollectiveActivityNet(Student):
    def __init__(self, neural_network_model, device, cfg):
        super(CollectiveActivityNet).__init__(neural_network_model)
        self.model = neural_network_model
        self.device = device
        self.cfg = cfg

    def predict(self, batch_data):
        self.model.train() ## This should be set to not be run during actual inference on Test
        self.model.apply(self.set_bn_eval)
        batch_data=[b.to(device=self.device) for b in batch_data]
        batch_size=batch_data[0].shape[0]
        num_frames=batch_data[0].shape[1]
        bounding_boxes = batch_data[4]

        actions_scores,activities_scores=self.model((batch_data[0],batch_data[1][:, :, :, :-1],bounding_boxes))

        actions_in = batch_data[2].reshape((batch_size, num_frames, self.cfg.num_boxes))
        activities_in = batch_data[3].reshape((batch_size, num_frames))
        bboxes_num = batch_data[4].reshape(batch_size, num_frames)

        actions_in_nopad=[]
        actions_in_nopad_same_shape = []

        for b in range(batch_size):
            actions_of_batch = []
            for i, N in enumerate(bboxes_num[b]):
                actions_in_nopad.append(actions_in[b][i, :N])
                actions_of_batch.append(actions_in[b][i, :N])
            actions_in_nopad_same_shape.append(actions_of_batch)

        bounding_boxes = batch_data[1].detach().cpu().numpy()


        actions_scores_original_size = self._convert_action_scores_to_original_shape(actions_scores, bboxes_num, batch_size)
        bboxes_nopad = self._remove_padding_from_bounding_boxes(bounding_boxes, bboxes_num, batch_size)
        action_targets = [[np.identity(self.cfg.num_actions)[action.cpu().numpy()] for action in batch] for batch in actions_in_nopad_same_shape]

        return [actions_scores, activities_scores], [bboxes_nopad, actions_scores_original_size, action_targets]


    def compute_loss(self, predictions, targets):
        predicted_action_scores, predicted_activitie_scores = predictions[0], predictions[1]
        target_action_scores, target_group_activity_scores = targets[0], targets[1]
        target_action_scores = torch.cat(target_action_scores, dim=0).reshape(-1, )
        target_group_activity_scores = target_group_activity_scores.reshape(-1, )
        individual_actions_loss = F.cross_entropy(predicted_action_scores, target_action_scores, weight=None)
        group_activity_loss = F.cross_entropy(predicted_activitie_scores, target_group_activity_scores)
        return individual_actions_loss, group_activity_loss


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


    @staticmethod
    def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()