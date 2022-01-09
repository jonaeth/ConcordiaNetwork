import torch
from Concordia.utils.torch_losses import cross_entropy


def convert_targets_to_right_shape(targets_actions, targets_activities):
    batch_size = targets_actions.shape[0]
    max_number_bboxes = targets_actions.shape[2]
    num_frames = targets_actions.shape[1]

    targets_actions = targets_actions.reshape((batch_size, num_frames, max_number_bboxes))
    targets_activities = targets_activities.reshape((batch_size, num_frames))

    targets_actions_nopad_same_shape = []
    targets_actions_nopad = []
    for b in range(batch_size):
        actions_of_batch = []
        number_of_bboxes_per_frame = [len([action for action in frame if action != -1]) for frame in
                                      targets_actions[b].tolist()]
        for i, N in enumerate(number_of_bboxes_per_frame):
            targets_actions_nopad.append(targets_actions[b][i, :N])
            actions_of_batch.append(targets_actions[b][i, :N])
        targets_actions_nopad_same_shape.append(actions_of_batch)

    targets_actions = torch.cat(targets_actions_nopad, dim=0).reshape(-1, )  # ALL_N,
    targets_activities = targets_activities.reshape(-1, )

    return targets_actions, targets_activities


def student_target_loss_function(student_predictions, targets):
    student_predictions_actions, student_predictions_activities = student_predictions
    target_actions, target_activities = targets
    target_actions, target_activities = convert_targets_to_right_shape(target_actions, target_activities)
    return cross_entropy(student_predictions_actions, target_actions) + cross_entropy(student_predictions_activities,
                                                                                      target_activities)


def actions_accuracy(student_predictions, targets):
    student_predictions_actions, _ = student_predictions
    target_actions, _ = convert_targets_to_right_shape(*targets)
    predicted_actions_labels = torch.argmax(student_predictions_actions, dim=1)
    actions_correct = torch.sum(torch.eq(predicted_actions_labels.int(), target_actions.int()).float())
    actions_accuracy = actions_correct.item() / student_predictions_actions.shape[0]
    return actions_accuracy


def activities_accuracy(student_predictions, targets):
    _, student_predictions_activities = student_predictions
    _, target_activities = convert_targets_to_right_shape(*targets)
    predicted_activities_labels = torch.argmax(student_predictions_activities, dim=1)
    activities_correct = torch.sum(torch.eq(predicted_activities_labels.int(), target_activities.int()).float())
    activities_accuracy = activities_correct.item() / student_predictions_activities.shape[0]
    return activities_accuracy
