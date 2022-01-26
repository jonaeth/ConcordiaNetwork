from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score

with open('psl_predictions_semi.txt', 'r') as fp:
    lines = fp.readlines()
    y_true = [eval(pred.split('_')[1]) for pred in lines]
    y_pred = [eval(pred.split('_')[0]) for pred in lines]

threshold = 0.75
print(threshold)
predicted_label = [1 if y > threshold else 0 for y in y_pred]
print(f'Accuracy Score: {accuracy_score(y_true, predicted_label)}')
print(f'F1 Score: {f1_score(y_true, predicted_label)}')
print(f'Recall Score: {recall_score(y_true, predicted_label)}')
print(f'Precision Score: {precision_score(y_true, predicted_label)}')

for i in range(1, 10):
    threshold = i/10
    print(threshold)
    predicted_label = [1 if y > threshold else 0 for y in y_pred]
    print(f'Accuracy Score: {accuracy_score(y_true, predicted_label)}')
    print(f'F1 Score: {f1_score(y_true, predicted_label)}')
    print(f'Recall Score: {recall_score(y_true, predicted_label)}')
    print(f'Precision Score: {precision_score(y_true, predicted_label)}')


