from Preprocessor import Preprocessor
from Analysis import Analysis
import numpy as np

n_fold = 10
train_file = "data/SemEval2018-T3-taskA.txt"
test_file = "data/SemEval2018-T3_input_test_taskA.txt"

d = Preprocessor()
train_data, test_data = d.process_data(train_file, test_file, load_saved_data=False)
k_fold_train, k_fold_valid = Preprocessor.split_kfolds(train_data, n_fold)

Analysis_predict = None

Analysis_f1_scores = []
print(len(k_fold_train), len(k_fold_valid))
for i in range(len(k_fold_train)):
    print("====================Fold %d=================" % (i + 1))
    _, _, Analysis_pred_test, Analysis_f1_score = Analysis().predict(k_fold_train[i], k_fold_valid[i], test_data)
    Analysis_f1_scores.append(Analysis_f1_score)
    if Analysis_predict is None:
        Analysis_predict = Analysis_pred_test
    else:
        Analysis_predict = np.column_stack((Analysis_predict, Analysis_pred_test))

Analysis_predict = np.average(Analysis_predict, axis=1)
file_out    = open("predictions-taskA.txt", "w")
for i in range(len(Analysis_predict)):
    if i > 0:
        label = Analysis_predict[i]
        # print(test_data["raw_data"][i])
        if label > 0.5:
            label = 1
        else:
            label = 0
        file_out.write("%d\n" % label)

file_out.close()
Analysis_f1_scores = np.array(Analysis_f1_scores)
print("Final Analysis F1: %0.4f (+/- %0.4f)" % (Analysis_f1_scores.mean(), Analysis_f1_scores.std() * 2))
