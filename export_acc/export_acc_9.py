"""
Collect Acc / Re / Pre / F1 from main_infer_impash.py outputs into a CSV.

Usage: python export_acc/export_acc_9.py [INFER_DIR]
INFER_DIR defaults to ./save_40_9_infer_40 (written by main_infer_impash.py --model_path ./save_40_9).
Each <INFER_DIR>/<run>/stat.json holds the 9x9 K19-class confusion matrix on K16; MUC is merged
into DEB and MUS into STR (the K16 label set), and macro metrics are averaged over the 7 K16 classes.
Writes ./Class9_Infer_test_k16_class_40.csv.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd


def f1(a, num_class):
    """macro F1"""
    f = 0
    for i in range(a.shape[0]):
        if a[i][i] == 0:
            f += 0
        else:
            f += (2 * a[i][i] / a[:, i].sum() * a[i][i] / a[i, :].sum()) / (
                    a[i][i] / a[:, i].sum() + a[i][i] / a[i, :].sum())
    return f / num_class


def weighted_f1(a, num_class):
    """support-weighted F1"""
    all = a.sum()
    f = 0
    for i in range(a.shape[0]):
        if a[i][i] == 0:
            f += 0
        else:
            w = a[:, i].sum() / all
            f += w * (2 * a[i][i] / a[:, i].sum() * a[i][i] / a[i, :].sum()) / (
                    a[i][i] / a[:, i].sum() + a[i][i] / a[i, :].sum())
    return f


def acc(a, num_class):
    """accuracy"""
    f = 0
    a_sum = 0
    for i in range(a.shape[0]):
        f += a[i][i]
        a_sum += a[:, i].sum()
    return f / a_sum


def recall(a, num_class):
    """macro recall"""
    f = 0
    for i in range(a.shape[0]):
        if a[i][i] == 0:
            f += 0
        else:
            f += a[i][i] / a[:, i].sum()
    return f / num_class


def cf(a, num_class):
    """per-class recall"""
    f = []
    for i in range(a.shape[0]):
        if a[i][i] == 0:
            f.append(0)
        else:
            f.append(a[i][i] / a[:, i].sum())
    return f


def precision(a, num_class):
    """macro precision"""
    f = 0
    for i in range(a.shape[0]):
        if a[i][i] == 0:
            f += 0
        else:
            f += a[i][i] / a[i].sum()
    return f / num_class


def group_class(test_c, name):
    """merge K19 MUC (4) into DEB (2) and MUS (5) into STR (7)"""
    if 'crc' in name:
        test_c[7] += test_c[5]
        test_c[5] -= test_c[5]

    test_c[:, 2] += test_c[:, 4]
    test_c[:, 7] += test_c[:, 5]

    test_c[:, 5] -= test_c[:, 5]
    test_c[:, 4] -= test_c[:, 4]
    return test_c


path = sys.argv[1] if len(sys.argv) > 1 else './save_40_9_infer_40'

name_list = []
test_accs = []
test_recalls = []
test_precisions = []
test_f1s = []
test_weighted_f1s = []
test_times = []
test_cfs = []

for name in sorted(os.listdir(path)):
    json_dirs = glob.glob(f'{path}/{name}/stat.json')
    if len(json_dirs) == 0:
        continue
    print(json_dirs[0])

    with open(json_dirs[0]) as json_file:
        data = json.load(json_file)

    if name.split('_')[2][:3] == 'crc':
        num_class = 5
    else:
        num_class = 7
    test_c = np.array(eval(data['cf'])[0])
    test_c = group_class(test_c, name).transpose()

    test_acc = acc(test_c, num_class)
    print(test_c)

    name_list.append(name)
    test_accs.append(test_acc)
    test_times.append(data.get('processing_time_second', 0))
    test_recalls.append(recall(test_c, num_class))
    test_precisions.append(precision(test_c, num_class))
    test_f1s.append(f1(test_c, num_class))
    test_weighted_f1s.append(weighted_f1(test_c, num_class))
    test_cfs.append(cf(test_c, num_class))
    print(test_acc, f1(test_c, num_class))

df = pd.DataFrame(list(zip(name_list, test_accs, test_times, test_recalls, test_precisions,
                           test_f1s, test_weighted_f1s, test_cfs)),
                  columns=['name_list', 'test_acc', 'test_time', 'test_recall', 'test_precision',
                           'test_f1', 'test_weighted_f1', 'test_cf'])
out_csv = os.path.join(os.getcwd(), 'Class9_Infer_test_k16_class_40.csv')
print(out_csv)
print(df)
df.to_csv(out_csv)
