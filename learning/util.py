from __future__ import print_function

import torch
from sklearn.metrics import confusion_matrix
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
            # Note
            # res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def process_accumulated_output(output, batch_size, nr_classes):
    #
    def uneven_seq_to_np(seq):
        item_count = batch_size * (len(seq) - 1) + len(seq[-1])
        cat_array = np.zeros((item_count,) + seq[0][0].shape, seq[0].dtype)
        # BUG: odd len even
        if len(seq) < 2:
            return seq[0]
        for idx in range(0, len(seq) - 1):
            cat_array[idx * batch_size:
                      (idx + 1) * batch_size] = seq[idx]
        cat_array[(idx + 1) * batch_size:] = seq[-1]
        return cat_array

    proc_output = dict()
    true = uneven_seq_to_np(output['true'])
    # threshold then get accuracy
    logit = uneven_seq_to_np(output['logit'])

    pred = np.argmax(logit, axis=-1)
    # pred_c = [covert_dict[pred_c[idx]] for idx in range(len(pred_c))]
    acc = np.mean(pred == true)
    print('acc', acc)
    # print(classification_report(true, pred_c, labels=[0, 1, 2, 3]))
    # confusion matrix
    conf_mat = confusion_matrix(true, pred, labels=np.arange(nr_classes))
    proc_output.update(acc=acc, conf_mat=conf_mat,)
    return proc_output