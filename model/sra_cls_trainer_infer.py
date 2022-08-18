from torch.utils.data import DataLoader
import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from model.utils import get_logger
import os
from numpy import ndarray
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Tuple, Iterable, Optional
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


class SRAClsTrainer:

    def __init__(
            self,
            model: torch.nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            name_classes: Optional[Iterable[str]] = None,
            opt_lr: Optional[float] = 1,
            t_max: Optional[int] = 100,
            device: Optional[str] = "cpu",
            logger: Optional[object] = None,
            prefix: Optional[str] = None,
            loadpath: Optional[str] = None,
            freeze: Optional[bool] = True,
    ) -> None:
        """
        Create trainer for histopathology dataset.

        Parameters
        ----------
        model: callable
            Model to train.
        train_loader: callable
            Dataloader for training data.
        val_loader: callable
            Dataloader for validation data.
        opt_lr: float
            Adam learning rate. Default value is 1.
        t_max: int
            T max for the cosine LR scheduler. Should be equal to the number of epochs. Default value is 200.
        device: str, optional
            Choose whether to use cuda for training or not.
        logger: object, optional
            Logger to use for output. If None, create a new one.
        prefix: str, optional
            Prefix to add in front of file to discriminate them. Default value is empty
        loadpath: str, optional
            Path to pretrained model.
        freeze: bool
            If True, the model weights are frozen except for the final fully connected (fc3)
        """

        # Create logger is needed
        self.logger = logger
        if self.logger is None:
            self.logger = get_logger()

        # Model
        self.prefix = prefix
        self.device = device
        self.model = model.to(device=self.device)
        self.freeze = freeze
        self.name_classes = name_classes

        # Data loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss = torch.nn.CrossEntropyLoss()

        # Load path if existing
        self.load(loadpath)

        # Optimizer and objective
        self.optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=opt_lr,
            weight_decay=0,
        )

        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        self.optimizer = torch.optim.SGD(
            params=parameters,
            lr=opt_lr,
            weight_decay=0
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=0)

        # Output files
        self.writer = SummaryWriter(comment=prefix)

    def load(self, loadpath: str) -> None:

        if loadpath is None:
            self.logger.error('No weights provided ...')
            return

        # Load path if existing
        if os.path.exists(loadpath):
            self.logger.debug('Loading weights from: {} ...'.format(loadpath))
            stat = torch.load(loadpath)
            err = self.model.load_state_dict(stat['model_state_dict'], strict=True)

            # Sanity check of the matching keys of the model
            self.logger.debug('Checking keys ...')
            if len(err.missing_keys) != 0:
                # Missing keys should be 3rd fully connected to train
                is_known = ['encoder_q.fc.3' in key for key in err.missing_keys]
                if not all(is_known):
                    unknown = np.array(err.missing_keys)[np.logical_not(is_known)]
                    raise Exception('Unexpected missing keys: {}'.format(unknown))
            if len(err.unexpected_keys) != 0:
                # Unexpected keys can be the Queue and encorder_q that are not used here.
                is_known = ['queue' in key or 'encoder_k' in key for key in err.unexpected_keys]
                if not all(is_known):
                    unknown = np.array(err.unexpected_keys)[np.logical_not(is_known)]
                    raise Exception('Unexpected missing keys: {}'.format(unknown))

            if self.freeze:
                self.logger.debug('Freezing layers ...')
                for name, param in self.model.named_parameters():
                    if 'encoder_q.fc.3' not in name:
                        param.requires_grad = False

            # self.logger.debug('Init model fc3 weights with N(0, 0.01) ...')
            # self.model.encoder_q.fc[3].weight.data.normal_(mean=0.0, std=0.01)
            # self.model.encoder_q.fc[3].bias.data.zero_()

        else:
            self.logger.error('Weights not found: {}'.format(loadpath))
            raise Exception("Non-valid loading path ...")

    @staticmethod
    def init_seed(seed: Optional[int] = 0) -> None:
        """
        Set seed for torch, numpy and random for reproducibility.

        Parameters
        ----------
        seed: int, optional
            Seed value for reproducibility. Default value is 0.

        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train(self, n_epochs: Optional[int] = 200) -> None:
        """
        Train model for a certain numbr of epochs.

        Parameters
        ----------
        n_epochs: int, optional
            Number of epochs to train. Default value is 200.
        """

        # Initialize seeds and loss max
        self.init_seed()
        best_valid_losses = np.Inf

        # Iterate over epochs


        y_pred_v, y_cgt_v, loss_eval, _ = self.eval()
        conf_mat = confusion_matrix(y_pred_v.argmax(axis=1), y_cgt_v, labels=np.arange(9))
        out = 0
        for i in range(9):
            if i == 2:
                b = (conf_mat[i][i] + conf_mat[4][i]) / conf_mat[:, i].sum()
            elif i == 7:
                b = (conf_mat[i][i] + conf_mat[5][i]) / conf_mat[:, i].sum()
            elif i in [4,5]:
                continue
            else:
                b = ((conf_mat[i][i]) / conf_mat[:, i].sum())
            print(b)
            out +=b
        print('all:', out/7 )
        print(conf_mat)
        o = 0
        for i in range(9):
            o += conf_mat[i][i]
        o += conf_mat[4,2]
        o += conf_mat[5,7]
        acc = o/conf_mat.sum()
        print(acc)

        metric_val = self.metrics(y_cgt_v, y_pred_v.argmax(axis=1), self.name_classes)

        # Scheduler step
        self.scheduler.step()

        # Log writer
        self.logger.debug('Loss val: {:.3f}'.format(loss_eval))
        self.logger.debug('F1 val:\n\t{}'.format("\t".join(["{}: {:.3f}".format(a, b) for a, b in metric_val.items()])))
        self.writer.add_scalars('Loss', {'val': loss_eval}, 0)
        self.writer.add_scalars('AccuracyVal', metric_val, 0)


    @staticmethod
    def metrics(cgt: ndarray, pred: ndarray, name_classes: Optional[Iterable[str]] = None):
        """
        Compute metrics (accuracy) for all classes

        Parameters
        ----------
        cgt: ndarray (, N)
            Ground truth of classes
        pred: ndarray (,N)
            Predicted classes
        name_classes: Iterable of string (, C)
            Classes names

        Returns
        -------
        metrics: dict
            Dictionary with name od the classes as entries and metric as values. "ALL" is used for the overall
            performance of the prediction.
        """
        # Compute accuracy over all classes
        results = {'ALL': f1_score(y_true=cgt, y_pred=pred, average='weighted')}
        # Check if name of classes fed otherwise returns
        if name_classes is None:
            return results
        # Accuracy over classes
        for i, name in enumerate(name_classes):
            results[name] = f1_score(y_true=np.array(cgt) == i, y_pred=np.array(pred) == i, average='binary')
        return results

    def eval(self) -> Tuple[ndarray, ndarray, float, float]:
        """
        Evaluate model on dataset

        Returns
        -------
        y_preds: ndarray
            Output prediction probabilities.
        y_labels: ndarray
            Classes ground truth.
        loss: float
            Average loss over all batches.
        accuracy: float
            Average accuracy over all batches.
        """

        self.model.eval()

        n_correct = []
        n_losses = []
        y_preds = []
        y_labels = []

        for x_img, y_label in tqdm(self.val_loader, desc='val'):

            # Use cuda or not
            x_img = x_img.to(self.device)
            y_label = y_label.to(self.device)

            # Calculate loss and metrics
            y_pred = self.model(x_img)
            loss = self.loss(y_pred, y_label)

            # Append accuracy and loss
            n_correct.extend(y_label.eq(y_pred.argmax(dim=1)).detach().cpu().numpy())
            n_losses.append(loss.detach().cpu().numpy())
            y_preds.extend(y_pred.detach().cpu().numpy())
            y_labels.extend(y_label.detach().cpu().numpy())

        return np.array(y_preds), np.array(y_labels), np.mean(n_losses), np.mean(n_correct)

    def save(self, path) -> None:
        """
        Save model and optimizer state.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, path)


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

# SRA
# array([[596,   0,   1,   0,   0,   0,   0,   2,   0],
#        [ 20, 622,   3,   0,   0,   0,   0,   0,   0],
#        [  0,   1, 170, 116,   0,   0,   0,   4,  68],
#        [  0,   0,   0, 254,   0,   0,   0,   0,   3],
#        [  7,   2, 303,   0,   0,   0,   0,  20,   0],
#        [  2,   0,  76,  18,   0,   0,   0, 356,   3],
#        [  0,   0,   4, 190,   0,   0, 624,   1,  11],
#        [  0,   0,  68,  18,   0,   0,   1, 239,   4],
#        [  0,   0,   0,  29,   0,   0,   0,   3, 536]])
# 0.8457142857142858

# a = 596 + 622 + 170+303+254+624+356+239+536 = 3700
# b = 544 + 387 + 171+249 +337+622+304+285+563 = 3462
# MoCo
# [[601   2   2   0   0   0   0   2   0]
#  [ 14 620   1   0   0   0   0   0   0]
#  [  0   1 169  73   0   0   0   3  27]
#  [  0   0   0 286   0   0   0   0   3]
#  [  7   2 279   0   0   0   0  14   1]
#  [  2   0  66  14   0   0   0 345   3]
#  [  0   0   7 201   0   0 625   2  17]
#  [  0   0  99  23   0   0   0 253   4]
#  [  1   0   2  28   0   0   0   6 570]]
# 0.8566857142857143
