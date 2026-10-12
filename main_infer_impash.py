"""
Test the IMPaSh encoder + linear classifier on the target domain (K16) (DDP).

Writes {model_path}_infer_40/<run>/stat.json, collected by export_acc/export_acc_9.py.
Add --colornorm for the stain-normalisation baselines.
"""
from __future__ import print_function
import json

import torch
import torch.nn as nn
import pandas as pd
import torch.multiprocessing as mp

from options.test_options import TestOptions
from learning.linear_trainer_ema import LinearTrainer
from networks.build_backbone import build_model
from networks.build_linear import build_linear, build_linear_head, build_linear_jigsaw
from datasets.util_v0 import build_test_loader


def main():
    args = TestOptions().parse()

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        raise NotImplementedError('Currently only DDP training')


def main_worker(gpu, ngpus_per_node, args):

    # initialize trainer and ddp environment
    trainer = LinearTrainer(args)
    trainer.init_ddp_environment(gpu, ngpus_per_node)

    # build encoder and classifier
    model, model_ema = build_model(args)

    if args.keephead == 'head':
        classifier = build_linear_head(args)
    elif args.keephead == 'jigsaw':
        classifier = build_linear_jigsaw(args)
    else:
        classifier = build_linear(args)

    print('args.keephead = ', args.keephead)
    print(classifier)

    # build dataset
    train_loader, val_loader, train_sampler = \
        build_test_loader(args, ngpus_per_node)

    criterion = nn.CrossEntropyLoss().cuda()

    if not args.ema_feat:
        model_ema = False
        # load pre-trained ckpt for encoder and classifier
        model = trainer.load_encoder_weights_all(model, model_ema)
        classifier = trainer.load_classifier_weights(classifier)

        # wrap up models
        model, classifier = trainer.wrap_up(model, classifier, model_ema)
    else:
        # load pre-trained ckpt for encoder and classifier
        model, _ = trainer.load_encoder_weights_all(model, model_ema)
        classifier = trainer.load_classifier_weights(classifier)

        # wrap up models
        model, classifier, model_ema = trainer.wrap_up(model, classifier, model_ema)

    # init tensorboard logger
    trainer.init_tensorboard_logger()

    print('args.ema_feat', args.ema_feat)

    if args.rank % ngpus_per_node == 0:
        top1_avg, losses_avg, output_stat, processing_time = trainer.test(0, val_loader, model, classifier, criterion, model_ema)
        outs = [top1_avg, losses_avg]
        trainer.logging(0, outs, train=False)

        for i in output_stat.keys():
            print(i, [output_stat[i]])

        # accuracy after merging K19 classes that map to the same K16 class
        # (MUC->DEB, STR<->MUS); printed for reference only
        conf_mat = output_stat['conf_mat']
        o = 0
        for i in range(9):
            o += conf_mat[i][i]
        o += conf_mat[2, 4]
        o += conf_mat[7, 5]
        o += conf_mat[5, 7]
        print(o / conf_mat.sum())

        log_file = f'{args.model_folder}/stat.json'
        json_data = {'acc': output_stat['acc'], 'processing_time_second': processing_time,
                     'cf': pd.Series({'conf_mat': output_stat['conf_mat']}).to_json(orient='records')}
        with open(log_file, 'w') as json_file:
            json.dump(json_data, json_file)


if __name__ == '__main__':
    main()
