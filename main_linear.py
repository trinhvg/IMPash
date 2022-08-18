"""
DDP training for Linear Probing
"""


from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5"

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np

from options.test_options import TestOptions
from learning.linear_trainer import LinearTrainer
from networks.build_backbone import build_model
from networks.build_linear import build_linear, build_linear_head, build_linear_jigsaw
from datasets.util import build_linear_loader, build_linear_stain_aug_loader

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
    model, _ = build_model(args)
    if args.keephead == 'head':
        classifier = build_linear_head(args)
        print(args.keephead == 'head')
        print(classifier)
    elif args.keephead == 'jigsaw':
        classifier = build_linear_jigsaw(args)
    else:

        classifier = build_linear(args)
        print('args.keephead' == 'None')
        print(classifier)


    # build dataset
    if args.colorAug:
        train_dataset, train_loader, val_loader, train_sampler = \
            build_linear_stain_aug_loader(args, ngpus_per_node)
    else:
        train_dataset, train_loader, val_loader, train_sampler = \
            build_linear_loader(args, ngpus_per_node)

    # show_augment(train_dataset, 0)


    # build criterion and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.finetune:
        optimizer = torch.optim.SGD(list(classifier.parameters()) + list(model.parameters()),
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(),
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)


    # load pre-trained ckpt for encoder
    model = trainer.load_encoder_weights_all(model)

    # wrap up models
    model, classifier = trainer.wrap_up(model, classifier)

    # check and resume a classifier
    start_epoch = trainer.resume_model(classifier, optimizer)

    # init tensorboard logger
    trainer.init_tensorboard_logger()

    best_val_acc = 0
    temp_best = False
    # routine
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        trainer.adjust_learning_rate(optimizer, epoch)

        outs = trainer.train(epoch, train_loader, model, classifier,
                             criterion, optimizer)

        # log to tensorbard
        trainer.logging(epoch, outs, optimizer.param_groups[0]['lr'], train=True)


        # evaluation and logging
        if args.rank % ngpus_per_node == 0:
            outs = trainer.validate(epoch, val_loader, model,
                                    classifier, criterion)

            trainer.logging(epoch, outs, train=False)
            print(outs)

        val_acc = outs[0]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            temp_best = True
        # saving model
        trainer.save(classifier, optimizer, epoch, temp_best)
        if args.finetune:
            trainer.save_encoder(model, optimizer, epoch)


if __name__ == '__main__':
    main()
