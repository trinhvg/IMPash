"""
IMPaSh self-supervised pre-training (DDP).

The paper encoder uses:
    --method InfoMin --jigsaw_stitch --jigsaw_ema --jig_version V0 --cosine
See README.md ("IMPaSh configuration") for the full resolved config.
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.utils.data.distributed
import torch.multiprocessing as mp

from options.train_options import TrainOptions
from learning.contrast_trainer_stitch import ContrastTrainer
from networks.build_backbone import build_model
from datasets.util import build_contrast_loader
from memory.build_memory import build_mem


# share DataLoader batches through files in /dev/shm instead of passing file descriptors;
# the default 'file_descriptor' strategy intermittently fails with 'Bad file descriptor' on this server
mp.set_sharing_strategy('file_system')


def main():
    args = TrainOptions().parse()

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        raise NotImplementedError('Currently only DDP training')


def main_worker(gpu, ngpus_per_node, args):

    # initialize trainer and ddp environment
    trainer = ContrastTrainer(args)
    trainer.init_ddp_environment(gpu, ngpus_per_node)

    # build model: encoder g + projectors p1 (InfoMin) / p2 (PatchShuffling), and its momentum copy
    model, model_ema = build_model(args)

    # build dataset
    train_dataset, train_loader, train_sampler = \
        build_contrast_loader(args, ngpus_per_node)

    # build memory: two momentum queues (k1, k2) when --jigsaw_ema
    contrast = build_mem(args, len(train_dataset))
    contrast.cuda()

    # build criterion and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # wrap up models
    model, model_ema, optimizer = trainer.wrap_up(model, model_ema, optimizer)

    # optional step: synchronize memory
    trainer.broadcast_memory(contrast)

    # check and resume a model
    start_epoch = trainer.resume_model(model, model_ema, contrast, optimizer)

    # init tensorboard logger
    trainer.init_tensorboard_logger()

    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        trainer.adjust_learning_rate(optimizer, epoch)

        outs = trainer.train(epoch, train_loader, model, model_ema,
                             contrast, criterion, optimizer)

        # log to tensorboard
        trainer.logging(epoch, outs, optimizer.param_groups[0]['lr'])

        # save model
        trainer.save(model, model_ema, contrast, optimizer, epoch)


if __name__ == '__main__':
    main()
