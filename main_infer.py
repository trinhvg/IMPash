"""
DDP training for Linear Probing
"""


from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

import torch
import torch.nn as nn
import pandas as pd
import torch.multiprocessing as mp

from options.test_options import TestOptions
from learning.linear_trainer import LinearTrainer
from networks.build_backbone import build_model
from networks.build_linear import build_linear, build_linear_head
from datasets.util import build_linear_loader, build_test_loader

def main():
    args = TestOptions().parse()

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        raise NotImplementedError('Currently only DDP training')

def acc(cf):
    out = 0
    for i in range(cf.shape[0]):
        out += cf[i][i]
    return out/cf.shape[0]

def main_worker(gpu, ngpus_per_node, args):

    # initialize trainer and ddp environment
    trainer = LinearTrainer(args)
    trainer.init_ddp_environment(gpu, ngpus_per_node)

    # build encoder and classifier
    model, _ = build_model(args)
    classifier = build_linear(args)

    # build dataset
    train_loader, val_loader, train_sampler = \
        build_test_loader(args, ngpus_per_node)

    # build criterion and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # load pre-trained ckpt for encoder
    model = trainer.load_encoder_weights(model)
    classifier = trainer.load_classifier_weights(classifier)

    # wrap up models
    model, classifier = trainer.wrap_up(model, classifier)

    # check and resume a classifier
    # start_epoch = trainer.resume_model(classifier, optimizer)

    # init tensorboard logger
    trainer.init_tensorboard_logger()

    print(args.rank, ngpus_per_node)
    if args.rank % ngpus_per_node == 0:
        # % Modulo	Remainder when a is divided by b
        top1_avg, losses_avg, output_stat, processing_time = trainer.test(0, val_loader, model, classifier, criterion)
        outs = [top1_avg, losses_avg]
        trainer.logging(0, outs, train=False)

        for i in output_stat.keys():
            print(i, [output_stat[i]])

        conf_mat = output_stat['conf_mat']
        o = 0
        for i in range(9):
            o += conf_mat[i][i]
        o += conf_mat[2, 4]
        o += conf_mat[7, 5]
        acc = o/conf_mat.sum()
        print(acc)


        import json
        log_file = f'{args.model_folder}/stat.json'
        json_data = {}
        # json stat log file, update and overwrite
        if not os.path.isfile(log_file):
            with open(log_file, 'w') as json_file:
                json.dump({}, json_file)  # create empty file
        # with open(log_file) as json_file:
        #     json_data = json.load(json_file)

        # current_epoch_dict = {net_name: out}
        # json_data.update(current_epoch_dict)
        json_data = output_stat
        # if ('crc' in args.model_name) or ('crc' in args.dataset_name):
        #     output_stat['conf_mat'] = output_stat['conf_mat'][:5, :5]
        # output_stat['acc'] = acc(output_stat['acc'])
        json_data = {'acc': output_stat['acc'], 'processing_time_second': processing_time,
                     'cf': pd.Series({'conf_mat': output_stat['conf_mat']}).to_json(orient='records')}
        with open(log_file, 'w') as json_file:
            json.dump(json_data, json_file)



    # routine
    # for epoch in range(start_epoch, args.epochs + 1):
    #     train_sampler.set_epoch(epoch)
    #     trainer.adjust_learning_rate(optimizer, epoch)
    #
    #     outs = trainer.train(epoch, train_loader, model, classifier,
    #                          criterion, optimizer)
    #
    #     # log to tensorbard
    #     trainer.logging(epoch, outs, optimizer.param_groups[0]['lr'], train=True)
    #
    #     # evaluation and logging
    #     print(args.rank, ngpus_per_node)
    #     if args.rank % ngpus_per_node == 0:
    #         outs = trainer.validate(epoch, val_loader, model,
    #                                 classifier, criterion)
    #         trainer.logging(epoch, outs, train=False)
    #
    #     # saving model
    #     trainer.save(classifier, optimizer, epoch)


if __name__ == '__main__':
    main()
