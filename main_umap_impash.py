"""
UMAP of K19 (source) vs K16 (target) features (Fig. 3) and silhouette scores (Table 3) (DDP).

Samples 3000 patches per domain, extracts features with the frozen encoder and writes
viz_final_eccv/{fig_name}0_class_labels.png, {fig_name}1_domain_labels.png and
{fig_name}0_silhouette_scores.json.
"""
from __future__ import print_function
from collections import OrderedDict

import torch
import torch.multiprocessing as mp

from options.test_options import TestOptions
from learning.linear_trainer_ema_viz import LinearTrainer
from networks.build_backbone_Stic import build_model
from networks.build_linear import build_linear
from datasets.util_viz import build_test_loader_visualize
from visualization.figure import BinaryMultimodalUmapPlot


def main():
    args = TestOptions().parse()

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        raise NotImplementedError('Currently only DDP training')


def post_epoch_visualization_umap(all_predictions, save_name, classes):
    cur_epoch = 0
    classes = list(classes.keys())

    # Extract feature and label predictions
    src_feats = all_predictions["source"]["all_feats"]
    src_labels = all_predictions["source"]["all_labels"]

    tgt_feats = all_predictions["target"]["all_feats"]
    tgt_labels = all_predictions["target"]["all_labels"]

    X = torch.cat([src_feats, tgt_feats])
    Y = torch.cat([src_labels, tgt_labels])

    # Extract domains
    domains = [0] * len(src_feats)
    domains += [1] * len(tgt_feats)
    domains = torch.tensor(domains)

    # Draw the figure
    umap_plot = BinaryMultimodalUmapPlot(X, Y, domains, classes,
                                         cur_epoch, "UMAP", save_name)

    return {"UMAP": umap_plot}


def load_encoder_weights_all(args, model, model_ema=False):
    """load pre-trained weights for encoder

    Args:
      model: pretrained encoder, should be frozen
    """
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location='cpu')

        state_dict = ckpt['model']
        if args.modal == 'RGB':
            # Unimodal (RGB) case
            encoder_state_dict = OrderedDict()
            for k, v in state_dict.items():
                k = k.replace('module.', '')
                encoder_state_dict[k] = v
            model.load_state_dict(encoder_state_dict)

            if model_ema:
                state_dict_ema = ckpt['model_ema']
                encoder_state_dict = OrderedDict()
                for k, v in state_dict_ema.items():
                    k = k.replace('module.', '')
                    encoder_state_dict[k] = v
                model_ema.load_state_dict(encoder_state_dict)
                return model, model_ema

        else:
            # Multimodal (CMC) case
            encoder1_state_dict = OrderedDict()
            encoder2_state_dict = OrderedDict()
            for k, v in state_dict.items():
                k = k.replace('module.', '')
                if 'encoder1' in k:
                    k = k.replace('encoder1.', '')
                    encoder1_state_dict[k] = v
                if 'encoder2' in k:
                    k = k.replace('encoder2.', '')
                    encoder2_state_dict[k] = v
            model.encoder1.load_state_dict(encoder1_state_dict)
            model.encoder2.load_state_dict(encoder2_state_dict)
        print('Pre-trained weights loaded!')
    elif args.preImageNet:
        print('Pre-trained weights on ImageNet loaded!')
    else:
        print('==============================')
        print('warning: no pre-trained model!')
        print('==============================')

    return model, False


def main_worker(gpu, ngpus_per_node, args):
    # initialize trainer and ddp environment
    trainer = LinearTrainer(args)
    trainer.init_ddp_environment(gpu, ngpus_per_node)

    # build encoder and classifier
    model, model_ema = build_model(args)
    classifier = build_linear(args)

    model_ema = False

    # build dataset: 3000 sampled patches from each domain
    train_loader, val_loader, train_sampler = \
        build_test_loader_visualize('k19', args, ngpus_per_node, colornorm=args.colornorm, colorAug=args.colorAug,
                                    num_sample=3000)

    train_loader_t, val_loader_t, train_sampler_t = \
        build_test_loader_visualize('k16', args, ngpus_per_node, colornorm=args.colornorm, num_sample=3000)

    # load pre-trained ckpt for encoder
    model, model_ema = load_encoder_weights_all(args, model, model_ema)

    # wrap up models
    model, classifier, model_ema = trainer.wrap_up(model, classifier, model_ema)

    # init tensorboard logger
    trainer.init_tensorboard_logger()

    all_predictions = {}
    feat_s = {}
    feat_t = {}

    if args.rank % ngpus_per_node == 0:
        all_feats_s, all_labels_s = trainer.predict(0, val_loader, model, model_ema)
        all_feats_t, all_labels_t = trainer.predict(0, val_loader_t, model, model_ema)

        # map the 9 K19 classes onto the 7 shared K16 classes
        k19_to_7 = {0.: 0., 1.: 1., 2.: 2., 3.: 3., 4.: 2., 5.: 4., 6.: 5., 7.: 4., 8.: 6.}

        all_labels_s = torch.tensor([k19_to_7[l] for l in all_labels_s.tolist()])
        all_labels_t = torch.tensor([k19_to_7[l] for l in all_labels_t.tolist()])

        feat_s["all_feats"] = all_feats_s
        feat_s["all_labels"] = all_labels_s

        feat_t["all_feats"] = all_feats_t
        feat_t["all_labels"] = all_labels_t

        all_predictions["source"] = feat_s
        all_predictions["target"] = feat_t

        print(args.fig_name)

        classes_16 = {'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'NORM': 4, 'STR': 5, 'TUM': 6}

        post_epoch_visualization_umap(all_predictions, args.fig_name, classes=classes_16)
        print('done')


if __name__ == '__main__':
    main()
