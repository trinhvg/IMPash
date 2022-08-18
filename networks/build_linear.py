import torch.nn as nn


def build_linear(opt):
    n_class = opt.n_class
    arch = opt.arch
    if arch.endswith('x4'):
        n_feat = 2048 * 4
    elif arch.endswith('x2'):
        n_feat = 2048 * 2
    else:
        n_feat = 2048
    if opt.ema_feat:
        n_feat += 2048
    classifier = nn.Linear(n_feat, n_class)
    return classifier


def build_linear_head(opt):
    n_class = opt.n_class
    arch = opt.arch
    if arch.endswith('x4'):
        n_feat = 2048 * 4
    elif arch.endswith('x2'):
        n_feat = 2048 * 2
    else:
        n_feat = 2048
    n_feat = 128
    if opt.ema_feat:
        n_feat += 128
    classifier = nn.Linear(n_feat, n_class)
    return classifier


def build_linear_jigsaw(opt):
    n_class = opt.n_class
    arch = opt.arch
    if arch.endswith('x4'):
        n_feat = 2048 * 4
    elif arch.endswith('x2'):
        n_feat = 2048 * 2
    else:
        n_feat = 2048

    n_feat = 256
    if opt.ema_feat:
        n_feat += 128
    classifier = nn.Linear(n_feat, n_class)
    return classifier
