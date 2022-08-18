from .mem_bank import RGBMem, CMCMem
from .mem_moco import RGBMoCo, RGBMoCoJig, RGBMoCoJig_v1, CMCMoCo


def build_mem(opt, n_data):
    if opt.mem == 'bank':
        mem_func = RGBMem if opt.modal == 'RGB' else CMCMem
        memory = mem_func(opt.feat_dim, n_data,
                          opt.nce_k, opt.nce_t, opt.nce_m)
    elif opt.mem == 'moco':
        if opt.jigsaw_ema:
            if opt.jig_version == 'V0':
                mem_func = RGBMoCoJig if opt.modal == 'RGB' else CMCMoCo
                memory = mem_func(opt.feat_dim, opt.nce_k, opt.nce_t)
            else:
                mem_func = RGBMoCoJig_v1 if opt.modal == 'RGB' else CMCMoCo
                memory = mem_func(opt.feat_dim, opt.nce_k, opt.nce_t)
        else:
            mem_func = RGBMoCo if opt.modal == 'RGB' else CMCMoCo
            memory = mem_func(opt.feat_dim, opt.nce_k, opt.nce_t)
    else:
        raise NotImplementedError(
            'mem not suported: {}'.format(opt.mem))

    return memory
