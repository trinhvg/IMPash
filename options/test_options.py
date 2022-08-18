import os
from .base_options import BaseOptions


class TestOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--ckpt', type=str, default=None,
                            help='the checkpoint to test')
        parser.add_argument('--ckpt_class', type=str, default=None,
                            help='the checkpoint to test')
        parser.add_argument('--aug_linear', type=str, default='NULL',
                            choices=['NULL', 'SRA', 'RA'],
                            help='linear evaluation augmentation')
        parser.add_argument('--crop', type=float, default=0.2,
                            help='crop threshold for RandomResizedCrop')
        parser.add_argument('--n_class', type=int, default=9,
                            help='number of classes for linear probing')
        parser.add_argument('--finetune', action='store_true',
                            help='using finetune')
        parser.add_argument('--colornorm', action='store_true',
                            help='using colornorm in infer phrase')
        parser.add_argument('--colorAug', action='store_true',
                            help='using colorAug in training phrase')
        parser.add_argument('--infer_only', action='store_true',
                            help='using mixed precision')
        parser.add_argument('--keephead', type=str, default='None',
                            choices=['None', 'head', 'jigsaw'],
                            help='using mixed precision')
        parser.add_argument('--ema_feat', action='store_true',
                            help='using ema feature')
        #
        # parser.set_defaults(epochs=60)
        # parser.set_defaults(learning_rate=30)

        parser.set_defaults(epochs=60)
        parser.set_defaults(learning_rate=30)
        # parser.set_defaults(learning_rate=1e-3)
        parser.set_defaults(lr_decay_epochs='30,40,50')
        parser.set_defaults(lr_decay_rate=0.2)
        parser.set_defaults(weight_decay=0)

        return parser



    def modify_options(self, opt):
        opt = self.override_options(opt)
        # if opt.dataset_name in ['k19', 'k16', 'crc']:
        #     opt.n_class = 7
        # elif opt.dataset_name in ['crc']:
        #     opt.n_class = 5

        iterations = opt.lr_decay_epochs.split(',')
        opt.lr_decay_epochs = list([])
        for it in iterations:
            opt.lr_decay_epochs.append(int(it))

        # set up saving name
        if opt.ckpt:
            opt.model_name = opt.ckpt.split('/')[-2]
        elif opt.ckpt_class:
            opt.model_name = opt.ckpt_class.split('/')[-2]
        else:
            # print('warning: no pre-trained model!')
            print('warning: no unsupervised pre-trained model!')
            if opt.preImageNet:
                opt.model_name = '{}_ImageNet'.format(opt.dataset_name)
            else:
                opt.model_name = '{}_Scratch'.format(opt.dataset_name)

        if opt.cosine:
            opt.model_name = '{}_cosine'.format(opt.model_name)

        if opt.colorAug:
            opt.model_name = '{}_colorAug'.format(opt.model_name)


        if opt.infer_only:
            opt.model_name = opt.ckpt_class.split('/')[-2]
            opt.model_name = 'Infer_test_{}_{}_model_{}_colornorm_Macenko_{}_{}_'.format(opt.dataset_name, opt.image_size, opt.model_name, opt.colornorm, opt.aug_linear)
            opt.model_path = '{}_infer_40'.format(opt.model_path)
            opt.model_name += opt.ckpt_class.split('/')[-1]

        else:
            opt.model_name = '{}_linear_head_{}_emahead_{}_on{}_{}_{}'.format(
                opt.model_name, opt.keephead, opt.ema_feat, opt.dataset_name,  opt.aug_linear, opt.crop)
            if opt.keephead == 'jigsaw':
                opt.jigsaw = True

        if opt.finetune:
            opt.model_name = opt.model_name.replace('_linear_', '_finetune_')



        # create folders
        opt.model_folder = os.path.join(opt.model_path, opt.model_name)
        # self.check_log_dir(opt.model_folder)
        if not os.path.isdir(opt.model_folder):
            os.makedirs(opt.model_folder)

        opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
        # self.check_log_dir(opt.tb_folder)

        if not os.path.isdir(opt.tb_folder):
            os.makedirs(opt.tb_folder)

        return opt
