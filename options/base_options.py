import os
import argparse
import shutil
from termcolor import colored

class BaseOptions(object):

    def __init__(self):
        self.initialized = False
        self.parser = None
        self.opt = None
        # config for predefined method
        self.override_dict = {
            'InsDis':       ['RGB', False, 'bank', 'A', 'linear', 0.07],
            'CMC':          ['CMC', False, 'bank', 'C', 'linear', 0.07],
            'MoCo':         ['RGB', False, 'moco', 'A', 'linear', 0.07],
            'PIRL':         ['RGB', True,  'bank', 'A', 'linear', 0.07],
            'PatchSMem':    ['RGB', True,  'bank', 'A', 'linear', 0.07],
            'PatchS':       ['RGB', True,  'bank', 'A', 'linear', 0.07],
            'MoCov2':       ['RGB', False, 'moco', 'B', 'mlp',    0.2],
            'CMCv2':        ['CMC', False, 'moco', 'E', 'mlp',    0.2],
            'InfoMin':      ['RGB', False,  'moco', 'D', 'mlp',    0.15],
            'InfoMinNoJig': ['RGB', False,  'moco', 'D', 'mlp',    0.15],
            'PatchSMoco':   ['RGB', False,  'moco', 'D', 'mlp',    0.15],
        }

    def initialize(self, parser):
        # specify folder
        parser.add_argument('--data_folder', type=str, default='/data1/trinh/data/raw_data/Domain_Invariance/colon_class/example/',
                            help='path to data')
        parser.add_argument('--dataset_name', type=str, default='k16',
                            help='path to save model')
        # parser.add_argument('--model_path', type=str, default='./save_40',
        #                     help='path to save model')
        # parser.add_argument('--tb_path', type=str, default='./tb_40',
        #                     help='path to tensorboard')
        parser.add_argument('--model_path', type=str, default='./save_dump',
                            help='path to save model')
        parser.add_argument('--tb_path', type=str, default='./tb_dump',
                            help='path to tensorboard')
        # parser.add_argument('--tb_path', type=str, default='./tb',
        #                     help='path to tensorboard')


        # basics
        parser.add_argument('--print_freq', type=int, default=20,
                            help='print frequency')
        parser.add_argument('--save_freq', type=int, default=20,
                            help='save frequency')
        parser.add_argument('--batch_size', type=int, default=256,
                            help='batch_size')
        parser.add_argument('--batch_size_infer', type=int, default=512,
                            help='batch_size_infer')
        parser.add_argument('--image_size', type=int, default=224,
                            help='image_size')
        parser.add_argument('-j', '--num_workers', type=int, default=40,
                            help='num of workers to use')

        # optimization
        parser.add_argument('--epochs', type=int, default=200,
                            help='number of training epochs')
        parser.add_argument('--learning_rate', type=float, default=0.03,
                            help='learning rate')
        parser.add_argument('--lr_decay_epochs', type=str, default='120,160',
                            help='where to decay lr, can be a list')
        parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                            help='decay rate for learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-4,
                            help='weight decay')
        parser.add_argument('--momentum', type=float, default=0.9,
                            help='momentum for SGD')
        parser.add_argument('--cosine', action='store_true',
                            help='using cosine annealing')

        # method selection
        parser.add_argument('--method', default='Customize', type=str,
                            choices=['InsDis', 'CMC', 'CMCv2', 'MoCo', 'MoCov2',
                                     'PIRL', 'InfoMin', 'Customize','PatchS', 'PatchSMem', 'PatchSMoco'],
                            help='Choose predefined method. Configs will be override '
                                 'for all methods except for `Customize`, which allows '
                                 'for user-defined combination of methods')
        # method configuration
        parser.add_argument('--modal', default='RGB', type=str, choices=['RGB', 'CMC'],
                            help='single RGB modal, or two modalities in CMC')
        parser.add_argument('--jigsaw', action='store_true',
                            help='adding PIRL branch')
        parser.add_argument('--jigsaw_stitch', action='store_true',
                            help='adding PIRL branch')
        parser.add_argument('--jigsaw_ema', action='store_true',
                            help='use jigsaw ema')
        parser.add_argument('--jigsaw_aug', action='store_true',
                            help='use jigsaw ema')
        parser.add_argument('--jig_version', default='V0', type=str, choices=['V0', 'V1'],
                            help='2 or 4 loss')
        parser.add_argument('--mem', default='bank', type=str, choices=['bank', 'moco'],
                            help='memory mechanism: memory bank, or moco encoder cache')

        # model setup
        parser.add_argument('--arch', default='resnet50', type=str,
                            help='e.g., resnet50, resnext50, resnext101'
                                 'and their wider variants, resnet50x4')
        parser.add_argument('--preImageNet', action='store_true',
                            help='using pretrained weight from ImageNet')
        parser.add_argument('-d', '--feat_dim', default=128, type=int,
                            help='feature dimension for contrastive loss')
        parser.add_argument('-k', '--nce_k', default=65536, type=int,
                            help='number of negatives')
        parser.add_argument('-m', '--nce_m', default=0.5, type=float,
                            help='momentum for memory update')
        parser.add_argument('-t', '--nce_t', default=0.07, type=float,
                            help='temperature')
        parser.add_argument('--alpha', default=0.999, type=float,
                            help='momentum coefficients for moco encoder update')
        parser.add_argument('--head', default='linear', type=str,
                            choices=['linear', 'mlp'], help='projection head')

        # resume
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')

        # Parallel setting
        parser.add_argument('--world-size', default=-1, type=int,
                            help='number of nodes for distributed training')
        parser.add_argument('--rank', default=-1, type=int,
                            help='node rank for distributed training')
        parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                            help='url used to set up distributed training')
        parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
        parser.add_argument('--seed', default=None, type=int,
                            help='seed for initializing training. ')
        parser.add_argument('--gpu', default=None, type=int,
                            help='GPU id to use.')
        parser.add_argument('--multiprocessing-distributed', action='store_true',
                            help='Use multi-processing distributed training to launch '
                                 'N processes per node, which has N GPUs. This is the '
                                 'fastest way to use PyTorch for either single node or '
                                 'multi node data parallel training')

        return parser

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def modify_options(self, opt):
        raise NotImplementedError

    def override_options(self, opt):
        # override parameters for predefined method
        if opt.method in self.override_dict.keys():
            opt.modal = self.override_dict[opt.method][0]
            opt.jigsaw = self.override_dict[opt.method][1]
            opt.mem = self.override_dict[opt.method][2]
            opt.aug = self.override_dict[opt.method][3]
            opt.head = self.override_dict[opt.method][4]
            opt.nce_t = self.override_dict[opt.method][5]
        return opt

    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser('arguments options')
            parser = self.initialize(parser)
            self.parser = parser
            self.initialized = True
        else:
            parser = self.parser

        opt = parser.parse_args()
        opt = self.modify_options(opt)
        self.opt = opt

        self.print_options(opt)

        return opt

    @staticmethod
    def check_log_dir(log_dir):
        # check if log dir exist
        if os.path.isdir(log_dir):
            color_word = colored('WARMING', color='red', attrs=['bold', 'blink'])
            print('%s: %s exist!' % (color_word, colored(log_dir, attrs=['underline'])))
            while (True):
                print('Select Action: d (delete)/ q (quit)', end='')
                key = input()
                if key == 'd':
                    shutil.rmtree(log_dir)
                    break
                elif key == 'q':
                    exit()
                else:
                    color_word = colored('ERR', color='red')
                    print('---[%s] Unrecognized character!' % color_word)
        return