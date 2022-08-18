from __future__ import print_function

import os
import sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from skimage import color
from torchvision import transforms, datasets

from .dataset import ImageFolderInstance, LinearInstance, LinearInstanceNorm, LinearInstanceColorAug
from .histo_data import prepare_colon_class_data, prepare_colon_class_crc_data, prepare_colon_class_k19val_data,\
    prepare_colon_class_k19crcBalance_data, prepare_colon_class_data_from_json
from .RandAugment import rand_augment_transform
from model.transform import get_supervised_train_augmentation, get_supervised_val_augmentation

import torch.nn as nn
import cv2
from random import shuffle


sys.path.insert(0, "/data1/trinh/code/tool/tiatoolbox/")
from tiatoolbox.tools.stainnorm import VahadaneNormalizer, MacenkoNormalizer

# sys.path.insert(0, "/data1/trinh/code/tool/HistomicsTK/")
from histomicstk.preprocessing.augmentation.color_augmentation import rgb_perturb_stain_concentration

# from histomicstk.preprocessing.augmentation.color_augmentation import rgb_perturb_stain_concentration


class StackTransform(object):
    """transform a group of images independently"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, imgs):
        return torch.stack([self.transform(crop) for crop in imgs])


class JigsawCrop(object):
    """Jigsaw style crop"""
    def __init__(self, n_grid=3, img_size=255, crop_size=64):
        self.n_grid = n_grid
        self.img_size = img_size
        self.crop_size = crop_size
        self.grid_size = int(img_size / self.n_grid)
        self.side = self.grid_size - self.crop_size

        yy, xx = np.meshgrid(np.arange(n_grid), np.arange(n_grid))
        self.yy = np.reshape(yy * self.grid_size, (n_grid * n_grid,))
        self.xx = np.reshape(xx * self.grid_size, (n_grid * n_grid,))

    def __call__(self, img):
        r_x = np.random.randint(0, self.side + 1, self.n_grid * self.n_grid)
        r_y = np.random.randint(0, self.side + 1, self.n_grid * self.n_grid)
        img = np.asarray(img, np.uint8)
        crops = []
        for i in range(self.n_grid * self.n_grid):
            crops.append(img[self.xx[i] + r_x[i]: self.xx[i] + r_x[i] + self.crop_size,
                         self.yy[i] + r_y[i]: self.yy[i] + r_y[i] + self.crop_size, :])
        crops = [Image.fromarray(crop) for crop in crops]
        return crops


class PatchShuffling(object):
    """Jigsaw style crop
    1st setting n_grid=3, img_size=255, crop_size=64
    2st setting n_grid=2, img_size=255, crop_size=56*2
    3st setting n_grid=3, img_size=224, crop_size=64

    """
    def __init__(self, n_grid=3, img_size=255, crop_size=64):
        self.n_grid = n_grid
        self.img_size = img_size
        self.crop_size = crop_size
        self.grid_size = int(img_size / self.n_grid)
        self.side = self.grid_size - self.crop_size

        yy, xx = np.meshgrid(np.arange(n_grid), np.arange(n_grid))
        self.yy = np.reshape(yy * self.grid_size, (n_grid * n_grid,))
        self.xx = np.reshape(xx * self.grid_size, (n_grid * n_grid,))

        self.re_yy = np.reshape(yy * self.crop_size, (n_grid * n_grid,))
        self.re_xx = np.reshape(xx * self.crop_size, (n_grid * n_grid,))

    def __call__(self, img):
        r_x = np.random.randint(0, self.side + 1, self.n_grid * self.n_grid)
        r_y = np.random.randint(0, self.side + 1, self.n_grid * self.n_grid)
        img = np.asarray(img, np.uint8)
        crops = []
        for i in range(self.n_grid * self.n_grid):
            crops.append(img[self.xx[i] + r_x[i]: self.xx[i] + r_x[i] + self.crop_size,
                         self.yy[i] + r_y[i]: self.yy[i] + r_y[i] + self.crop_size, :])
        shuffle(crops)

        shuffling_img = np.zeros([self.crop_size*self.n_grid, self.crop_size*self.n_grid, 3], dtype='uint8')
        for i in range(self.n_grid * self.n_grid):
            shuffling_img[self.re_xx[i]: self.re_xx[i] + self.crop_size, self.re_yy[i]: self.re_yy[i] + self.crop_size] \
                = crops[i]

        return Image.fromarray(shuffling_img)


class ReJigsawCrop(object):
    """Jigsaw style crop
    1st setting n_grid=3, img_size=255, crop_size=64
    2st setting n_grid=2, img_size=255, crop_size=56*2
    3st setting n_grid=3, img_size=224, crop_size=64

    """
    def __init__(self, n_grid=3, img_size=255, crop_size=64):
        self.n_grid = n_grid
        self.img_size = img_size
        self.crop_size = crop_size
        self.grid_size = int(img_size / self.n_grid)
        self.side = self.grid_size - self.crop_size

        yy, xx = np.meshgrid(np.arange(n_grid), np.arange(n_grid))
        self.yy = np.reshape(yy * self.grid_size, (n_grid * n_grid,))
        self.xx = np.reshape(xx * self.grid_size, (n_grid * n_grid,))

        self.re_yy = np.reshape(yy * self.crop_size, (n_grid * n_grid,))
        self.re_xx = np.reshape(xx * self.crop_size, (n_grid * n_grid,))

    def __call__(self, img):
        r_x = np.random.randint(0, self.side + 1, self.n_grid * self.n_grid)
        r_y = np.random.randint(0, self.side + 1, self.n_grid * self.n_grid)
        img = np.asarray(img, np.uint8)
        crops = []
        for i in range(self.n_grid * self.n_grid):
            crops.append(img[self.xx[i] + r_x[i]: self.xx[i] + r_x[i] + self.crop_size,
                         self.yy[i] + r_y[i]: self.yy[i] + r_y[i] + self.crop_size, :])
        shuffle(crops)

        rejig_img = np.zeros([self.crop_size*self.n_grid, self.crop_size*self.n_grid, 3], dtype='uint8')
        for i in range(self.n_grid * self.n_grid):
            rejig_img[self.re_xx[i]: self.re_xx[i] + self.crop_size, self.re_yy[i]: self.re_yy[i] + self.crop_size] \
                = crops[i]

        return Image.fromarray(rejig_img)


class Rotate(object):
    """rotation"""
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, img):
        angle = np.random.choice(self.angles)
        if isinstance(img, Image.Image):
            img = img.rotate(angle, fillcolor=(128, 128, 128))
            return img
        elif isinstance(img, np.ndarray):
            if angle == 0:
                pass
            elif angle == 90:
                img = np.flipud(np.transpose(img, (1, 0, 2)))
            elif angle == 180:
                img = np.fliplr(np.flipud(img))
            elif angle == 270:
                img = np.transpose(np.flipud(img), (1, 0, 2))
            else:
                img = Image.fromarray(img)
                img = img.rotate(angle, fillcolor=(128, 128, 128))
                img = np.asarray(img)
            return img
        else:
            raise TypeError('not supported type in rotation: ', type(img))


class RGB2RGB(object):
    """Dummy RGB transfer."""
    def __call__(self, img):
        return img


class RGB2Lab(object):
    """Convert RGB PIL image to ndarray Lab."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2lab(img)
        return img


class RGB2YCbCr(object):
    """Convert RGB PIL image to ndarray YCbCr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ycbcr(img)
        return img


class RGB2YDbDr(object):
    """Convert RGB PIL image to ndarray YDbDr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ydbdr(img)
        return img


class RGB2YPbPr(object):
    """Convert RGB PIL image to ndarray YPbPr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ypbpr(img)
        return img


class RGB2YIQ(object):
    """Convert RGB PIL image to ndarray YIQ."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2yiq(img)
        return img


class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


class GaussianBlur2(object):
    """Gaussian Blur version 2"""
    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class GaussianBlurBatch(object):
    """blur a batch of images on CPU or GPU"""
    def __init__(self, kernel_size, use_cuda=False, p=0.5):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        assert 0 <= p <= 1.0, 'p is out of range [0, 1]'
        self.p = p
        self.use_cuda = use_cuda
        if use_cuda:
            self.blur = self.blur.cuda()

    def __call__(self, imgs):

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)
        if self.use_cuda:
            x = x.cuda()

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        bsz = imgs.shape[0]
        n_blur = int(bsz * self.p)
        with torch.no_grad():
            if n_blur == bsz:
                imgs = self.blur(imgs)
            elif n_blur == 0:
                pass
            else:
                imgs_1, imgs_2 = torch.split(imgs, [n_blur, bsz - n_blur], dim=0)
                imgs_1 = self.blur(imgs_1)
                imgs = torch.cat([imgs_1, imgs_2], dim=0)

        return imgs


def build_transforms(aug, modal, use_memory_bank=True, image_size=224):
    if use_memory_bank:
        # memory bank likes 0.08
        crop = 0.08
    else:
        # moco cache likes 0.2
        crop = 0.2

    if modal == 'RGB':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        color_transfer = RGB2RGB()
    else:
        mean = [0.457, -0.082, -0.052]
        std = [0.500, 1.331, 1.333]
        color_transfer = RGB2YDbDr()
    normalize = transforms.Normalize(mean=mean, std=std)


    if aug == 'A':
        # used in InsDis, MoCo, PIRL
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(crop, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
        ])
    elif aug == 'B':
        # used in MoCoV2
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(crop, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur2()], p=0.5),
        ])
    elif aug == 'C':
        # used in CMC, CMCPIRL
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(crop, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        ])
    elif aug == 'D':
        # used in InfoMin
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(
            translate_const=int(image_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
        )
        # ccrop = transforms.CenterCrop(150)

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(crop, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomApply([GaussianBlur(22)], p=0.5),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                   ra_params,
                                   use_cmc=(modal == 'CMC')),
            transforms.RandomGrayscale(p=0.2),
        ])
    elif aug == 'E':
        # used in CMCv2
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(
            translate_const=int(image_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
        )
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(crop, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomApply([GaussianBlur(22)], p=0.5),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                   ra_params,
                                   use_cmc=(modal == 'CMC')),
        ])
    else:
        raise NotImplementedError('transform not found: {}'.format(aug))

    # if image_size ==224:
    #     resize = transforms.RandomResizedCrop(image_size, scale=(crop, 1.))
    # else:
    #     resize = transforms.CenterCrop(150)
    if image_size != 224:
        train_transform = transforms.Compose([
            transforms.CenterCrop(150),
            train_transform
        ])

    norm_transforms = transforms.Compose([
        color_transfer,
        transforms.ToTensor(),
        normalize,])

    jigsaw_transform = transforms.Compose([
        transforms.RandomResizedCrop(255, scale=(0.6, 1)),
        transforms.RandomHorizontalFlip(),
        JigsawCrop(),
        StackTransform(transforms.Compose([
            color_transfer,
            transforms.ToTensor(),
            normalize,
        ]))
    ])

    jigstitch_transform = transforms.Compose([
        transforms.RandomResizedCrop(255, scale=(0.6, 1)),
        transforms.RandomHorizontalFlip(),
        ReJigsawCrop(),
        color_transfer,
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, norm_transforms, jigsaw_transform, jigstitch_transform

def get_jigsaw_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    color_transfer = RGB2RGB()
    normalize = transforms.Normalize(mean=mean, std=std)


    jigsaw_transform = transforms.Compose([
        transforms.RandomResizedCrop(255, scale=(0.6, 1)),
        transforms.RandomHorizontalFlip(),
        JigsawCrop(),
        StackTransform(transforms.Compose([
            color_transfer,
            transforms.ToTensor(),
            normalize,
        ]))
    ])
    return jigsaw_transform

def build_contrast_loader(opt, ngpus_per_node):
    """build loaders for contrastive training"""
    data_folder = opt.data_folder
    aug = opt.aug
    modal = opt.modal
    jigsaw_ema = opt.jigsaw_ema
    use_jigsaw = opt.jigsaw
    use_jigstitch = opt.jigsaw_stitch
    use_memory_bank = (opt.mem == 'bank')
    batch_size = int(opt.batch_size / opt.world_size)
    num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    train_transform, norm_transforms, jigsaw_transform, jigstitch_transform = \
        build_transforms(aug, modal, use_memory_bank, opt.image_size)

    # train_dir = os.path.join(data_folder, 'train')
    if opt.dataset_name in ['k16', 'k19']:
        train_pairs, valid_pairs, test_pairs = prepare_colon_class_data_from_json(dataset_name=opt.dataset_name)
    elif opt.dataset_name == 'crc':
        train_pairs, valid_pairs, test_pairs = prepare_colon_class_crc_data(dataset_name=opt.dataset_name)
    elif opt.dataset_name == 'k19+crc':
        train_pairs_k, valid_pairs_k, test_pairs_k = prepare_colon_class_data(dataset_name='k19')
        train_pairs_c, valid_pairs_c, test_pairs_c = prepare_colon_class_crc_data(dataset_name='crc')
        train_pairs = train_pairs_k + train_pairs_c
        valid_pairs = valid_pairs_k + valid_pairs_c
    elif opt.dataset_name == 'k19crcBalance':
        train_pairs, valid_pairs, test_pairs = prepare_colon_class_k19crcBalance_data()
    else:
        f"Don't support dataset_name = {opt.dataset_name}"

    if use_jigstitch:
        if opt.method == "PatchSMoco":
            train_dataset = ImageFolderInstance(
                train_pairs, transform=train_transform, norm_transforms =norm_transforms,
                two_crop=False,
                jigsaw_transform=jigstitch_transform,
                jigsaw_ema=jigsaw_ema
            )
        else:
            train_dataset = ImageFolderInstance(
                train_pairs, transform=train_transform, norm_transforms =norm_transforms,
                two_crop=(not use_memory_bank),
                jigsaw_transform=jigstitch_transform,
                jigsaw_ema=jigsaw_ema
            )
    elif use_jigsaw:
        if opt.jigsaw_aug:
            train_dataset = ImageFolderInstance(
                train_pairs, transform=train_transform, norm_transforms =norm_transforms,
                two_crop=(not use_memory_bank),
                jigsaw_transform=jigsaw_transform,
                jigsaw_ema=jigsaw_ema
            )
        else:
            "k19_224_InfoMin_resnet50_RGB_Jig_False_JigStitch_False_JigEMA_False_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine "
            train_dataset = ImageFolderInstance(
                train_pairs, transform=train_transform, norm_transforms=norm_transforms,
                two_crop=(not use_memory_bank),
                jigsaw_transform=jigsaw_transform,
            )
    else:
        train_dataset = ImageFolderInstance(
            train_pairs, transform=train_transform, norm_transforms =norm_transforms,
            two_crop=(not use_memory_bank)
        )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    #Note drop_last=True
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    print('train images: {}'.format(len(train_dataset)))

    return train_dataset, train_loader, train_sampler



def build_contrast_loader_2(opt, ngpus_per_node):
    """build loaders for contrastive training"""
    data_folder = opt.data_folder
    aug = opt.aug
    modal = opt.modal
    use_jigsaw = opt.jigsaw
    use_memory_bank = (opt.mem == 'bank')
    batch_size = int(opt.batch_size / opt.world_size)
    num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    train_transform, jigsaw_transform = \
        build_transforms(aug, modal, use_memory_bank, opt.image_size)

    # train_dir = os.path.join(data_folder, 'train')
    if opt.dataset_name in ['k16', 'k19']:
        train_pairs, valid_pairs, test_pairs = prepare_colon_class_data(dataset_name=opt.dataset_name)
    elif opt.dataset_name == 'crc':
        train_pairs, valid_pairs, test_pairs = prepare_colon_class_crc_data(dataset_name=opt.dataset_name)
    elif opt.dataset_name == 'k19+crc':
        train_pairs_k, valid_pairs_k, test_pairs_k = prepare_colon_class_data(dataset_name='k19')
        train_pairs_c, valid_pairs_c, test_pairs_c = prepare_colon_class_crc_data(dataset_name='crc')
        train_pairs = train_pairs_k + train_pairs_c
        valid_pairs = valid_pairs_k + valid_pairs_c
    elif opt.dataset_name == 'k19crcBalance':
        train_pairs, valid_pairs, test_pairs = prepare_colon_class_k19crcBalance_data()
    else:
        f"Don't support dataset_name = {opt.dataset_name}"

    if use_jigsaw:
        train_dataset = ImageFolderInstance(
            train_pairs, transform=train_transform,
            two_crop=(not use_memory_bank),
            jigsaw_transform=jigsaw_transform
        )
    else:
        train_dataset = ImageFolderInstance(
            train_pairs, transform=train_transform,
            two_crop=(not use_memory_bank)
        )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    #Note drop_last=True
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    print('train images: {}'.format(len(train_dataset)))

    return train_dataset, train_loader, train_sampler




from histomicstk.preprocessing.augmentation.color_augmentation import rgb_perturb_stain_concentration

class Stain_Augment(object):
    def __call__(self, sample):
        sample = np.array(sample)
        rgbaug = rgb_perturb_stain_concentration(sample, sigma1=1., sigma2=1.)
        return rgbaug

# class RGB2Lab(object):
#     """Convert RGB PIL image to ndarray Lab."""
#
#     def __call__(self, img):
#         img = np.asarray(img, np.uint8)
#         img = color.rgb2lab(img)
#         return img


def build_linear_stain_aug_loader(opt, ngpus_per_node):
    """build loaders for linear evaluation"""

    stain_augment = Stain_Augment()

    # transform
    if opt.modal == 'RGB':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        color_transfer = RGB2RGB()
    else:
        mean = [0.457, -0.082, -0.052]
        std = [0.500, 1.331, 1.333]
        color_transfer = RGB2YDbDr()
    normalize = transforms.Normalize(mean=mean, std=std)

    if opt.aug_linear == 'NULL':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
            transforms.RandomHorizontalFlip(),
            color_transfer,
            transforms.ToTensor(),
            normalize,
        ])
    elif opt.aug_linear == 'RA':
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(
            translate_const=100,
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
        )
        train_transform = transforms.Compose([
            # stain_augment,
            # transforms.ToPILImage(),
            transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
            transforms.RandomHorizontalFlip(),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                   ra_params,
                                   use_cmc=(opt.modal == 'CMC')),
            color_transfer,
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise NotImplementedError('aug not found: {}'.format(opt.aug_linear))

    # dataset
    # data_folder = opt.data_folder
    # train_dir = os.path.join(data_folder, 'train')
    # val_dir = os.path.join(data_folder, 'val')
    if opt.dataset_name in ['k16', 'k19']:
        train_pairs, valid_pairs, test_pairs = prepare_colon_class_data_from_json(dataset_name=opt.dataset_name)
    elif opt.dataset_name == 'crc':
        train_pairs, valid_pairs, test_pairs = prepare_colon_class_crc_data(dataset_name=opt.dataset_name)
    elif opt.dataset_name == 'k19+crc':
        train_pairs_k, valid_pairs_k, test_pairs_k = prepare_colon_class_data(dataset_name='k19')
        train_pairs_c, valid_pairs_c, test_pairs_c = prepare_colon_class_crc_data(dataset_name='crc')
        train_pairs = train_pairs_k + train_pairs_c
        valid_pairs = valid_pairs_k + valid_pairs_c
    elif opt.dataset_name == 'k19crcBalance':
        train_pairs, valid_pairs, test_pairs = prepare_colon_class_k19crcBalance_data()
    else:
        f"Don't support dataset_name = {opt.dataset_name}"

    if opt.infer_only:
        valid_pairs = train_pairs + valid_pairs + test_pairs

    # train_dataset = datasets.ImageFolder(train_pairs, train_transform)
    # val_dataset = datasets.ImageFolder(
    #     valid_pairs,
    #     transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         color_transfer,
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # )


    train_dataset = LinearInstanceColorAug(
        train_pairs, transform=train_transform)
    val_dataset = LinearInstance(
        valid_pairs,
        transform=transforms.Compose([
            transforms.Resize(int(opt.image_size*256/224)),
            transforms.CenterCrop(opt.image_size),
            color_transfer,
            transforms.ToTensor(),
            normalize,
        ])
    )


    # loader
    batch_size = int(opt.batch_size / opt.world_size)
    num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler)
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=128, shuffle=False,
    #     num_workers=32, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size_infer, shuffle=False,
        num_workers=32, pin_memory=True)

    print('train images: {}'.format(len(train_dataset)))
    print('test images: {}'.format(len(val_dataset)))

    return train_dataset, train_loader, val_loader, train_sampler



def build_linear_loader(opt, ngpus_per_node):
    """build loaders for linear evaluation"""
    # transform
    if opt.modal == 'RGB':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        color_transfer = RGB2RGB()
    else:
        mean = [0.457, -0.082, -0.052]
        std = [0.500, 1.331, 1.333]
        color_transfer = RGB2YDbDr()
    normalize = transforms.Normalize(mean=mean, std=std)

    if opt.aug_linear == 'NULL':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
            transforms.RandomHorizontalFlip(),
            color_transfer,
            transforms.ToTensor(),
            normalize,
        ])
    elif opt.aug_linear == 'RA':
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(
            translate_const=100,
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
        )
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
            transforms.RandomHorizontalFlip(),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                   ra_params,
                                   use_cmc=(opt.modal == 'CMC')),
            color_transfer,
            transforms.ToTensor(),
            normalize,
        ])
    elif opt.aug_linear == 'SRA':
        train_transform = get_supervised_train_augmentation()
        # train_transform = transforms.Compose([
        # transforms.Resize(224),
        # transforms.ToTensor(),
        # # transforms.Normalize(mean=[0.742, 0.616, 0.731], std=[0.211, 0.274, 0.202]),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    else:
        raise NotImplementedError('aug not found: {}'.format(opt.aug_linear))

    if opt.aug_linear == 'SRA':
        val_transform = get_supervised_val_augmentation()
    #     val_transform = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.ToTensor(),
    #     # transforms.Normalize(mean=[0.742, 0.616, 0.731], std=[0.211, 0.274, 0.202]),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    else:
        val_transform = transforms.Compose([
            transforms.Resize(int(opt.image_size * 256 / 224)),
            transforms.CenterCrop(opt.image_size),
            color_transfer,
            transforms.ToTensor(),
            normalize,
        ])

    # dataset
    # data_folder = opt.data_folder
    # train_dir = os.path.join(data_folder, 'train')
    # val_dir = os.path.join(data_folder, 'val')
    if opt.dataset_name in ['k16', 'k19']:
        train_pairs, valid_pairs, test_pairs = prepare_colon_class_data_from_json(dataset_name=opt.dataset_name)
    elif opt.dataset_name == 'crc':
        train_pairs, valid_pairs, test_pairs = prepare_colon_class_crc_data(dataset_name=opt.dataset_name)
    elif opt.dataset_name == 'k19+crc':
        train_pairs_k, valid_pairs_k, test_pairs_k = prepare_colon_class_data(dataset_name='k19')
        train_pairs_c, valid_pairs_c, test_pairs_c = prepare_colon_class_crc_data(dataset_name='crc')
        train_pairs = train_pairs_k + train_pairs_c
        valid_pairs = valid_pairs_k + valid_pairs_c
    elif opt.dataset_name == 'k19crcBalance':
        train_pairs, valid_pairs, test_pairs = prepare_colon_class_k19crcBalance_data()
    else:
        f"Don't support dataset_name = {opt.dataset_name}"

    if opt.infer_only:
        valid_pairs = train_pairs + valid_pairs + test_pairs

    # train_dataset = datasets.ImageFolder(train_pairs, train_transform)
    # val_dataset = datasets.ImageFolder(
    #     valid_pairs,
    #     transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         color_transfer,
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # )

    if opt.jigsaw:
        jigsaw_transform = get_jigsaw_transform()
        train_dataset = LinearInstance(
            train_pairs, transform=train_transform,
            jigsaw_transform=jigsaw_transform
        )
        val_dataset = LinearInstance(
            valid_pairs,
            transform=val_transform,
            jigsaw_transform=jigsaw_transform
        )
    else:
        train_dataset = LinearInstance(
            train_pairs, transform=train_transform)
        val_dataset = LinearInstance(
            valid_pairs,
            transform=val_transform
        )


    # loader
    batch_size = int(opt.batch_size / opt.world_size)
    num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler)
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=128, shuffle=False,
    #     num_workers=32, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size_infer, shuffle=False,
        num_workers=32, pin_memory=True)

    print('train images: {}'.format(len(train_dataset)))
    print('test images: {}'.format(len(val_dataset)))

    return train_dataset, train_loader, val_loader, train_sampler


def build_test_loader(opt, ngpus_per_node):
    """build loaders for linear evaluation"""
    # transform
    if opt.modal == 'RGB':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        color_transfer = RGB2RGB()
    else:
        mean = [0.457, -0.082, -0.052]
        std = [0.500, 1.331, 1.333]
        color_transfer = RGB2YDbDr()
    normalize = transforms.Normalize(mean=mean, std=std)

    if opt.aug_linear == 'NULL':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
            transforms.RandomHorizontalFlip(),
            color_transfer,
            transforms.ToTensor(),
            normalize,
        ])
    elif opt.aug_linear == 'RA':
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(
            translate_const=100,
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
        )
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
            transforms.RandomHorizontalFlip(),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                   ra_params,
                                   use_cmc=(opt.modal == 'CMC')),
            color_transfer,
            transforms.ToTensor(),
            normalize,
        ])
    elif opt.aug_linear == 'SRA':
        train_transform = get_supervised_train_augmentation()
        # train_transform = transforms.Compose([
        # transforms.Resize(224),
        # transforms.ToTensor(),
        # # transforms.Normalize(mean=[0.742, 0.616, 0.731], std=[0.211, 0.274, 0.202]),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    else:
        raise NotImplementedError('aug not found: {}'.format(opt.aug_linear))
    if opt.image_size != 224:
        train_transform = transforms.Compose([
            transforms.CenterCrop(opt.image_size),
            train_transform
        ])

    # dataset
    # data_folder = opt.data_folder
    # train_dir = os.path.join(data_folder, 'train')
    # val_dir = os.path.join(data_folder, 'val')
    if opt.dataset_name in ['k16', 'k19']:
        train_pairs, valid_pairs, test_pairs = prepare_colon_class_data(dataset_name=opt.dataset_name)
    elif opt.dataset_name == 'crc':
        train_pairs, valid_pairs, test_pairs = prepare_colon_class_crc_data(dataset_name=opt.dataset_name)
    elif opt.dataset_name == 'crcval':
        train_pairs, valid_pairs, test_pairs = prepare_colon_class_crc_data(dataset_name=opt.dataset_name)
        valid_pairs = test_pairs
    elif opt.dataset_name == 'k19val':
        valid_pairs = prepare_colon_class_k19val_data()
        train_pairs = []
    elif opt.dataset_name == 'k16val':
        train_pairs, valid_pairs, test_pairs = prepare_colon_class_data(dataset_name='k16')
        valid_pairs = valid_pairs + test_pairs
    elif opt.dataset_name == 'k19+crc':
        train_pairs_k, valid_pairs_k, test_pairs_k = prepare_colon_class_data(dataset_name='k19')
        train_pairs_c, valid_pairs_c, test_pairs_c = prepare_colon_class_crc_data(dataset_name='crc')
        train_pairs = train_pairs_k + train_pairs_c
        valid_pairs = valid_pairs_k + valid_pairs_c
    elif opt.dataset_name == 'k19crcBalance':
        train_pairs, valid_pairs, test_pairs = prepare_colon_class_k19crcBalance_data()
    else:
        f"Don't support dataset_name = {opt.dataset_name}"

    # if opt.infer_only & ('val' not in opt.dataset_name):
    #     valid_pairs = train_pairs + valid_pairs + test_pairs

    if opt.infer_only & ('val' not in opt.dataset_name):
        valid_pairs = train_pairs + valid_pairs + test_pairs

    if opt.colornorm:
        target_color = opt.model_name.split('_model_')[-1][:3]
        print(target_color)

        if target_color == 'crc':
            target_color_path = '/data1/trinh/data/raw_data/Domain_Invariance/colon_class/CRCTP/Training/Tumor/9.png'
        elif target_color == 'k19':
            target_color_path = '/data1/trinh/data/raw_data/Domain_Invariance/colon_class/NCT-CRC-HE-100K/TUM/TUM-YYYRHCTW.tif'
        elif target_color == 'k16':
            target_color_path = '/data1/trinh/data/raw_data/Domain_Invariance/colon_class/Kather_texture_2016_image_tiles_5000/01_TUMOR/A8D0_CRC-Prim-HE-10_002c.tif_Row_1_Col_451.tif'
        else:
            raise(" Invalid target color")


        target_image = cv2.imread(target_color_path)
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        # norm = VahadaneNormalizer()
        norm = MacenkoNormalizer()
        norm.fit(target_image)


    train_dataset = LinearInstance(
        train_pairs, transform=train_transform)
    if opt.colornorm:
        val_dataset = LinearInstanceNorm(
            valid_pairs,
            transform=transforms.Compose([
                transforms.Resize(int(opt.image_size * 256 / 224)),
                transforms.CenterCrop(opt.image_size),
                color_transfer,
                transforms.ToTensor(),
                normalize,
            ]),
            colornorm=norm
        )
    else:
        val_dataset = LinearInstance(
            valid_pairs,
            transform=transforms.Compose([
                transforms.Resize(int(opt.image_size * 256 / 224)),
                transforms.CenterCrop(opt.image_size),
                color_transfer,
                transforms.ToTensor(),
                normalize,
            ]))

    # loader
    batch_size = int(opt.batch_size / opt.world_size)
    num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler)
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=128, shuffle=False,
    #     num_workers=32, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size_infer, shuffle=False,
        num_workers=32, pin_memory=True)

    print('train images: {}'.format(len(train_dataset)))
    print('test images: {}'.format(len(val_dataset)))

    return train_loader, val_loader, train_sampler
