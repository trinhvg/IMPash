from __future__ import print_function
import torch.utils.data as data
import torch
from PIL import Image
import cv2


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ImageFolderInstance(data.Dataset):
    """Folder datasets which returns the index of the image (for memory_bank)
    """
    def __init__(self, pair_list, transform=None, norm_transforms=None,target_transform=None,
                 two_crop=False, jigsaw_transform=None, jigsaw_ema=False):
        self.pair_list = pair_list
        self.transform = transform
        self.norm_transforms = norm_transforms
        self.target_transform = target_transform
        self.two_crop = two_crop
        self.jigsaw_ema = jigsaw_ema
        self.jigsaw_transform = jigsaw_transform
        self.use_jigsaw = (jigsaw_transform is not None)
        self.num = self.__len__()

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """
        path, target = self.pair_list[index]

        image = pil_loader(path)


        # # image
        if self.transform is not None:
            img = self.transform(image)
            img = self.norm_transforms(img)
            if self.two_crop:
                img2_temp = self.transform(image)
                img2 = self.norm_transforms(img2_temp)
                img = torch.cat([img, img2], dim=0)
        else:
            img = image

        # # jigsaw
        if self.use_jigsaw:
            jigsaw_image = self.jigsaw_transform(image)
            if self.jigsaw_ema:
                jigsaw_image2 = self.jigsaw_transform(image)
                jigsaw_image = torch.cat([jigsaw_image, jigsaw_image2], dim=0)


        if self.use_jigsaw:
            return img, index, jigsaw_image
        else:
            return img, index


    def __len__(self):
        return len(self.pair_list)


class LinearInstance(data.Dataset):
    """Folder datasets which returns the index of the image (for memory_bank)
    """
    def __init__(self, pair_list, transform=None, target_transform=None,
                 two_crop=False, jigsaw_transform=None):
        self.pair_list = pair_list
        self.transform = transform
        self.target_transform = target_transform
        self.two_crop = two_crop
        self.jigsaw_transform = jigsaw_transform
        self.use_jigsaw = (jigsaw_transform is not None)
        self.num = self.__len__()

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """

        path, target = self.pair_list[index]
        image = pil_loader(path)

        # # image
        if self.transform is not None:
            img = self.transform(image)
            if self.two_crop:
                img2 = self.transform(image)
                img = torch.cat([img, img2], dim=0)
        else:
            img = image

        # # jigsaw
        if self.use_jigsaw:
            jigsaw_image = self.jigsaw_transform(image)

        if self.use_jigsaw:
            return img, target, jigsaw_image
        else:
            return img, target


    def __len__(self):
        return len(self.pair_list)


from histomicstk.preprocessing.augmentation.color_augmentation import rgb_perturb_stain_concentration

class LinearInstanceColorAug(data.Dataset):
    """Folder datasets which returns the index of the image (for memory_bank)
    """
    def __init__(self, pair_list, transform=None, target_transform=None,
                 two_crop=False, jigsaw_transform=None):
        self.pair_list = pair_list
        self.transform = transform
        self.target_transform = target_transform
        self.two_crop = two_crop
        self.jigsaw_transform = jigsaw_transform
        self.use_jigsaw = (jigsaw_transform is not None)
        self.num = self.__len__()

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """


        path, target = self.pair_list[index]

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if target != 1:
            try:
                image = rgb_perturb_stain_concentration(image, sigma1=1., sigma2=1.)
            except:
                pass
        image = Image.fromarray(image)

        # # image
        if self.transform is not None:
            img = self.transform(image)
            if self.two_crop:
                img2 = self.transform(image)
                img = torch.cat([img, img2], dim=0)
        else:
            img = image

        # # jigsaw
        if self.use_jigsaw:
            jigsaw_image = self.jigsaw_transform(image)

        if self.use_jigsaw:
            return img, target, jigsaw_image
        else:
            return img, target


    def __len__(self):
        return len(self.pair_list)


class LinearInstanceNorm(data.Dataset):
    """Folder datasets which returns the index of the image (for memory_bank)
    """
    def __init__(self, pair_list, colornorm, transform=None, target_transform=None,
                 two_crop=False, jigsaw_transform=None):
        self.pair_list = pair_list
        self.transform = transform
        self.target_transform = target_transform
        self.two_crop = two_crop
        self.jigsaw_transform = jigsaw_transform
        self.use_jigsaw = (jigsaw_transform is not None)
        self.norm = colornorm
        self.num = self.__len__()

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """
        path, target = self.pair_list[index]

        #cv2 load for color norm then we need to convert back to PIL

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if target != 1:
            image = self.norm.transform(image)


        # Convert back to PIL
        image = Image.fromarray(image)


        # # image
        if self.transform is not None:
            img = self.transform(image)
            if self.two_crop:
                img2 = self.transform(image)
                img = torch.cat([img, img2], dim=0)
        else:
            img = image

        # # jigsaw
        if self.use_jigsaw:
            jigsaw_image = self.jigsaw_transform(image)

        if self.use_jigsaw:
            return img, target, jigsaw_image
        else:
            return img, target


    def __len__(self):
        return len(self.pair_list)


