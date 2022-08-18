import torch
import torch.nn as nn
from .resnet import model_dict


class Classifier(nn.Module):
    """RGB model with a single linear/mlp projection head"""

    def __init__(self, name='resnet50',num_class=1000, pretrained=False):
        super(Classifier, self).__init__()

        name, width = self._parse_width(name)
        dim_in = int(2048 * width)
        self.width = width

        self.encoder = model_dict[name](width=width, pretrained=pretrained)
        self.classifier = nn.Linear(dim_in, num_class)

    @staticmethod
    def _parse_width(name):
        if name.endswith('x4'):
            return name[:-2], 4
        elif name.endswith('x2'):
            return name[:-2], 2
        else:
            return name, 1

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.classifier(feat)
        return feat


def build_finetune(opt):
    if opt.finetune:
        pretrained = True
    else:
        pretrained = False

    return Classifier(num_class=opt.n_class, pretrained=pretrained)


# if __name__ == '__main__':
#     model = Classifier(num_class=4)
#     # model = resnet50(width=0.5, pretrained=True)
#     data = torch.randn(2, 3, 224, 224)
#     out = model(data)
#     print(out.shape)


# def build_linear(opt):
#     n_class = opt.n_class
#     arch = opt.arch
#     if arch.endswith('x4'):
#         n_feat = 2048 * 4
#     elif arch.endswith('x2'):
#         n_feat = 2048 * 2
#     else:
#         n_feat = 2048
#
#     classifier = nn.Linear(n_feat, n_class)
#     return classifier