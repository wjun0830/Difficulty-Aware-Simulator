import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import numpy as np
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class classifier32(nn.Module):
    def __init__(self, num_classes=10):
        super(self.__class__, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)
        self.bn10 = nn.BatchNorm2d(128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes, bias=False)
        self.fc1 = nn.Linear(128, num_classes, bias=False)
        self.fc2 = nn.Linear(128, num_classes, bias=False)

        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_feature=False, layers=None, input_layers=None, onlyfc=False):
        if onlyfc:
            return self.fc(x)
        if input_layers is None:
            x = self.dr1(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = nn.LeakyReLU(0.2)(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = nn.LeakyReLU(0.2)(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = nn.LeakyReLU(0.2)(x)
            if layers is not None:
                if 1 in layers:
                    x1 = x.clone()

        if input_layers is None or input_layers == 1:
            x = self.dr2(x)
            x = self.conv4(x)
            x = self.bn4(x)
            x = nn.LeakyReLU(0.2)(x)
            x = self.conv5(x)
            x = self.bn5(x)
            x = nn.LeakyReLU(0.2)(x)
            x = self.conv6(x)
            x = self.bn6(x)
            x = nn.LeakyReLU(0.2)(x)
            if layers is not None:
                if 2 in layers:
                    x2 = x.clone()


        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.fc(x)
        if layers is None:
            if return_feature:
                return x, y
        if return_feature:
            return x1, x2, x, y
        else:
            if layers is None:
                return y
            if 1 and 2 in layers:
                return y, x1, x2
            if 1 in layers:
                return y, x1
            if 2 in layers:
                return y, x2
            return y


def weights_init_ABN(m):
    classname = m.__class__.__name__
    # TODO: what about fully-connected layers?
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('MultiBatchNorm') != -1:
        m.bns[0].weight.data.normal_(1.0, 0.02)
        m.bns[0].bias.data.fill_(0)
        m.bns[1].weight.data.normal_(1.0, 0.02)
        m.bns[1].bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class classifier32ABN(nn.Module):
    def __init__(self, num_classes=10, num_ABN=2):
        super(self.__class__, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.bn1 = MultiBatchNorm(64, num_ABN)
        self.bn2 = MultiBatchNorm(64, num_ABN)
        self.bn3 = MultiBatchNorm(128, num_ABN)

        self.bn4 = MultiBatchNorm(128, num_ABN)
        self.bn5 = MultiBatchNorm(128, num_ABN)
        self.bn6 = MultiBatchNorm(128, num_ABN)

        self.bn7 = MultiBatchNorm(128, num_ABN)
        self.bn8 = MultiBatchNorm(128, num_ABN)
        self.bn9 = MultiBatchNorm(128, num_ABN)
        self.bn10 = MultiBatchNorm(128, num_ABN)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes, bias=False)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.apply(weights_init_ABN)
        self.cuda()

    def forward(self, x, return_feature=False, layers=None, input_layers=None, onlyfc=False, bn_label=None):
        if bn_label is None:
            bn_label = 0 * torch.ones(x.shape[0], dtype=torch.long).cuda()
        if onlyfc:
            if return_feature:
                return x, self.fc(x)
            return self.fc(x)
        if input_layers is None:
            x = self.dr1(x)
            x = self.conv1(x)
            x, _ = self.bn1(x, bn_label)
            x = nn.LeakyReLU(0.2)(x)
            x = self.conv2(x)
            x, _ = self.bn2(x, bn_label)
            x = nn.LeakyReLU(0.2)(x)
            x = self.conv3(x)
            x, _ = self.bn3(x, bn_label)
            x = nn.LeakyReLU(0.2)(x)
            if layers is not None:
                if 1 in layers:
                    x1 = x.clone()

        if input_layers is None or input_layers == 1:
            x = self.dr2(x)
            x = self.conv4(x)
            x, _ = self.bn4(x, bn_label)
            x = nn.LeakyReLU(0.2)(x)
            x = self.conv5(x)
            x, _ = self.bn5(x, bn_label)
            x = nn.LeakyReLU(0.2)(x)
            x = self.conv6(x)
            x, _ = self.bn6(x, bn_label)
            x = nn.LeakyReLU(0.2)(x)
            if layers is not None:
                if 2 in layers:
                    x2 = x.clone()


        x = self.dr3(x)
        x = self.conv7(x)
        x, _ = self.bn7(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x, _ = self.bn8(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x, _ = self.bn9(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.fc(x)
        if layers is None:
            if return_feature:
                return x, y
        if return_feature:
            return x1, x2, x, y
        else:
            if layers is None:
                return y
            if 1 and 2 in layers:
                return y, x1, x2
            if 1 in layers:
                return y, x1
            if 2 in layers:
                return y, x2
            return y




class _MultiBatchNorm(nn.Module):
    _version = 2

    def __init__(self, num_features, num_classes, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_MultiBatchNorm, self).__init__()
        #         self.bns = nn.ModuleList([nn.modules.batchnorm._BatchNorm(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, domain_label):
        self._check_input_dim(x)
        bn = self.bns[domain_label[0]]
        return bn(x), domain_label


class MultiBatchNorm(_MultiBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class Discriminator(nn.Module):
    def __init__(self, ngpu=1, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input size. (nc) x 32 x 32
            nn.Conv2d(nc, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 16, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(ndf * 16, 1),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.classifier(output).flatten()

        return output


class Generator(nn.Module):
    def __init__(self, ngpu=1, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 32 x 32
        )
        self.apply(weights_init)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

