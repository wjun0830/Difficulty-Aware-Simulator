import os
import argparse
import csv
import pandas as pd
import importlib
import torch
from torch.autograd import Variable
import torch.nn as nn
import sys
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from models.models import classifier32, classifier32ABN, Generator, Discriminator
from datasets.osr_dataloader import MNIST_OSR, CIFAR10_OSR, CIFAR100_OSR, SVHN_OSR, Tiny_ImageNet_OSR
from torch.utils.tensorboard import SummaryWriter
from split import splits_AUROC, splits_F1
from sklearn.metrics import f1_score
import os.path as osp
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, roc_auc_score
parser = argparse.ArgumentParser("Training")
parser.add_argument('--dataset', type=str, default='mnist', help="mnist | svhn | cifar10 | cifar100 | tiny_imagenet")
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--out-num', type=int, default=50, help='For CIFAR100')
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
parser.add_argument('--max-epoch', type=int, default=200)
parser.add_argument('--img_size', type=int, default=32)
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--fake_ratio', type=float, default=0.1)
parser.add_argument('--pull_ratio', type=float, default=1.0)
parser.add_argument('--smoothing', type=float, default=0.5)
parser.add_argument('--smoothing2', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save-dir', type=str, default='../log')
parser.add_argument('--split', type=str, default='F1', help="F1 | AUROC")
class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()

def train(net1, net2, criterion, optimizer1, optimizer2, trainloader, **options):
    lsr_criterion = SmoothCrossEntropy(options['smoothing'])
    lsr_criterion2 = SmoothCrossEntropy(options['smoothing2'])
    l1_loss = nn.L1Loss()
    torch.cuda.empty_cache()
    for batch_idx, (data, labels) in enumerate(trainloader):
        data, labels = data.cuda(), labels.cuda()
        bsz = labels.size(0)
        with torch.set_grad_enabled(True):
            net1.train()
            net2.eval()
            optimizer1.zero_grad()
            feat11, feat12, feat1, out1 = net1(data, return_feature=True, layers=[1,2])
            feat21, feat22, feat2, out2 = net2(data, return_feature=True, layers=[1,2])
            loss1 = criterion(out1, labels)
            pullloss1 = l1_loss(feat11.reshape(bsz, -1), feat21.reshape(bsz, -1).detach())#,options['margin'])
            pullloss2 = l1_loss(feat12.reshape(bsz, -1), feat22.reshape(bsz, -1).detach())#,options['margin'])
            pullloss = (pullloss1 + pullloss2) / 2
            loss1 = loss1 + options['pull_ratio'] * pullloss
            loss1.backward()
            optimizer1.step()

            net1.eval()
            net2.train()
            optimizer2.zero_grad()
            feat11, feat12, feat1, out1 = net1(data, return_feature=True, layers=[1,2])
            feat21, feat22, feat2, out2 = net2(data, return_feature=True, layers=[1,2])
            out21 = net2(feat11.detach(), input_layers=1)
            out22 = net2(feat12.detach(), input_layers=2)
            out20 = net2(feat1.clone().detach(), onlyfc=True)
            klu0 = lsr_criterion2(out20, labels)
            klu1 = lsr_criterion(out21, labels)
            klu2 = lsr_criterion(out22, labels)
            klu = (klu0 + klu1 + klu2) / 3
            loss2 = criterion(out2, labels)
            loss2 = loss2 + klu * options['fake_ratio']
            loss2.backward()
            optimizer2.step()

def train_gan(net, netD, netG, criterion, criterionD, optimizer2, optimizerD, optimizerG, trainloader, **options):
    net.train()
    netD.train()
    netG.train()
    torch.cuda.empty_cache()
    lsr_criterion = SmoothCrossEntropy(options['smoothing2'])
    real_label, fake_label = 1, 0
    for batch_idx, (data, labels) in enumerate(trainloader):
        gan_target = torch.FloatTensor(labels.size()).fill_(0)
        noise = torch.FloatTensor(data.size(0), 100, 1, 1).normal_(0, 1).cuda()
        data, labels, gan_target, noise = data.cuda(non_blocking=True), labels.cuda(non_blocking=True), gan_target.cuda(), noise.cuda()

        fake = netG(noise)
        gan_target.fill_(real_label)
        targetv = Variable(gan_target)
        optimizerD.zero_grad()
        output = netD(data)
        errD_real = criterionD(output, targetv)
        errD_real.backward()

        targetv = Variable(gan_target.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterionD(output, targetv)
        errD_fake.backward()
        optimizerD.step()

        optimizerG.zero_grad()
        targetv = Variable(gan_target.fill_(real_label))
        output = netD(fake)
        errG = criterionD(output, targetv)
        x = net(fake, bn_label=1 * torch.ones(data.shape[0], dtype=torch.long).cuda())
        errG_F = lsr_criterion(x, labels)
        generator_loss = errG + options['beta'] * errG_F * -1.
        generator_loss.backward()
        optimizerG.step()

        optimizer2.zero_grad()
        x = net(data,bn_label=0 * torch.ones(data.shape[0], dtype=torch.long).cuda())
        loss = criterion(x, labels)
        noise = torch.FloatTensor(data.size(0), 100, 1, 1).normal_(0, 1).cuda()
        noise = noise.cuda()
        noise = Variable(noise)
        fake = netG(noise)
        x= net(fake, bn_label=1 * torch.ones(data.shape[0], dtype=torch.long).cuda())
        F_loss_fake = lsr_criterion(x, labels)
        total_loss = loss + options['beta'] * F_loss_fake
        total_loss.backward()
        optimizer2.step()


def evaluation(net2, testloader, outloader, **options):
    net2.eval()
    correct, total, n = 0, 0, 0
    torch.cuda.empty_cache()
    pred_close, pred_open, labels_close, labels_open = [], [], [], []
    open_labels = torch.zeros(50000)
    probs = torch.zeros(50000)
    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.cuda(), labels.cuda()
            bsz = labels.size(0)
            with torch.set_grad_enabled(False):
                _, logits = net2(data, return_feature=True)
                logits = torch.softmax(logits / options['temp'], dim=1)
                confidence = logits.data.max(1)[0]
                for b in range(bsz):
                    probs[n] = confidence[b]
                    open_labels[n] = 1
                    n += 1
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()
                pred_close.append(logits.data.cpu().numpy())
                labels_close.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels) in enumerate(outloader):
            data, labels = data.cuda(), labels.cuda()
            bsz = labels.size(0)
            oodlabel = torch.zeros_like(labels) - 1
            with torch.set_grad_enabled(False):
                _, logits = net2(data, return_feature=True)
                logits = torch.softmax(logits / options['temp'], dim=1)
                confidence = logits.data.max(1)[0]
                for b in range(bsz):
                    probs[n] = confidence[b]
                    open_labels[n] = 0
                    n += 1
                pred_open.append(logits.data.cpu().numpy())
                labels_open.append(oodlabel.data.cpu().numpy())
    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    pred_close = np.concatenate(pred_close, 0)
    pred_open = np.concatenate(pred_open, 0)
    labels_close = np.concatenate(labels_close, 0)
    labels_open = np.concatenate(labels_open, 0)
    # F1 score Evaluation
    x1, x2 = np.max(pred_close, axis=1), np.max(pred_open, axis=1)
    pred1, pred2 = np.argmax(pred_close, axis=1), np.argmax(pred_open, axis=1)
    total_pred_label = np.concatenate([pred1, pred2], axis=0)
    total_label = np.concatenate([labels_close, labels_open], axis=0)
    total_pred = np.concatenate([x1, x2], axis=0)
    thr = options['smoothing'] / options['num_classes'] + (1 - options['smoothing'])
    open_pred = (total_pred > thr - 0.05).astype(np.float32)
    f = f1_score(total_label, ((total_pred_label + 1) * open_pred) - 1, average='macro')

    # AUROC score Evaluation
    open_labels = open_labels[:n].cpu().numpy()
    prob = probs[:n].reshape(-1, 1)
    auc = roc_auc_score(open_labels, prob)

    return acc, auc, f

def main(options):
    torch.manual_seed(options['seed'])
    use_gpu = torch.cuda.is_available()
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(options['seed'])


    # Dataset
    print("{} Preparation".format(options['dataset']))
    if 'mnist' in options['dataset']:
        Data = MNIST_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'],
                         img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'cifar10' == options['dataset']:
        Data = CIFAR10_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'],
                           img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'svhn' in options['dataset']:
        Data = SVHN_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'],
                        img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'cifar100' in options['dataset']:
        Data = CIFAR10_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'],
                           img_size=options['img_size'])
        trainloader, testloader = Data.train_loader, Data.test_loader
        out_Data = CIFAR100_OSR(known=options['unknown'], dataroot=options['dataroot'],
                                batch_size=options['batch_size'], img_size=options['img_size'])
        outloader = out_Data.test_loader
    else:
        Data = Tiny_ImageNet_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'],
                                 img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    options['num_classes'] = Data.num_classes
    net1 = classifier32ABN(num_classes=options['num_classes'])
    net2 = classifier32ABN(num_classes=options['num_classes'])
    netG = Generator()
    netD = Discriminator()
    criterionD = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    net1 = nn.DataParallel(net1).cuda()
    net2 = nn.DataParallel(net2).cuda()
    criterion = criterion.cuda()
    netG = nn.DataParallel(netG).cuda()
    netD = nn.DataParallel(netD).cuda()


    if options['optimizer'] == 'adam':
        options['lr'] = 0.001
        optimizer1 = torch.optim.Adam(net1.parameters(), lr=options['lr'])
        optimizer2 = torch.optim.Adam(net2.parameters(), lr=options['lr'])
        scheduler1 = lr_scheduler.MultiStepLR(optimizer1, milestones=[options['max_epoch'] // 2])
        scheduler2 = lr_scheduler.MultiStepLR(optimizer2, milestones=[options['max_epoch'] // 2])
    else:
        options['lr'] = 0.1
        optimizer1 = torch.optim.SGD(net1.parameters(), lr=options['lr'], momentum=0.9, weight_decay=1e-4)
        optimizer2 = torch.optim.SGD(net2.parameters(), lr=options['lr'], momentum=0.9, weight_decay=1e-4)
        scheduler1 = lr_scheduler.MultiStepLR(optimizer1, milestones=[60, 120, 180, 240])
        scheduler2 = lr_scheduler.MultiStepLR(optimizer2, milestones=[60, 120, 180, 240])


    optimizerD = torch.optim.Adam(netD.parameters(), lr=options['gan_lr'], betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=options['gan_lr'], betas=(0.5, 0.999))


    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch + 1, options['max_epoch']))
        train_gan(net2, netD, netG, criterion, criterionD,
            optimizer2, optimizerD, optimizerG,
            trainloader, **options)
        train(net1, net2, criterion, optimizer1=optimizer1, optimizer2=optimizer2, trainloader=trainloader, **options)

        scheduler1.step()
        scheduler2.step()

        if (epoch+1) % 25 == 0:

            save_checkpoint(epoch, {
                'epoch': epoch,
                'net1_state_dict': net1.state_dict(),
                'net2_state_dict': net2.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'netG_state_dict': netG.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'optimizer2': optimizer2.state_dict(),
                'optimizerD': optimizerD.state_dict(),
                'optimizerG': optimizerG.state_dict()
            }, options['batch_size'], options['item'], True)

    acc, auc, f1 = evaluation(net2, testloader, outloader, **options)
    return acc, auc, f1

def save_checkpoint(epoch, state, bsz, item, is_best=True, filename='checkpoint.pth.tar'):
    directory = "runs/%s/%s/" % (str(options['split']), options['dataset'])
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + str(item) + '_' + str(epoch) + '_' + str(bsz) + '_' + filename
    torch.save(state, filename)



if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    if options['split'] == 'AUROC':
        splits = splits_AUROC
    elif options['split'] == 'F1':
        splits = splits_F1
    else:
        raise NotImplementedError()
    options['dataroot'] = os.path.join(options['dataroot'], options['dataset'])

    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./logs/AUROC'):
        os.makedirs('./logs/AUROC')
    if not os.path.exists('./logs/F1'):
        os.makedirs('./logs/F1')
    if not os.path.exists('./logs/AUROC/' + options['dataset']):
        os.makedirs('./logs/AUROC/' + options['dataset'])
    if not os.path.exists('./logs/F1/' + options['dataset']):
        os.makedirs('./logs/F1/' + options['dataset'])
    for i in range(len(splits[options['dataset']])):
        options['item'] = i
        options['writer'] = SummaryWriter(f"results/DIAS_{i}")
        known = splits[options['dataset']][len(splits[options['dataset']]) - i - 1]
        if options['dataset'] == 'cifar100':
            unknown = splits[options['dataset'] + '-' + str(options['out_num'])][
                len(splits[options['dataset']]) - i - 1]
        elif options['dataset'] == 'tiny_imagenet':
            options['lr'] = 0.001
            unknown = list(set(list(range(0, 200))) - set(known))
        else:
            unknown = list(set(list(range(0, 10))) - set(known))
        options.update({'known': known, 'unknown': unknown})
        if options['optimizer'] == 'adam':
            options['lr'] = 0.001
        else:
            options['lr'] = 0.1

        acc, auc, f1 = main(options)
        if options['dataset'] == 'cifar100':
            stats_log = open('./logs/' + options['split'] + '/' + options['dataset'] + '/DIAS_%d' % (i) + str(options[
                'out_num']) + '.txt', 'w')
        else:
            stats_log = open('./logs/' + options['split'] + '/' + options['dataset'] + '/DIAS_%d' % (i), '.txt', 'w')
        stats_log.write("ACC AUC F1: [%.3f], [%.3f], [%.3f]\n" % (acc, auc, f1))
        stats_log.write("-------------------------")
        stats_log.flush()
        stats_log.close()