import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from resnet_wider import resnet50x1, resnet50x2, resnet50x4
from advertorch.attacks import LinfPGDAttack, L2PGDAttack
import numpy as np

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', default='resnet50-1x')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--attack', default='', type=str,
                    help='attack type')

best_acc1 = 0


def main():
    args = parser.parse_args()

    # create model
    if args.arch == 'resnet50-1x':
        model = resnet50x1()
        sd = '/content/gdrive/MyDrive/model_checkpoints/resnet50-1x.pth'
    elif args.arch == 'resnet50-2x':
        model = resnet50x2()
        sd = 'resnet50-2x.pth'
    elif args.arch == 'resnet50-4x':
        model = resnet50x4()
        sd = 'resnet50-4x.pth'
    else:
        raise NotImplementedError
    sd = torch.load(sd, map_location='cuda')
    model.load_state_dict(sd['state_dict'])

    model = torch.nn.DataParallel(model).to('cuda')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    cudnn.benchmark = True

    # Data loading code
    valdir = os.path.join(args.data, 'val')

    # NOTICE, the original model do not have normalization
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    validate(val_loader, model, criterion, args)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if (args.attack=='l2_3'):
      adversary = L2PGDAttack(
      model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=3,
      nb_iter=20, eps_iter=0.375, rand_init=True, clip_min=0, clip_max=1,
      targeted=False)
    if (args.attack=='l2_0.15'):
      adversary = L2PGDAttack(
      model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.15,
      nb_iter=20, eps_iter=0.01875, rand_init=True, clip_min=0, clip_max=1,
      targeted=False)
    if (args.attack=='linf1_1020'):
      adversary = LinfPGDAttack(
      model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=1/1020,
      nb_iter=20, eps_iter=0.00012255, rand_init=True, clip_min=0, clip_max=1,
      targeted=False)
    if args.attack=='linf4_255':
        adversary = LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=19.0316/255,
        nb_iter=20, eps_iter=47.579/5100, rand_init=True, clip_min=-2.1179, clip_max=2.6400,
        targeted=False)
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            with torch.enable_grad():
              adv_untargeted = adversary.perturb(images.to('cuda'), target.to('cuda'))
            target = target.to('cuda')

            # compute output
            output = model(adv_untargeted)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        accuracy_array=[]
        accuracy_array.append(top1.avg.to('cpu'))
        accuracy_array.append(top5.avg.to('cpu'))
        np.save(f'/content/gdrive/MyDrive/model_adv_loss/{args.attack}/simclr_accuracy.npy', accuracy_array)
        

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
