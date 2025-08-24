import argparse
import random
from dataloader.dataset import MedicalDataSets,IristDataSets,IristDataSets_SKI
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize
from utils.util import AverageMeter
import os
import utils.losses as losses
from utils.metrics import iou_score
from network.SRS_RTS import SRS_R
from network.SRS_STS import SRS_S

import time
def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed) #CPU随机种子确定
    torch.cuda.manual_seed(seed) #GPU随机种子确定
    torch.cuda.manual_seed_all(seed) #所有的GPU设置种子
    torch.backends.cudnn.benchmark = False #模型卷积层预先优化关闭
    torch.backends.cudnn.deterministic = True #确定为默认卷积算法
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_torch(41)

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, default="/root/autodl-tmp/bus", help='dir')
parser.add_argument('--train_file_dir', type=str, default="bus_train3.txt", help='dir')
parser.add_argument('--val_file_dir', type=str, default="bus_val3.txt", help='dir')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu'),
parser.add_argument('--Dataset', type=str, default="BUS3",
                    help='Iris Dataset to choice'),
parser.add_argument('--gpu', default=1,type=int,
                    help='choice gpu'),
                  
args = parser.parse_args()



def getDataloader():
    img_size = 256
    train_transform = Compose([
        RandomRotate90(),
        transforms.Flip(),
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])
    db_train = MedicalDataSets(base_dir=args.base_dir, split="train",
                            transform=train_transform, train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
    db_val = MedicalDataSets(base_dir=args.base_dir, split="val", transform=val_transform,
                          train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
    print("train num:{}, val num:{}".format(len(db_train), len(db_val)))

    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=8, pin_memory=False)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    return trainloader, valloader


def get_model(args):
    model_R = SRS_R(bs=args.batch_size).cuda(args.gpu)
    model_R.load_state_dict(torch.load(os.path.join('权重文件/'+args.Dataset+'_SRS_R'+'.pth')))
    print("已经加载预训练权重")
    model_S = SRS_S(bs=args.batch_size).cuda(args.gpu)
    return model_R,model_S


def train(args):
    torch.cuda.set_device(args.gpu)
    base_lr = args.base_lr
    trainloader, valloader = getDataloader()
    model_r,model_s = get_model(args)
    print("train file dir:{} val file dir:{}".format(args.train_file_dir, args.val_file_dir))
    optimizer = optim.SGD(model_s.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    criterion = losses.__dict__['BCEDiceLoss']().cuda()

    print("{} iterations per epoch".format(len(trainloader)))
    best_iou = 0
    iter_num = 0
    max_epoch = 300
    max_iterations = len(trainloader) * max_epoch
    for epoch_num in range(max_epoch):
        model_s.train()
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter(),
                      'val_loss': AverageMeter(),
                      'val_iou': AverageMeter(),
                      'SE': AverageMeter(),
                      'PC': AverageMeter(),
                      'F1': AverageMeter(),
                      'ACC': AverageMeter()
                      }
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            with torch.no_grad():
            	batch_r = model_r(volume_batch)
            outputs = model_s(volume_batch,batch_r)
            loss = criterion(outputs, label_batch)
            iou, dice, _, _, _, _, _ = iou_score(outputs, label_batch)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            avg_meters['loss'].update(loss.item(), volume_batch.size(0))
            avg_meters['iou'].update(iou, volume_batch.size(0))
        model_s.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader):
                input, target = sampled_batch['image'], sampled_batch['label']
                input = input.cuda()
                target = target.cuda()
                batch_r = model_r(input)
                output = model_s(input,batch_r)
                loss = criterion(output, target)

                iou, _, SE, PC, F1, _, ACC = iou_score(output, target)
                avg_meters['val_loss'].update(loss.item(), input.size(0))
                avg_meters['val_iou'].update(iou, input.size(0))
                avg_meters['SE'].update(SE, input.size(0))
                avg_meters['PC'].update(PC, input.size(0))
                avg_meters['F1'].update(F1, input.size(0))
                avg_meters['ACC'].update(ACC, input.size(0))

        print(
            'epoch [%d/%d] lr: %.6f train_loss : %.4f, train_iou: %.4f '
            '- val_loss %.4f - val_iou %.4f - val_SE %.4f - val_PC %.4f - val_F1 %.4f - val_ACC %.4f'
            % (epoch_num, max_epoch, lr_ ,avg_meters['loss'].avg, avg_meters['iou'].avg,
               avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['SE'].avg,
               avg_meters['PC'].avg, avg_meters['F1'].avg, avg_meters['ACC'].avg))
        if avg_meters['val_iou'].avg > best_iou:
            torch.save(model_s.state_dict(), '权重文件/reconstruction消融/'+args.Dataset+'_SRS_STS'+'.pth')
            best_iou = avg_meters['val_iou'].avg
            print("=> saved best model")
    return "Training Finished!"

def trainer_SRS_S():
    train(args)

if __name__ == "__main__":
    train(args)

