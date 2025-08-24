import argparse
import random
from dataloader.dataset import MedicalDataSets
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
from network.SRS_R import SRS_R

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
    db_val = MedicalDataSets(base_dir=args.base_dir, split="train", transform=val_transform,
                          train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
    print("train num:{}, val num:{}".format(len(db_train), len(db_val)))

    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=8, pin_memory=False)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    return trainloader, valloader


def get_model(args):
    model_R = SRS_R(bs=args.batch_size).cuda(args.gpu)
#    model_R.load_state_dict(torch.load(os.path.join(r"./权重文件/unet_r_model.pth")))
    return model_R


def train(args):
    torch.cuda.set_device(args.gpu)
    base_lr = args.base_lr
    trainloader, valloader = getDataloader()
    model_R = get_model(args)
    print("train file dir:{} val file dir:{}".format(args.train_file_dir, args.val_file_dir))
    optimizer = optim.SGD(model_R.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    criterion = torch.nn.MSELoss().cuda()

    print("{} iterations per epoch".format(len(trainloader)))
    best_iou = 0
    iter_num = 0
    max_epoch = 100
    best_loss = 1
    max_iterations = len(trainloader) * max_epoch
    for epoch_num in range(max_epoch):
        model_R.train()
        avg_meters = {'loss': AverageMeter(), 'val_loss': AverageMeter()}
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch = sampled_batch['image'].cuda()
            outputs = model_R(volume_batch)
            loss = criterion(outputs, volume_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            avg_meters['loss'].update(loss.item(), volume_batch.size(0))
        model_R.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader):
                input = sampled_batch['image'].cuda()
                output = model_R(input)
                loss = criterion(output, input)
                avg_meters['val_loss'].update(loss.item(), input.size(0))

        print('epoch [%d/%d]  train_loss : %.8f  val_loss : %.8f' % (epoch_num, max_epoch, avg_meters['loss'].avg,
                                                                     avg_meters['val_loss'].avg))
        if avg_meters['val_loss'].avg < best_loss and epoch_num > 1:
            torch.save(model_R.state_dict(), '权重文件/'+args.Dataset+'_SRS_R'+'.pth')
            best_loss = avg_meters['val_loss'].avg
            print("=> saved best model")
    return "Training Finished!"

def trainer_SRS_R():
    train(args)

if __name__ == "__main__":
    train(args)
