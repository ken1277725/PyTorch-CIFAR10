""" Use ResNet to classify CIFAR images """

from __future__ import print_function

import os
import csv
import time
import shutil
import numpy as np

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from model.resnet import resnet50
from model.vgg import vgg16_bn

from util.data.dataset import CifarDataset
from util.data.dataloader import CifarDataloader
from util.train import train
from util.test import test
from util.checkpoint import save_checkpoint, load_checkpoint

from tensorboardX import SummaryWriter 
writer = SummaryWriter('runs/exp-4')


TRAIN_CSV_PATH = os.path.join('csv', 'train_labels.csv')
TRAIN_IMG_PATH = os.path.join('image', 'train')
TEST_CSV_PATH = os.path.join('csv', 'test_labels.csv')
TEST_IMG_PATH = os.path.join('image', 'test')


USE_GPU = True


################## Part:0 Begin of ArgumentParse Setting##################

import argparse
from argparse import ArgumentParser

parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# batch size parser , use -b={batchsize} to set.
parser.add_argument("-b","--batch-size", help="bach_size",default=128, type=int,dest="batch_size")

# epoch parser , use -e={epoch} to set. 
parser.add_argument("-e","--epoch",help="how many epoches will run , default will be 20",default=12, type=int,dest="epoch")

# Learning Rate 
 parser.add_argument("-lr","--leraning_rate",help="how the leraning_rate will be, default will be 1e-3",default=1e-3, type=float,dest="leraning_rate")


args = parser.parse_args()
################## Part:0 End of ArgumentParse Setting##################



################## PyTorch: Part:1-t Begining of horovod initialization Setting for PyTorch ##################
import horovod.torch as hvd
hvd.init()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))
################## PyTorch: Part:1-t End of horovod initialization Setting for PyTorch ##################




EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.leraning_rate

def main():


    #Data preprocessing for Transfrom , and Normalize the data with the imageNet weight
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    #Get traing data and test data
    train_dataset = CifarDataset(TRAIN_CSV_PATH, TRAIN_IMG_PATH, transformations)
    train_loader = CifarDataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_dataset = CifarDataset(TEST_CSV_PATH, TEST_IMG_PATH, transformations)
    test_loader = CifarDataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


    #choose the model with resnet50
    model = resnet50(pretrained=True, num_classes=10)
    criterion = nn.CrossEntropyLoss()

    if USE_GPU:
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # load_checkpoint(os.path.join('checkpoint', 'last_checkpoint.pth.tar'), model, optimizer)

    for epoch in range(EPOCHS):
        train(train_loader, model, criterion, optimizer, epoch+1, USE_GPU,writer = writer)
        test(test_loader, model, USE_GPU)
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join('checkpoint'))

if __name__ == "__main__":
    main()
