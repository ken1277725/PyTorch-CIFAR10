from __future__ import division
from __future__ import print_function

import os
import time
import shutil

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from torch.autograd import Variable
from tensorboardX import SummaryWriter


#ummaryWriter('runs/exp-1')
#writer = SummaryWriter('runs/exp-1')

#ummaryWriter('runs/exp-1')

MSG_DISPLAY_FREQ = 200


def train(train_loader, model, criterion, optimizer, epoch, USE_GPU=False,writer=None):

    batch_time = 0.0
    # switch to train mode
    model.train()

    end = time.time()

    running_loss = 0.0
    total_batch_num = len(train_loader)
    for i, (inputs, labels) in enumerate(train_loader):
        n_iter = total_batch_num * epoch + i
        labels = torch.squeeze(labels, 1)

        if USE_GPU:
            inputs, labels = Variable(inputs).cuda(async=True), Variable(labels).cuda(async=True)
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]

        batch_time += time.time()-end
        end = time.time()
        if writer != None :
            writer.add_scalar('data/s1',loss.data[0],n_iter)
            if i % MSG_DISPLAY_FREQ == (MSG_DISPLAY_FREQ-1):
                #print("loss_value:{}".format(loss_value.data.item()))
                
                out = torch.cat((outputs.data, torch.ones(len(outputs), 1)), 1)
                writer.add_embedding(out, metadata=labels, label_img=inputs, global_step=n_iter)
        if i % MSG_DISPLAY_FREQ == (MSG_DISPLAY_FREQ-1):
            print("[{}][{}/{}]\t Loss: {:0.5f}\t Batch time: {:0.3f}s".format(epoch, i+1, len(train_loader), running_loss/MSG_DISPLAY_FREQ, batch_time/MSG_DISPLAY_FREQ))
            running_loss = 0.0
            batch_time = 0.0
