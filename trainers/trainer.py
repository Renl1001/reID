# encoding: utf-8
import math
import time
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.loss import euclidean_dist, hard_example_mining
from utils.meters import AverageMeter


class Trainer:
    def __init__(self, opt, model, optimzier, criterion, summary_writer):
        self.opt = opt
        self.model = model
        self.optimizer= optimzier
        self.criterion = criterion
        self.summary_writer = summary_writer

    def train(self, epoch, data_loader):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        start = time.time()
        self.corrects = 0
        self.cnt = 0
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - start)
            # model optimizer
            self._parse_data(inputs)
            self._forward()
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

            self.corrects += float(torch.sum(self.preds == self.target.data))

            batch_time.update(time.time() - start)
            losses.update(self.loss.item())

            # tensorboard
            global_step = epoch * len(data_loader) + i
            self.summary_writer.add_scalar('loss', self.loss.item(), global_step)
            self.summary_writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], global_step)

            start = time.time()

            if (i + 1) % self.opt.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Batch Time {:.3f} ({:.3f})\t'
                      'Data Time {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.mean,
                              data_time.val, data_time.mean,
                              losses.val, losses.mean))
        param_group = self.optimizer.param_groups
        epoch_acc = self.corrects / self.cnt
        print('Epoch: [{}]\tEpoch Time {:.3f} s\tLoss {:.3f}\t'
              'Lr {:.2e}\tAcc {:.4f}[{}/{}]'
              .format(epoch, batch_time.sum, losses.mean, param_group[0]['lr'], epoch_acc, self.corrects, self.cnt))
        print()

    def _parse_data(self, inputs):
        imgs, pids, _ = inputs
        self.data = imgs
        self.cnt += imgs.shape[0]
        self.target = pids
        if torch.cuda.is_available():
            self.data = imgs.cuda()
            self.target = pids.cuda()

    def _forward(self):
        feat, score = self.model(self.data)
        _, preds = torch.max(score.data, 1)
        self.loss = self.criterion(feat, score, self.target)
        self.preds = preds
