# encoding: utf-8
import os
import sys
from os import path as osp
from pprint import pprint

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from config import opt
from datasets import data_manager
from datasets.data_loader import ImageData
from datasets.samplers import RandomIdentitySampler
from torchvision.datasets import ImageFolder
from models.networks import ResNetBuilder
from trainers.evaluator import ResNetEvaluator
from trainers.trainer import Trainer
from utils.loss import TripletLoss
from utils.serialization import Logger, save_checkpoint
from utils.transforms import TestTransform, TrainTransform
from utils.lr_scheduler import WarmupMultiStepLR


def train(**kwargs):
    opt._parse(kwargs)

    os.makedirs(opt.save_dir, exist_ok=True)
    use_gpu = torch.cuda.is_available()
    sys.stdout = Logger(osp.join(opt.save_dir, 'log_train.txt'))

    print('=========user config==========')
    pprint(opt._state_dict())
    print('============end===============')

    if use_gpu:
        print('currently using GPU')
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(opt.seed)
    else:
        print('currently using cpu')

    print('initializing dataset {}'.format(opt.dataset))
    dataset = data_manager.init_dataset(name=opt.dataset, use_all = opt.use_all)

    summary_writer = SummaryWriter(osp.join(opt.save_dir, 'tensorboard_log'))
    # load data
    pin_memory = True if use_gpu else False
    dataloader = load_data(dataset, pin_memory)

    print('initializing model ...')
    if opt.loss == 'softmax' or opt.loss == 'softmax_triplet':
        model = ResNetBuilder(dataset.num_train_pids, opt.last_stride, True)
    elif opt.loss == 'triplet':
        model = ResNetBuilder(None, opt.last_stride, True)
    
    if opt.pretrained_model:
        state_dict = torch.load(opt.pretrained_model)['state_dict']
        model.load_state_dict(state_dict, False)
        print('load pretrained model ' + opt.pretrained_model)
        
    print('model size: {:.5f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

    optim_policy = model.get_optim_policy()
    if use_gpu:
        model = nn.DataParallel(model).cuda()
    reid_evaluator = ResNetEvaluator(model)

    if opt.evaluate:
        reid_evaluator.evaluate(dataloader['query'], dataloader['gallery'], 
            dataloader['queryFlip'], dataloader['galleryFlip'], savefig=opt.savefig)
        return

    criterion = get_loss()
    

    # optimizer
    if opt.optim == "sgd":
        optimizer = torch.optim.SGD(optim_policy, lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = torch.optim.Adam(optim_policy, lr=opt.lr, weight_decay=5e-4)

    scheduler = WarmupMultiStepLR(optimizer, [40, 70], 0.1, 0.01,
                                  10, 'linear')

    start_epoch = opt.start_epoch
    # get trainer and evaluator
    reid_trainer = Trainer(opt, model, optimizer, criterion, summary_writer)


    # start training
    best_rank1 = opt.best_rank
    best_epoch = 0
    for epoch in range(start_epoch, opt.max_epoch):
        scheduler.step()

        reid_trainer.train(epoch, dataloader['train'])

        # skip if not save model
        if opt.eval_step > 0 and (epoch + 1) % opt.eval_step == 0 or (epoch + 1) == opt.max_epoch:
            rank1 = reid_evaluator.evaluate(dataloader['query'], dataloader['gallery'], 
            dataloader['queryFlip'], dataloader['galleryFlip'])
    
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            
            state_dict = model.state_dict()
            save_checkpoint({'state_dict': state_dict, 'epoch': epoch + 1}, 
                is_best=is_best, save_dir=opt.save_dir, 
                filename='checkpoint_ep' + str(epoch + 1) + '.pth.tar')

    print('Best rank-1 {:.1%}, achived at epoch {}'.format(best_rank1, best_epoch))

def get_loss():
    xent_criterion = nn.CrossEntropyLoss()
    triplet = TripletLoss(opt.margin)
    if opt.loss == 'softmax':
        def criterion(feat, score, labels):
            return xent_criterion(score, labels)
    elif opt.loss == 'triplet':
        def criterion(feat, score, labels):
            return triplet(feat, labels)[0]
    else:
        def criterion(feat, score, labels):
            return xent_criterion(score, labels)+triplet(feat, labels)[0]
    
    return criterion

def load_data(dataset, pin_memory):
    dataloader = {}
    if opt.loss=='softmax':
        trainloader = DataLoader(
            ImageData(dataset.train, TrainTransform()),
            shuffle=True,
            batch_size=opt.train_batch, num_workers=8,
            pin_memory=pin_memory, drop_last=True
        )
    else:
        trainloader = DataLoader(
            ImageData(dataset.train, TrainTransform()),
            sampler=RandomIdentitySampler(dataset.train, opt.train_batch, opt.num_instances),
            batch_size=opt.train_batch, num_workers=8,
            pin_memory=pin_memory, drop_last=True
        )
    dataloader['train'] = trainloader
    queryloader = DataLoader(
        ImageData(dataset.query, TestTransform()),
        batch_size=opt.test_batch, num_workers=8,
        pin_memory=pin_memory
    )
    dataloader['query'] = queryloader
    galleryloader = DataLoader(
        ImageData(dataset.gallery, TestTransform()),
        batch_size=opt.test_batch, num_workers=8,
        pin_memory=pin_memory
    )
    dataloader['gallery'] = galleryloader
    queryFliploader = DataLoader(
        ImageData(dataset.query, TestTransform(True)),
        batch_size=opt.test_batch, num_workers=8,
        pin_memory=pin_memory
    )
    dataloader['queryFlip'] = queryFliploader
    galleryFliploader = DataLoader(
        ImageData(dataset.gallery, TestTransform(True)),
        batch_size=opt.test_batch, num_workers=8,
        pin_memory=pin_memory
    )
    dataloader['galleryFlip'] = galleryFliploader
    return dataloader

if __name__ == '__main__':
    import fire
    fire.Fire()
