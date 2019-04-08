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
from trainers.trainer import cls_tripletTrainer
from utils.loss import CrossEntropyLabelSmooth, TripletLoss
from utils.serialization import Logger, save_checkpoint
from utils.transforms import TestTransform, TrainTransform


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
    
    optim_policy = model.get_optim_policy()

    if opt.pretrained_model:
        state_dict = torch.load(opt.pretrained_model)['state_dict']
        #state_dict = {k: v for k, v in state_dict.items() \
        #        if not ('reduction' in k or 'softmax' in k)}
        model.load_state_dict(state_dict, False)
        print('load pretrained model ' + opt.pretrained_model)
        
    print('model size: {:.5f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

    if use_gpu:
        model = nn.DataParallel(model).cuda()
    reid_evaluator = ResNetEvaluator(model)

    if opt.evaluate:
        reid_evaluator.evaluate(dataloader['query'], dataloader['gallery'], 
            dataloader['queryFlip'], dataloader['galleryFlip'], re_ranking=opt.re_ranking, savefig=opt.savefig)
        return

    xent_criterion = nn.CrossEntropyLoss()
    # xent_criterion = CrossEntropyLabelSmooth(dataset.num_train_pids, use_gpu=use_gpu)

    

    def criterion(feat, score, labels):
        loss = xent_criterion(score, labels)
        if opt.loss == 'triplet' or opt.loss == 'softmax_triplet':
            embedding_criterion = TripletLoss(opt.margin)
            loss += embedding_criterion(feat, labels)[0]
        return loss

    # get optimizer
    if opt.optim == "sgd":
        optimizer = torch.optim.SGD(optim_policy, lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = torch.optim.Adam(optim_policy, lr=opt.lr, weight_decay=5e-4)


    start_epoch = opt.start_epoch
    # get trainer and evaluator
    reid_trainer = cls_tripletTrainer(opt, model, optimizer, criterion, summary_writer)

    def adjust_lr(optimizer, ep):
        if ep < 50:
            lr = 1e-4*(ep//5+1)
        elif ep < 200:
            lr = 1e-3
        elif ep < 300:
            lr = 1e-4
        else:
            lr = 1e-5
        for p in optimizer.param_groups:
            p['lr'] = lr

    # start training
    best_rank1 = opt.best_rank
    best_epoch = 0
    for epoch in range(start_epoch, opt.max_epoch):
        if opt.adjust_lr:
            adjust_lr(optimizer, epoch + 1)

        reid_trainer.train(epoch, dataloader['train'])

        # skip if not save model
        if opt.eval_step > 0 and (epoch + 1) % opt.eval_step == 0 or (epoch + 1) == opt.max_epoch:
            rank1 = reid_evaluator.evaluate(dataloader['query'], dataloader['gallery'], 
            dataloader['queryFlip'], dataloader['galleryFlip'])
    
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({'state_dict': state_dict, 'epoch': epoch + 1}, 
                is_best=is_best, save_dir=opt.save_dir, 
                filename='checkpoint_ep' + str(epoch + 1) + '.pth.tar')

    print('Best rank-1 {:.1%}, achived at epoch {}'.format(best_rank1, best_epoch))

def load_data(dataset, pin_memory):
    dataloader = {}
    trainloader = DataLoader(
        ImageData(dataset.train, TrainTransform()),
        sampler=RandomIdentitySampler(dataset.train, opt.num_instances),
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
