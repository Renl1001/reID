# encoding: utf-8
import warnings
import numpy as np


class DefaultConfig(object):
    seed = 0

    # dataset options
    dataset = 'market1501'
    use_all = False
    data_aug = 'RandomErase'
    # optimization options
    loss = 'softmax'  # triplet, softmax_triplet
    optim = 'Adam'
    max_epoch = 120
    train_batch = 64 
    test_batch = 64
    warmup = False # warmup
    lr = 0.00035
    margin = 0.3
    num_instances = 4
    num_gpu = 1

    # eval
    evaluate = False
    savefig = None #./save 
    findid = None

    # model options
    last_stride = 1
    pretrained_model = None #./pytorch-ckpt/market
    
    # miscs
    print_freq = 10
    eval_step = 30
    save_dir = './pytorch-ckpt/market'
    start_epoch = 0
    best_rank = -np.inf

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
           
    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in DefaultConfig.__dict__.items()
                if not k.startswith('_')}

opt = DefaultConfig()
