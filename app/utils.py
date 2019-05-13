import codecs
import json
import os
import time
import torch
import pandas as pd
import numpy as np
from models.networks import ResNetBuilder
from utils.transforms import TestTransform
from PIL import Image
import matplotlib.pyplot as plt

print('initializing model ...')
model = ResNetBuilder(751, 1, True)
pretrained_model = 'pytorch-ckpt/softmax_triplet/checkpoint_ep120.pth.tar'
use_gpu = torch.cuda.is_available()
if use_gpu:
    state_dict = torch.load(pretrained_model)['state_dict']
else:
    state_dict = torch.load(pretrained_model, map_location='cpu')['state_dict']
model.load_state_dict(state_dict, False)
print('load pretrained model ' + pretrained_model)    
print('model size: {:.5f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

gf = torch.Tensor(np.load(os.path.join('save', 'feature', 'gf.npy')))
df = pd.read_csv(os.path.join('app', 'data', 'g_data.csv'))

g_pids = df['g_pids'].tolist()
g_camids = df['g_camids'].tolist()
g_path = df['g_path'].tolist()
print("Extracted features for gallery set: {} x {}".format(gf.size(0), gf.size(1)))

def read_image(img_path):
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def recognition(path):
    model.eval()

    # qf, q_pids, q_camids = [], [], []
    
    inputs = read_image(path)
    transform = TestTransform()
    inputs = transform(inputs)
    inputs = inputs.reshape(-1,3,256,128)

    # 计算特征
    with torch.no_grad():
        feature = model(inputs)
    qf = feature.cpu()
    qf = torch.Tensor(qf)
    
    m, n = qf.size(0), gf.size(0)
    q_g_dist = torch.pow(qf, 2).sum().expand(m, n) + \
        torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    q_g_dist.addmm_(1, -2, qf, gf.t())

    distmat = q_g_dist
    distmat = distmat.numpy()

    indices = np.argsort(distmat, axis=1)

    image_paths = []
    for j in range(15):
        gallery_index = indices[0][j]
        img_path = g_path[gallery_index]
        img_path = os.path.join('static','images',img_path)
        image_paths.append(img_path)
    return image_paths
