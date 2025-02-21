"""
randomly generate index
"""

from __future__ import print_function

import os
import argparse
import socket
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import random
#from models import model_dict
import torch.nn.functional as F

from dataset.cifar100 import get_cifar100_dataloaders
from dataset.imagenet import get_imagenet_dataloader

import re
import math
from torchvision import datasets, transforms

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100','imagenet'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='vgg8',
                        choices=['wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 
                                 'resnet8x4', 'MobileNetV2', 'MobileNet', 
                                 'RESNET18','RESNET34','RESNET50'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    parser.add_argument('--ratio', type=str, default='30', choices=['30', '35', '40', '45', '50', '70'], help='selection ratio')
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    opt = parser.parse_args()

    return opt

def load_network(model_path, model_t, n_cls):
    print('==> loading model')
    model = model_dict[model_t](num_classes=n_cls)
    
    if model_t in ['RESNET34','RESNET50', 'repvgg']:
        model.load_state_dict(torch.load(model_path))
    elif model_t in ['densenet121', 'densenet169', 'densenet201', 'densenet161']:
        pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(model_path)
        for key in list(state_dict.keys()):
          res = pattern.match(key)
          if res:
              new_key = res.group(1) + res.group(2)
              state_dict[new_key] = state_dict[key]
              del state_dict[key]
        model = model_dict[model_t](num_classes=n_cls)
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(torch.load(model_path)['model'])
        print(torch.load(model_path)['accuracy'])
    print('==> done')
    return model

def load_network2(model_path, model_t, n_cls):
    print('==> loading model')
    model = model_dict[model_t](num_classes=n_cls)  
    model.load_state_dict(torch.load(model_path)['model'])
    print(torch.load(model_path)['accuracy'])
    print('==> done')
    return model
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def has_duplicates(seq):
    return len(seq) != len(set(seq))
        
def main():
    opt = parse_option()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    
    # fix seed
    print('Seed = ', int(opt.trial) * 10)
    setup_seed(int(opt.trial) * 10)
    
    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                    num_workers=opt.num_workers,
                                                                    is_instance=True)
        n_cls = 100
        total_num = 50000
    elif opt.dataset == 'imagenet':
        n_cls = 1000
        opt.save_freq = 30
        opt.batch_size = 256
        opt.num_workers = 16
        opt.epochs = 100
        opt.learning_rate = 0.1
        opt.lr_decay_epochs = [30,60,90]
        opt.lr_decay_rate = 0.1
        opt.weight_decay = 1e-4
        opt.momentum = 0.9
        train_loader, val_loader, n_data = get_imagenet_dataloader(batch_size=opt.batch_size, 
                                                                   num_workers=opt.num_workers, 
                                                                   is_instance=True)
        total_num = 1281167
    else:
        raise NotImplementedError(opt.dataset)
                                                                
    '''    
    # load teacher network
    model_t = load_network('./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth', 'resnet32x4', n_cls)
    model_t.cuda()
    model_t.eval()
    '''                

    num_per_class = int(total_num / n_cls * (int(opt.ratio) / 100) ) 
    selected_index = []
    count_per_class = np.zeros(n_cls)
    for idx, data in enumerate(train_loader):
        input, target, index = data
        #input = input.cuda()
        #target = target.cuda()    
        for i in range(0,input.size(0)):
            if count_per_class[target[i]] < num_per_class:
                #logit_t = model_t(input[i].unsqueeze(0))
                selected_index.append(int(index[i]))
                count_per_class[target[i]] = count_per_class[target[i]] + 1           
    
    if has_duplicates(selected_index):
        print("contains duplicate index!!!!")
    else:
        print("no duplicate index!!!!")
    print("total number of training samples: " + str(len(selected_index)))
    print("sum of index: " + str(sum(selected_index)))
    print(" ")
    np.save('ratio_'+opt.ratio+'.npy', selected_index)
    
if __name__ == '__main__':
    main()
