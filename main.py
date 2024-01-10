import json
import os
import shutil
import time

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from data.prepare_data import generate_dataloader  # Prepare the data and dataloader
from models.resnet import resnet  # The model construction
from opts import opts  # The options for the project
from trainer import train  # For the training process
from trainer import validate  # For the validate (test) process
from trainer import adjust_learning_rate
from models.DomainClassifierTarget import DClassifierForTarget
from models.DomainClassifierSource import DClassifierForSource
from models.EntropyMinimizationPrinciple import EMLossForTarget
import ipdb

best_prec1 = 0

def main():
    global args, best_prec1
    current_epoch = 0
    epoch_count_dataset = 'source' ##
    args = opts()
    if args.arch == 'alexnet':
        raise ValueError('the request arch is not prepared', args.arch)
        # model = alexnet(args)
        # for param in model.named_parameters():
        #     if param[0].find('features1') != -1:
        #         param[1].require_grad = False
    elif args.arch.find('resnet') != -1:
        model = resnet(args)
    else:
        raise ValueError('Unavailable model architecture!!!')
    # define-multi GPU
    #分块在指定设备之间拆分输入来并行化给定模块的应用进程。
    model = torch.nn.DataParallel(model).cuda()
    print(model)
    criterion_classifier_target = DClassifierForTarget(nClass=args.num_classes).cuda()
    criterion_classifier_source = DClassifierForSource(nClass=args.num_classes).cuda()
    criterion_em_target = EMLossForTarget(nClass=args.num_classes).cuda()
    criterion = nn.CrossEntropyLoss().cuda()