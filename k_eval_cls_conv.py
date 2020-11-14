#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Wenxuan Wu, Zhongang Qi, Li Fuxin.
@Contact: wuwen@oregonstate.edu
@File: eval_cls_conv.py

Modified by 
@Author: Jiawei Chen, Linlin Li
@Contact: jc762@duke.edu
@File: k_eval_cls_conv.py
"""

import argparse
import os
import sys
import numpy as np 
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from data_utils.ModelNetDataLoader import ModelNetDataLoader, load_data
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
from utils.utils import test, save_checkpoint
from model.pointconv import PointConvDensityClsSsg as PointConvClsSsg


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointConv')
    parser.add_argument('--batchsize', type=int, default=16, help='batch size')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--kb1checkpoint', type=str, default=None, help='k but 1 checkpoint')
    parser.add_argument('--binarycheckpoint', type=str, default=None, help='binary checkpoint')
    parser.add_argument('--num_view', type=int, default=3, help='num of view')
    parser.add_argument('--model_name', default='my_pointconv', help='model name')
    return parser.parse_args()

def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    datapath = './data/ModelNet/'

    '''CREATE DIR'''
    experiment_dir = Path('./eval_experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%sModelNet40-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    os.system('cp %s %s' % (args.kb1checkpoint, checkpoints_dir))
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'eval_%s_cls.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------EVAL---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    train_data, train_label, test_data, test_label = load_data(datapath, classification=True)
    logger.info("The number of training data is: %d",train_data.shape[0])
    logger.info("The number of test data is: %d", test_data.shape[0])
    testDataset = ModelNetDataLoader(test_data, test_label)
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=args.batchsize, shuffle=False)
    

    '''MODEL LOADING'''
    num_class = 39
    kb1classifier = PointConvClsSsg(num_class).cuda()
    if args.kb1checkpoint is not None:
        print('Load k but 1 CheckPoint...')
        logger.info('Load k but 1 CheckPoint')
        kb1checkpoint = torch.load(args.kb1checkpoint)
        start_epoch = kb1checkpoint['epoch']
        kb1classifier.load_state_dict(kb1checkpoint['model_state_dict'])
    else:
        print('Please load k but 1 Checkpoint to eval...')
        sys.exit(0)
        start_epoch = 0
        
    num_class1 = 2
    binaryclassifier = PointConvClsSsg(num_class1).cuda()
    if args.binarycheckpoint is not None:
        print('Load binary CheckPoint...')
        logger.info('Load binary CheckPoint')
        binarycheckpoint = torch.load(args.binarycheckpoint)
        start_epoch = binarycheckpoint['epoch']
        binaryclassifier.load_state_dict(binarycheckpoint['model_state_dict'])
    else:
        print('Please load binary Checkpoint to eval...')
        sys.exit(0)
        start_epoch2 = 0

    blue = lambda x: '\033[94m' + x + '\033[0m'

    '''EVAL'''
    logger.info('Start evaluating...')
    print('Start evaluating...')

    total_correct = 0
    total_seen = 0
    preds = []
    for batch_id, data in tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9):
        pointcloud, target = data
        target = target[:, 0]
        #import ipdb; ipdb.set_trace()
        pred_view = torch.zeros(pointcloud.shape[0], num_class).cuda()
        binary_view = torch.zeros(pointcloud.shape[0], num_class1).cuda()

        for _ in range(args.num_view):
            pointcloud = generate_new_view(pointcloud)
            #import ipdb; ipdb.set_trace()
            #points = torch.from_numpy(pointcloud).permute(0, 2, 1)
            points = pointcloud.permute(0, 2, 1)
            points, target = points.cuda(), target.cuda()
            kb1classifier = kb1classifier.eval()
            binaryclassifier = binaryclassifier.eval()
            with torch.no_grad():
                pred = kb1classifier(points)
                pred_binary = binaryclassifier(points)
            pred_view += pred
            binary_view += pred_binary        
        
        kb1_logprob = pred_view.data
        binary_logprob = binary_view.data
        ## since we assigned the composite class the largest label, we will split the log-probability for the last label to two part, one for binary 0 and one for binary 1.
        binary_pred_logprob = kb1_logprob[:,-1].reshape(1,len(kb1_logprob[:,-1])).transpose(0,1).repeat(1,2).view(-1, 2) + binary_logprob
        ## concatenate to get log-probability for all (40) classes 
        pred_logprob = torch.from_numpy(np.c_[kb1_logprob[:,0:-1].cpu().detach().numpy(), binary_pred_logprob.cpu().detach().numpy()]).to('cuda')
        pred_choices = pred_logprob.max(1)[1] 
        
        ## reset labels
        mapper_dict = {**{key: key + 1 for key in range(12, 32)}, **{key: key + 2 for key in range(32, 38)}, **{38: 33, 39: 12}}

        def mp(entry):
            return mapper_dict[entry] if entry in mapper_dict else entry
        mp = np.vectorize(mp)
        
        pred_choice = torch.from_numpy(np.array(mp(pred_choices.cpu().detach().numpy()))).to('cuda')
        preds.append(pred_choice.cpu().detach().numpy())
        correct = pred_choice.eq(target.long().data).cpu().detach().numpy().sum()
        total_correct += correct.item()
        total_seen += float(points.size()[0])

    accuracy = total_correct / total_seen
    ## confusion matrix
    cm = confusion_matrix(test_label.ravel(), np.concatenate(preds).ravel())
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    t = pd.read_table('data/ModelNet/shape_names.txt', names = ['label'])
    d = {key:val for key,val in zip(t.label, cm.diagonal())}
    print('Total Accuracy: %f'%accuracy)
    print('Accuracy per class:', d)

    logger.info('Total Accuracy: %f'%accuracy)
    logger.info('End of evaluation...')

def generate_new_view(points):
    points_idx = np.arange(points.shape[1])
    np.random.shuffle(points_idx)

    points = points[:, points_idx, :]
    return points


def rotate_point_cloud_by_angle(data, rotation_angle):
    """
    Rotate the point cloud along up direction with certain angle.
    :param batch_data: Nx3 array, original batch of point clouds
    :param rotation_angle: range of rotation
    :return:  Nx3 array, rotated batch of point clouds
    """
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]], dtype=np.float32)
    rotated_data = np.dot(data, rotation_matrix)

    return rotated_data

if __name__ == '__main__':
    args = parse_args()
    main(args)
