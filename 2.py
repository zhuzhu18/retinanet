import argparse
import collections

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

# from retinanet import coco_eval
# from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', default='coco')
    parser.add_argument('--coco_path', help='Path to COCO directory', default='/media/zhuzhu/ec114170-f406-444f-bee7-a3dc0a86cfa2/dataset/coco')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

    parser.add_argument('--use-gpu', help='training on cpu or gpu', action='store_false', default=True)
    parser.add_argument('--device-ids', help='GPU device ids', default=[0])

    args = parser.parse_args()

    # ------------------------------ Create the data loaders -----------------------------
    if args.dataset == 'coco':

        if args.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(args.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        # dataset_val = CocoDataset(args.coco_path, set_name='val2017',
        #                           transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)
    # sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    # dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    cnt = 0
    for data in dataloader_train:
        print(data['img'].size())
        print(data['annot'].size())
        cnt += 1
        if cnt == 10:
            exit()
main()