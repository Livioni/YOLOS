# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .voc import build as build_voc
from .pandas import build as build_pandas
from .mot17det import build as build_mot17
from .mot15det import build as build_mot15


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'voc':
        return build_voc(image_set, args)
    if args.dataset_file == 'pandas' or args.dataset_file == 'PANDAS':
        return build_pandas(image_set, args)
    if args.dataset_file == 'mot17' or args.dataset_file == 'MOT17':
        return build_mot17(image_set, args)
    if args.dataset_file == 'mot15' or args.dataset_file == 'MOT15':
        return build_mot15(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
