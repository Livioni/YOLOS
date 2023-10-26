
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse,copy,os
import random
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
import util.misc as utils
from models.detector import ProgressivelyDrop,PostProcess
from torch.utils.data import DataLoader
import warnings
import json
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.coco_eval import CocoEvaluator

# 忽略UserWarning
warnings.simplefilter(action='ignore', category=UserWarning)


def get_args_parser():
    parser = argparse.ArgumentParser('Set YOLOS', add_help=False)

    parser.add_argument('--use_checkpoint', action='store_true',
                        help='use checkpoint.checkpoint to save mem')
    parser.add_argument('--num_class', default=1, type=int,
                        help='num of classes in the dataset')
    parser.add_argument('--token_drop', default=True, action='store_true',
                        help='whether to reuse the token in the image')
    parser.add_argument('--drop_proportion', default=0.1, type=float,
                        help='the proportion of the patch to mask')
    parser.add_argument('--dataset_file', default='mot15', type=str,
                        help='the dataset to train on')
    parser.add_argument('--vals_folder', default='/home/livion/Documents/github/dataset/MOT15_coco/val', type=str,
                        help='the folder of the validation set')
    parser.add_argument('--coco_path', type=str,default='/home/livion/Documents/github/dataset/MOT15_coco/')
    parser.add_argument('--eval_size', default=512, type=int,
                        help='the size of the validation set')

    # * model setting
    parser.add_argument("--det_token_num", default=100, type=int,
                        help="Number of det token in the deit backbone")
    parser.add_argument('--backbone_name', default='tiny', type=str,
                        help="Name of the deit backbone to use")
    parser.add_argument('--pre_trained', default='',
                        help="set imagenet pretrained model path if not train yolos from scratch")
    parser.add_argument("--max_reuse_frame", default=5, type=int,
                        help="reuse frame interval")
    # dataset parameters
    parser.add_argument('--output_dir', default='results',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--input',default='inputs/ADL-Rundle-8/ADL-Rundle-8-000498.jpg',type=str)
    parser.add_argument('--reuse_image',default='inputs/ADL-Rundle-8/ADL-Rundle-8-000491.jpg',type=str)

    return parser

def extract_bboxes_from_coco(json_path, image_name):
    """
    从COCO格式的json文件中提取指定图片的bbox信息。

    参数:
    - json_path: JSON文件的路径。
    - image_name: 要提取bbox信息的图片的名称。

    返回:
    - bboxes: 指定图片的所有bbox信息。
    """
    with open(json_path, 'r') as file:
        data = json.load(file)

    # 查找指定图片的ID
    image_id = None
    for image in data['images']:
        if image['file_name'] == image_name:
            image_id = image['id']
            break

    if image_id is None:
        print(f"No image with the name {image_name} found.")
        return []

    # 提取与该图片ID关联的所有bbox信息
    bboxes = []
    for annotation in data['annotations']:
        if annotation['image_id'] == image_id:
            bbox = annotation['bbox']
            # 转换COCO的bbox格式 [x_top_left, y_top_left, width, height] 到 [x1, y1, x2, y2]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            bboxes.append(bbox)

    return bboxes

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox).to('cpu')
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def val_dataset_preporcess(args):
    image_paths = []
    reuse_image_paths = []
    files = sorted(os.listdir(args.vals_folder))
    cnt = 0
    for index,file in enumerate(files):
        file_part = file.split('-')
        file_id = file_part[-1][0:-4]
        reuse_frame_id = int(file_id) - cnt
        if reuse_frame_id >= 1000:
            reuse_file_name = file[:-8] + str(reuse_frame_id) + '.jpg'
        else:
            reuse_file_name = file[:-7] + str(reuse_frame_id) + '.jpg'
        reuse_frame_path = os.path.join(args.vals_folder,reuse_file_name)
        image_paths.append(os.path.join(args.vals_folder,file))
        if os.path.exists(reuse_frame_path):
            reuse_image_paths.append(reuse_frame_path)
            cnt += 1
            if cnt == args.max_reuse_frame:
                cnt = 0
        else:
            reuse_image_paths.append(os.path.join(args.vals_folder,file))
            cnt = 0
    return image_paths, reuse_image_paths

def main(args, init_pe_size, mid_pe_size, resume):
    # utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = ProgressivelyDrop(
        num_classes=args.num_class, #类别数91
        pre_trained= args.pre_trained, #pre_train模型pth文件
        det_token_num=args.det_token_num, #100个det token
        backbone_name=args.backbone_name, #vit backbone的型号
        init_pe_size=init_pe_size, #初始化position embedding大小
        mid_pe_size=mid_pe_size, #mid position embedding 大小
        use_checkpoint=args.use_checkpoint, #是否使用checkpoint
    )
    model.to(device)
    checkpoint = torch.load(resume, map_location=args.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    ######build_dataset########
    dataset_val = build_dataset(image_set='val', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, batch_size=1, sampler=sampler_val,
                                drop_last=False, collate_fn=utils.collate_fn, num_workers=2)
    
    base_ds = get_coco_api_from_dataset(dataset_val)
    postprocessors = {'bbox': PostProcess()}
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    image_paths, reuse_image_paths= val_dataset_preporcess(args)
    for image in tqdm(image_paths):
        files = sorted(os.listdir(args.vals_folder))
        index = files.index(image.split('/')[-1])
        image_tensor, target = dataset_val[index]
        patch_dim_w, patch_dim_h = image_tensor.shape[2] // 16, image_tensor.shape[1] // 16
        patch_num = patch_dim_h * patch_dim_w

        bboxes = target['boxes']
        ground_truth_bboxes = rescale_bboxes(torch.tensor(bboxes), [image_tensor.shape[-1],image_tensor.shape[-2]]).tolist()  
                
        all_indices = set(range(patch_num))
        for bbox in ground_truth_bboxes:
            x1, y1, x2, y2 = bbox
            # 计算与bounding box相关的patch的开始和结束索引
            start_row_idx = y1 // 16
            start_col_idx = x1 // 16
            end_row_idx = y2 // 16
            end_col_idx = x2 // 16
            
            # 从所有的patch中移除当前bounding box内的patch
            for i in range(int(start_row_idx), int(end_row_idx)+1):
                for j in range(int(start_col_idx), int(end_col_idx)+1):
                    patch_idx = i * patch_dim_w + j
                    all_indices.discard(patch_idx)

        if args.token_drop:
            mask_num = int(len(all_indices) * args.drop_proportion)
            # 从除所有bounding boxes外的patches中随机选择要mask的patches
            row = np.random.choice(list(all_indices), size=mask_num, replace=False)
            
        ###############mask_inference#########
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor,row)
        orig_target_sizes = torch.stack([target["orig_size"]], dim=0).to(args.device)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): results[0]}
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser('YOLOS inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.dataset_file == 'coco':
        if args.backbone_name == 'base':
            init_pe_size = [800,1344]
            mid_pe_size = [800,1344]
            resume = 'yolos_base_raw.pth'
        elif args.backbone_name == 'small':
            init_pe_size = [512, 864]
            mid_pe_size = [512, 864]
            resume = 'yolos_s_300_pre.pth'
        elif args.backbone_name == 'tiny':
            init_pe_size = [800, 1333]
            mid_pe_size = None
            resume = 'yolos_ti_raw.pth'
        else:
            raise('backbone_name not supported')
    elif args.dataset_file == 'mot15':
        if args.backbone_name == 'base':
            init_pe_size = [800,1344]
            mid_pe_size = [800,1344]
            resume = 'results/MOT17Det_base/checkpoint.pth'
        elif args.backbone_name == 'tiny':
            init_pe_size = [800, 1333]
            mid_pe_size = None
            resume = 'results/MOT15Det_tiny/checkpoint0299.pth'
        else:
            raise('backbone_name not supported')
        

    main(args, init_pe_size, mid_pe_size, resume)