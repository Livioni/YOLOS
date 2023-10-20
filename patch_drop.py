
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse,copy
from time import time
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import torch
import torchvision.transforms as transforms
import util.misc as utils
from models.detector import DropDetector
from einops import rearrange
from util.video_preprocess import file_type
import warnings
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 忽略UserWarning
warnings.simplefilter(action='ignore', category=UserWarning)

# COCO classes
COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

MOT_CLASSES = ['person']

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

TRANSFORM_tiny = transforms.Compose([
            transforms.Resize((512,688)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

TRANSFORM_base = transforms.Compose([
            transforms.Resize((608,896)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

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


    # * model setting
    parser.add_argument("--det_token_num", default=100, type=int,
                        help="Number of det token in the deit backbone")
    parser.add_argument('--backbone_name', default='tiny', type=str,
                        help="Name of the deit backbone to use")
    parser.add_argument('--pre_trained', default='',
                        help="set imagenet pretrained model path if not train yolos from scatch")

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

def iou(box1, box2):
    """计算IoU"""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    inter_x1 = max(x1, x1g)
    inter_y1 = max(y1, y1g)
    inter_x2 = min(x2, x2g)
    inter_y2 = min(y2, y2g)

    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

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

def plot_results(image,prob,bboxes):
    fig, ax = plt.subplots(1, figsize=(16, 8))
    ax.imshow(image)
    colors = COLORS * 100
    #左的图是用原图推理的得到的结果
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, bboxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{MOT_CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/'+'result.png')


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

    model = DropDetector(
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
    model.load_state_dict(checkpoint['model'], strict=False)

    if file_type(args.input) == 'image':
        img = Image.open(args.input)
        TRANSFORM = TRANSFORM_tiny if args.backbone_name == 'tiny' else TRANSFORM_base
        input_tensor = TRANSFORM(img).unsqueeze(0)  # tensor数据格式是torch(C,H,W)
        original_img = copy.deepcopy(input_tensor)
        patch_dim_w, patch_dim_h = input_tensor.shape[3] // 16, input_tensor.shape[2] // 16
        patch_num = patch_dim_h * patch_dim_w

        bboxes = extract_bboxes_from_coco('/home/livion/Documents/github/dataset/MOT15_coco/annotations/MOT15_instances_vals.json', 'ADL-Rundle-8-000498.jpg')   
            
        all_indices = set(range(patch_num))
        reference_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            reference_bboxes.append([x1, y1, x2, y2])
            original_width, original_height = img.size
            new_width, new_height = input_tensor.shape[3], input_tensor.shape[2]
            x1 = x1 * (new_width / original_width)
            y1 = y1 * (new_height / original_height)
            x2 = x2 * (new_width / original_width)
            y2 = y2 * (new_height / original_height)
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
            
        start_time = time()
        ###############mask_inference#########
        with torch.no_grad():
            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor,row)
        end_time = time()
        print(f'Inference Time:{end_time-start_time:.3f}s.')

        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1].to('cpu')
        keep = probas.max(-1).values > 0.9
        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], img.size)

        if args.output_dir != '':
            if args.token_drop:
                plot_results(img, probas[keep], bboxes_scaled)

        # 加载Ground Truth数据
        bboxes_scaled_add_confidence = torch.cat((bboxes_scaled, probas[keep]), dim=1)
        # 转换检测结果到COCO格式
        image_id = 139  # 根据你的数据设置
        coco_detections = [
            {
                "image_id": image_id, 
                "category_id": 0,  
                "bbox": [x1, y1, x2-x1, y2-y1], 
                "score": confidence
            } 
            for [x1, y1, x2, y2, confidence] in bboxes_scaled_add_confidence
        ]

        # 加载Ground Truth数据
        coco_gt = COCO("/home/livion/Documents/github/dataset/MOT15_coco/annotations/MOT15_instances_vals.json")  # 替换为你的ground truth的COCO格式json文件的路径

        # 使用检测结果来加载COCO detections
        coco_dt = coco_gt.loadRes(coco_detections)

        # 执行评估
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = [image_id]  # 设定要评估的图片ID
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

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