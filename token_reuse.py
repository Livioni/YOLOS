
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
from models.detector import Detector
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
    parser.add_argument('--num_class', default=2, type=int,
                        help='num of classes in the dataset')
    parser.add_argument('--random_drop', default=False, action='store_true',
                        help='random_drop the patch in the image')
    parser.add_argument('--no_patch_drop', default=False, action='store_true',
                        help='drop the non ROIpatch in the image')
    parser.add_argument('--token_reuse', default=True, action='store_true',
                        help='whether to reuse the token in the image')
    parser.add_argument('--drop_porpotion', default=0.8, type=float,
                        help='the porpotion of the patch to drop')
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
    parser.add_argument('--device', default='cpu',
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

def compute_ap(predictions, ground_truth, iou_threshold=0.5):
    """
    计算Average Precision (AP)
    """
    ground_truth_copy = copy.deepcopy(ground_truth)
    predictions = sorted(predictions, key=lambda x: x[4], reverse=True)  # 根据置信度排序

    tp = 0
    fp = 0
    fn = len(ground_truth_copy)
    
    precisions = []
    recalls = []

    for pred in predictions:
        bbox_pred = pred[:4]
        best_iou = -1
        best_gt_idx = -1
        for idx, bbox_gt in enumerate(ground_truth_copy):
            current_iou = iou(bbox_pred, bbox_gt)
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_idx = idx

        if best_iou > iou_threshold:
            tp += 1
            fn -= 1
            ground_truth_copy.pop(best_gt_idx)
        else:
            fp += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        precisions.append(precision)
        recalls.append(recall)

    # 插值Precision-Recall曲线
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    for i in range(precisions.size - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i+1])

    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])

    return ap

def compute_mAP(predictions, ground_truth):
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []
    for iou_threshold in iou_thresholds:
        ap = compute_ap(predictions, ground_truth, iou_threshold)
        aps.append(ap)

    return np.mean(aps)

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

# plot_results(img, probas[keep],probas_raw[keep_raw], bboxes_scaled,bboxes_scaled_raw, row)
def plot_results(pil_img, prob,prob_raw, boxes,boxes_raw,row,backbone='base'):
    fig, ax = plt.subplots(2, 2, figsize=(30, 20))
    ax[0,0].imshow(pil_img)
    ax[1,0].imshow(pil_img)
    colors = COLORS * 100
    #左下的图是用原图推理的得到的结果
    for p, (xmin, ymin, xmax, ymax), c in zip(prob_raw, boxes_raw.tolist(), colors):
        ax[1,0].add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{MOT_CLASSES[cl]}: {p[cl]:0.2f}'
        ax[1,0].text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))

    ax[1,1].imshow(pil_img)
    colors = COLORS * 100
    #右下的图是用drop patch后的图推理的得到的结果
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax[1,1].add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{MOT_CLASSES[cl]}: {p[cl]:0.2f}'
        ax[1,1].text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    img_copy = plot_masked(pil_img,row,backbone=backbone)
    ax[0,1].imshow(img_copy)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/'+'result.png')
    # plt.show()

def plot_results_reuse(pil_img, prob,prob_raw, boxes,boxes_raw,row, reuse,backbone='base'):
    fig, ax = plt.subplots(2, 2, figsize=(30, 20))
    ax[0,0].imshow(pil_img)
    ax[1,0].imshow(pil_img)
    colors = COLORS * 100
    #左下的图是用原图推理的得到的结果
    for p, (xmin, ymin, xmax, ymax), c in zip(prob_raw, boxes_raw.tolist(), colors):
        ax[1,0].add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{MOT_CLASSES[cl]}: {p[cl]:0.2f}'
        ax[1,0].text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))

    ax[1,1].imshow(pil_img)
    colors = COLORS * 100
    #右下的图是用drop patch后的图推理的得到的结果
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax[1,1].add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{MOT_CLASSES[cl]}: {p[cl]:0.2f}'
        ax[1,1].text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    img_copy = plot_masked_reuse(pil_img,row,reuse_image=reuse,backbone=backbone)
    ax[0,1].imshow(img_copy)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/'+'result.png')
    # plt.show()

def plot_masked_reuse(img,row,reuse_image,backbone='tiny'):
    original_size = img.size
    if backbone == 'tiny':
        transformed_img = TRANSFORM_tiny.transforms[0](img)
        transformed_img = TRANSFORM_tiny.transforms[1](transformed_img)
        transformed_img_reuse = TRANSFORM_tiny.transforms[0](reuse_image)
        transformed_img_reuse = TRANSFORM_tiny.transforms[1](transformed_img_reuse)

    else:
        transformed_img = TRANSFORM_base.transforms[0](img)
        transformed_img = TRANSFORM_base.transforms[1](transformed_img)
        transformed_img_reuse = TRANSFORM_base.transforms[0](reuse_image)
        transformed_img_reuse = TRANSFORM_base.transforms[1](transformed_img_reuse)

    patch_dim_w, patch_dim_h = transformed_img.shape[2] // 16, transformed_img.shape[1] // 16
    img_copy = rearrange(transformed_img, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=16, p2=16)
    img_copy[row, :] = 0.0
    img_copy = rearrange(img_copy, '(h w) (p1 p2 c) -> c (h p1) (w p2)',p1=16, p2=16,h=patch_dim_h,w=patch_dim_w)

    ref_tensor_patches = rearrange(transformed_img_reuse, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=16, p2=16)
            
    # 使用row中的索引从ref_tensor_patches中选取patches
    replacement_patches = ref_tensor_patches[row, :]

    # 将input_tensor重新整形为patch形式
    input_tensor_patches = rearrange(transformed_img, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=16, p2=16)
    
    # 使用replacement_patches替换input_tensor_patches中的patches
    input_tensor_patches[row, :] = replacement_patches

    # 将input_tensor_patches重新整形回原始尺寸
    img_copy = rearrange(input_tensor_patches, '(h w) (p1 p2 c) -> c (h p1) (w p2)', p1=16, p2=16, h=patch_dim_h, w=patch_dim_w)

    resize = transforms.Resize((original_size[1],original_size[0]))
    img_copy = resize(img_copy)
    #tensor to PIL
    img_copy = transforms.ToPILImage()(img_copy)
    return img_copy

def plot_masked(img,row,backbone='tiny'):
    original_size = img.size
    if backbone == 'tiny':
        transformed_img = TRANSFORM_tiny.transforms[0](img)
        transformed_img = TRANSFORM_tiny.transforms[1](transformed_img)
    else:
        transformed_img = TRANSFORM_base.transforms[0](img)
        transformed_img = TRANSFORM_base.transforms[1](transformed_img)
    patch_dim_w, patch_dim_h = transformed_img.shape[2] // 16, transformed_img.shape[1] // 16
    img_copy = rearrange(transformed_img, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=16, p2=16)
    img_copy[row, :] = 0.0
    img_copy = rearrange(img_copy, '(h w) (p1 p2 c) -> c (h p1) (w p2)',p1=16, p2=16,h=patch_dim_h,w=patch_dim_w)
    resize = transforms.Resize((original_size[1],original_size[0]))
    img_copy = resize(img_copy)
    #tensor to PIL
    img_copy = transforms.ToPILImage()(img_copy)
    return img_copy

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

    model = Detector(
        num_classes=args.num_class, #类别数91
        pre_trained= args.pre_trained, #pre_train模型pth文件
        det_token_num=args.det_token_num, #100个det token
        backbone_name=args.backbone_name, #vit backbone的型号
        init_pe_size=init_pe_size, #初始化position embedding大小
        mid_pe_size=mid_pe_size, #mid position embedding 大小
        use_checkpoint=args.use_checkpoint, #是否使用checkpoint
    )
    model.to(device)
    checkpoint = torch.load(resume, map_location='cpu')
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

        if args.random_drop:
            drop_num = int(patch_num * args.drop_porpotion)
            input_tensor = rearrange(input_tensor, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)
            row = np.random.choice(range(patch_num), size=drop_num, replace=False)
            input_tensor[:, row, :] = 0.0
            input_tensor = rearrange(input_tensor, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',p1=16, p2=16,h=patch_dim_h,w=patch_dim_w)


        if args.no_patch_drop:
            # 假设有一个bounding boxes的列表，其中每个bounding box的格式是：bbox = [x1, y1, x2, y2]
            drop_num = int(len(all_indices) * args.drop_porpotion)
            # 从除所有bounding boxes外的patches中随机选择要drop的patches
            row = np.random.choice(list(all_indices), size=drop_num, replace=False)
            
            input_tensor = rearrange(input_tensor, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)
            input_tensor[:, row, :] = 0.0
            input_tensor = rearrange(input_tensor, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=16, p2=16, h=patch_dim_h, w=patch_dim_w)


        if args.token_reuse:
            drop_num = int(len(all_indices) * args.drop_porpotion)
            # 从除所有bounding boxes外的patches中随机选择要drop的patches
            row = np.random.choice(list(all_indices), size=drop_num, replace=False)
            
            input_tensor = rearrange(input_tensor, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)
            input_tensor[:, row, :] = 0.0
            input_tensor = rearrange(input_tensor, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=16, p2=16, h=patch_dim_h, w=patch_dim_w)
            # 假设你有一个ref_tensor与input_tensor尺寸相同
            reuse_image = Image.open(args.reuse_image)

            ref_tensor = TRANSFORM(reuse_image).unsqueeze(0)  # tensor数据格式是torch(C,H,W)

            # 将ref_tensor重新整形为patch形式
            ref_tensor_patches = rearrange(ref_tensor, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)
            
            # 使用row中的索引从ref_tensor_patches中选取patches
            replacement_patches = ref_tensor_patches[:, row, :]

            # 将input_tensor重新整形为patch形式
            input_tensor_patches = rearrange(input_tensor, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)
            
            # 使用replacement_patches替换input_tensor_patches中的patches
            input_tensor_patches[:, row, :] = replacement_patches

            # 将input_tensor_patches重新整形回原始尺寸
            input_tensor = rearrange(input_tensor_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=16, p2=16, h=patch_dim_h, w=patch_dim_w)

        ###############raw_inference#########
        with torch.no_grad():
            original_img = original_img.to(device)
            outputs_raw = model(original_img)
        probas_raw = outputs_raw['pred_logits'].softmax(-1)[0, :, :-2]
        keep_raw = probas_raw.max(-1).values > 0.9
        # convert boxes from [0; 1] to image scales
        bboxes_scaled_raw = rescale_bboxes(outputs_raw['pred_boxes'][0, keep_raw], img.size)

        start_time = time()
        ###############drop_inference#########
        with torch.no_grad():
            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor)
        end_time = time()
        print(f'Inference Time:{end_time-start_time:.3f}s.')

        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-2]
        keep = probas.max(-1).values > 0.9
        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], img.size)

        if args.output_dir != '':
            if args.token_reuse:
                plot_results_reuse(img, probas[keep],probas_raw[keep_raw], bboxes_scaled, bboxes_scaled_raw, row, reuse_image, backbone=args.backbone_name)
            else:
                plot_results(img, probas[keep],probas_raw[keep_raw], bboxes_scaled, bboxes_scaled_raw, row, backbone=args.backbone_name)
        
            
        # 加载Ground Truth数据
        bboxes_scaled_add_confidence = torch.cat((bboxes_scaled, probas[keep]), dim=1)
        bboxes_scale_raw_add_confidence = torch.cat((bboxes_scaled_raw, probas_raw[keep_raw]), dim=1).tolist()
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

        # print('AP@.5 = ',compute_ap(bboxes_scaled_add_confidence, reference_bboxes, 0.5))
        # print('mAP@.50:.05:.95 = ',compute_mAP(bboxes_scaled_add_confidence, reference_bboxes))

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