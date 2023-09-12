# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
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

# COCO classes
CLASSES = [
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

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def get_args_parser():
    parser = argparse.ArgumentParser('Set YOLOS', add_help=False)

    parser.add_argument('--use_checkpoint', action='store_true',
                        help='use checkpoint.checkpoint to save mem')

    # * model setting
    parser.add_argument("--det_token_num", default=100, type=int,
                        help="Number of det token in the deit backbone")
    parser.add_argument('--backbone_name', default='base', type=str,
                        help="Name of the deit backbone to use")
    parser.add_argument('--pre_trained', default='',
                        help="set imagenet pretrained model path if not train yolos from scatch")
    parser.add_argument('--init_pe_size', default=[800,1344], nargs='+', type=int,
                        help="init pe size (h,w)")
    parser.add_argument('--mid_pe_size', default=[800,1344], nargs='+', type=int,
                        help="mid pe size (h,w)")

    # dataset parameters
    parser.add_argument('--output_dir', default='results',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='yolos_base.pth', help='resume from checkpoint')
    parser.add_argument('--input',default='images/IMG_0040.jpeg',type=str)

    return parser

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

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/'+'result.png')
    # plt.show()

def main(args):
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
        num_classes=91, #类别数91
        pre_trained=args.pre_trained, #pre_train模型pth文件
        det_token_num=args.det_token_num, #100个det token
        backbone_name=args.backbone_name, #vit backbone的型号
        init_pe_size=args.init_pe_size, #初始化position embedding大小
        mid_pe_size=args.mid_pe_size, #mid position embedding 大小
        use_checkpoint=args.use_checkpoint, #是否使用checkpoint
    )
    model.to(device)
    model.eval()
    checkpoint = torch.load(args.resume, map_location='cuda')
    model.load_state_dict(checkpoint['model'])
    img = Image.open(args.input)
    transform = transforms.Compose([
                transforms.Resize(800),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    input_tensor = transform(img).unsqueeze(0)  # tensor数据格式是torch(C,H,W)
    start_time = time()
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        outputs = model(input_tensor)
    end_time = time()
    print(f'Inference Time:{end_time-start_time:.3f}s.')
    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], img.size)
    if args.output_dir != '':
        plot_results(img, probas[keep], bboxes_scaled)
    return    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('YOLOS inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)