# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch,warnings,argparse,random,os,sys,cv2
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from PIL import Image
import numpy as np
from patch_reuse_tools import TRANSFORM_tiny, TRANSFORM_base, MOT_CLASSES, COLORS, extract_bboxes_from_coco, \
                                                                                    rescale_bboxes, resize_bbox, \
                                                                                    detections_to_coco_format
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
sys.path.append('/home/livion/Documents/github/fork/YOLOS/') 
import util.misc as utils
from models.detector import ReuseDetector
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 忽略UserWarning
warnings.simplefilter(action='ignore', category=UserWarning)

def get_args_parser():
    parser = argparse.ArgumentParser('Set YOLOS', add_help=False)
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--use_checkpoint', action='store_true',
                        help='use checkpoint.checkpoint to save mem')
    parser.add_argument('--num_class', default=1, type=int,
                        help='num of classes in the dataset')
    parser.add_argument('--drop_proportion', default=0.1, type=float,
                        help='the proportion of the patch to drop')
    parser.add_argument('--dataset_file', default='mot15', type=str,
                        help='the dataset to train on')
    parser.add_argument('--vals_folder', default='/home/livion/Documents/github/dataset/MOT15_coco/val', type=str,
                        help='the folder of the validation set')
    parser.add_argument('--vals_json', default='/home/livion/Documents/github/dataset/MOT15_coco/annotations/MOT15_instances_vals.json', type=str,
                        help='the json file of the validation set')

    # * model setting
    parser.add_argument("--det_token_num", default=100, type=int,
                        help="Number of det token in the deit backbone")
    parser.add_argument("--max_reuse_frame", default=0, type=int,
                        help="reuse frame interval")
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
    parser.add_argument('--resume', default='', help='resume from checkpoint')


    return parser

def visualize_attention(fname, bbox_scaled_c, attention_map, color):
    im = Image.open(fname)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 9), dpi=300)  # Two subplots side by side

    # Left subplot: Original image with bbox
    ax1.imshow(im)
    xmin, ymin, xmax, ymax, p = bbox_scaled_c
    color = [x / 255.0 for x in color]
    rect1 = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor=color, facecolor=None, fill=False)
    ax1.add_patch(rect1)
    text1 = f'{MOT_CLASSES[0]}: {p:0.2f}'
    ax1.text(xmin, ymin, text1, bbox=dict(facecolor=color, alpha=0.6), fontsize=12, color='white', va="bottom")

    # Right subplot: Attention map with bbox
    im_display = ax2.imshow(attention_map)
    rect2 = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor=color, facecolor=None, fill=False)
    ax2.add_patch(rect2)
    text2 = f'{MOT_CLASSES[0]}: {p:0.2f}'
    ax2.text(xmin, ymin, text2, bbox=dict(facecolor=color, alpha=0.6), fontsize=12, color='white', va="bottom")
    ax2.axis('off')
    
    # Add colorbar for the heatmap
    fig.colorbar(im_display, ax=ax2,fraction=0.026, pad=0.04)

    plt.tight_layout()
    plt.savefig('results/attention_visualize.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

def visualize_self_attention(fname, bbox_scaled_c, attention_map, color):
    pass

def get_one_query_meanattn(vis_attn,h_featmap,w_featmap):
    mean_attentions = vis_attn.mean(0).reshape(h_featmap, w_featmap)
    mean_attentions = nn.functional.interpolate(mean_attentions.unsqueeze(0).unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()
    return mean_attentions

def val_dataset_preporcess(args):
    image_paths = []
    reuse_image_paths = []
    files = sorted(os.listdir(args.vals_folder))
    for file in files:
        file_part = file.split('-')
        file_id = file_part[-1][0:-4]
        reuse_frame_id = int(file_id) - args.max_reuse_frame
        reuse_file_name = file[:-7] + str(reuse_frame_id) + '.jpg'
        if os.path.exists(os.path.join(args.vals_folder, reuse_file_name)):
            image_paths.append(os.path.join(args.vals_folder, file))
            reuse_image_paths.append(os.path.join(args.vals_folder, reuse_file_name))
        else:
            continue

    return image_paths, reuse_image_paths

def get_intersection_area(att_map, bbox, threshold):
    # 获取边界框区域内的注意力图
    x1, y1, x2, y2, _ = bbox
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    att_map_roi = att_map[y1:y2, x1:x2]

    # 二值化注意力图
    binary_att_map = (att_map_roi > threshold)
    # 计算大于阈值的区域与边界框的交集面积
    intersection_area = binary_att_map.sum().item()
    original_area = att_map_roi.shape[0] * att_map_roi.shape[1]
    intersection_proportion = round(intersection_area / original_area,4)
    return intersection_area, intersection_proportion

def attention_inference(model, image_path : str, args):
    img = Image.open(image_path)
    TRANSFORM = TRANSFORM_tiny if args.backbone_name == 'tiny' else TRANSFORM_base
    reference_tensor = TRANSFORM(img).unsqueeze(0)  # tensor数据格式是torch(C,H,W)

    # 将参考的图像的patch tokenized
    image_name = image_path.split('/')[-1]
    bboxes,reference_image_id = extract_bboxes_from_coco(args.vals_json, image_name) 
    ground_truth_bbox = [resize_bbox(bbox, img.size, reference_tensor.shape[-2:]) for bbox in bboxes]
    ###############reference_inference#########
    with torch.no_grad():
        reuse_embedding = None
        reuse_region = None
        reference_tensor = reference_tensor.to(args.device)
        outputs_reference,saved_embedding,_ = model(reference_tensor,reuse_embedding,reuse_region,args.drop_proportion)
        attention = model.forward_return_attention(reference_tensor)
        attention = attention[-1].detach().cpu()
        nh = attention.shape[1] # number of head
        attention = attention[0, :, -args.det_token_num:, 1:-args.det_token_num]
    probas_reference = outputs_reference['pred_logits'].softmax(-1)[0, :, :-1].to('cpu')
    keep_reference = probas_reference.max(-1).values > 0.9
    # convert boxes from [0; 1] to image scales
    bboxes_scaled_reference = rescale_bboxes(outputs_reference['pred_boxes'][0, keep_reference], img.size)
    bboxes_scale_reference_add_confidence = torch.cat((bboxes_scaled_reference, probas_reference[keep_reference]), dim=1).tolist()
    return reference_tensor,outputs_reference,bboxes_scale_reference_add_confidence, reference_image_id, attention, ground_truth_bbox

def correlation(image,image_path,output,ground_truth_bbox,predict_bbox_c,attention):
    w_featmap = image.shape[3] // 16
    h_featmap = image.shape[2] // 16
    probas = output['pred_logits'].softmax(-1)[0, :, :-1].cpu()
    keep = probas.max(-1).values > 0.9
    scaled_bboxes = rescale_bboxes(output['pred_boxes'][0, keep], [image.shape[3], image.shape[2]])
    scaled_bboxes_c = torch.cat((scaled_bboxes, probas[keep]), dim=1).tolist()
    #预测中置信度较高的det-token
    vis_indexs = torch.nonzero(keep).squeeze(1)
    for ind,vis_index in enumerate(vis_indexs):
        vis_attn = attention[:, vis_index, :]
        mean_attention = get_one_query_meanattn(vis_attn, h_featmap, w_featmap)
        mean_attention = mean_attention[0]
        intersection_area,intersection_proportion = get_intersection_area(mean_attention, scaled_bboxes_c[ind], threshold=0.0004)
        visualize_attention(image_path, predict_bbox_c[ind], attention_map=mean_attention, color=[0,0,255])

    return

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

    if args.dataset_file == 'mot15':
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
    else:
        raise('dataset_file not supported')
        
    model = ReuseDetector(
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
    model.eval()

    image_paths, _ = val_dataset_preporcess(args)
    reference_prediction = []
    reference_id_pool = []
    for image in tqdm(image_paths):
        # reference_tensor,outputs_reference,bboxes_scale_reference_add_confidence, reference_image_id, attention, ground_truth_bbox
        image_tensor,outputs_reference,reference_bbox_c,reference_id, attention, ground_truth_bbox = attention_inference(model,image,args)
        # correlation(image=image_tensor,\
        #             image_path = image,\
        #             output=outputs_reference,\
        #             ground_truth_bbox=ground_truth_bbox,\
        #             predict_bbox_c=reference_bbox_c,\
        #             attention=attention)

        reference_prediction.append(reference_bbox_c)
        reference_id_pool.append(reference_id)
        # print('image_id: ', reference_id,'inference time: ', round(end_time - start_time,4), 's')

    reference_result = detections_to_coco_format(reference_prediction, reference_id_pool)

    # 加载Ground Truth数据
    reference_coco_gt = COCO(args.vals_json)  # 替换为你的ground truth的COCO格式json文件的路径

    # 使用检测结果来加载COCO detections
    reference_coco_dt = reference_coco_gt.loadRes(reference_result['annotations'])

    reference_detected_image_ids = set([detection["image_id"] for detection in reference_result['annotations']])

    # 执行评估
    print("##########raw_inference_evaluation#########")
    raw_coco_eval = COCOeval(reference_coco_gt, reference_coco_dt, 'bbox')
    raw_coco_eval.params.imgIds = list(reference_detected_image_ids)  # 仅评估实际进行了预测的图像
    raw_coco_eval.evaluate()
    raw_coco_eval.accumulate()
    raw_coco_eval.summarize()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('YOLOS token reuse evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)        
    main(args)