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

def create_subplots(n):
    fig = plt.figure(figsize=(10, 5))  # Adjust the figure size as needed
    rows = n // 2
    rows = 1 if rows < 1 else rows
    gs = fig.add_gridspec(rows, 4)
    
    axs = []
    for i in range(rows):
        axs.append(fig.add_subplot(gs[i, 0]))  # First column
        if len(axs) < n:  # To avoid creating extra subplots if n is odd
            axs.append(fig.add_subplot(gs[i, -1]))  # Last column
            
    return fig, axs, gs

def get_one_query_meanattn(vis_attn,h_featmap,w_featmap):
    mean_attentions = vis_attn.mean(0).reshape(h_featmap, w_featmap)
    mean_attentions = nn.functional.interpolate(mean_attentions.unsqueeze(0).unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()
    return mean_attentions

def get_one_query_sumattn(vis_attn,h_featmap,w_featmap):
    mean_attentions = vis_attn.sum(0).reshape(h_featmap, w_featmap)
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
    reference_tensor = reference_tensor.to(args.device)
    attention = model.forward_return_attention(reference_tensor)
    attention = attention[-1].detach().cpu()
    nh = attention.shape[1] # number of head
    attention = attention[0, :, -args.det_token_num:, 1:-args.det_token_num]
    #forward input to get pred
    reuse_embedding = None
    reuse_region = None
    result_dic, _, _ = model(reference_tensor,reuse_embedding,reuse_region,args.drop_proportion)
    # result_dic = model(reference_tensor)
    probas = result_dic['pred_logits'].softmax(-1)[0, :, :-1].cpu()
    keep = probas.max(-1).values > 0.9
    sacled_bbox = rescale_bboxes(result_dic['pred_boxes'][0, keep],img.size)
    bbox_scaled_c = torch.cat((sacled_bbox, probas[keep]), dim=1).tolist()
    transform_bbox = rescale_bboxes(result_dic['pred_boxes'][0, keep],[reference_tensor.shape[-1],reference_tensor.shape[-2]])
    transform_bbox_c = torch.cat((transform_bbox, probas[keep]), dim=1).tolist()
    vis_indexs = torch.nonzero(keep).squeeze(1)
    # save token image
    h, w = reference_tensor.shape[2:]
    w_featmap = reference_tensor.shape[3] // args.patch_size
    h_featmap = reference_tensor.shape[2] // args.patch_size
    fig, axs, gs = create_subplots(len(vis_indexs))
    idxs = []
    raw_idxs = []
    for bbox in transform_bbox_c:
        xmin, ymin, xmax, ymax, p = bbox
        y_center = int((xmin + xmax) / 2)
        x_center = int((ymin + ymax) / 2)
        idxs.append((x_center, y_center))
        
    for bbox in bbox_scaled_c :
        xmin, ymin, xmax, ymax, p = bbox
        y_center = int((xmin + xmax) / 2)
        x_center = int((ymin + ymax) / 2)
        raw_idxs.append((x_center, y_center))


    fact = 16
    for ind,[idx_o, ax] in enumerate(zip(idxs, axs)):
        idx = (idx_o[0], idx_o[1])
        vis_index = vis_indexs[ind]
        vis_attn = attention[:, vis_index, :]
        mean_attention = get_one_query_sumattn(vis_attn, h_featmap, w_featmap)
        mean_attention = mean_attention[0]
        ax.imshow(mean_attention, cmap='cividis', interpolation='nearest')
        x1 = transform_bbox_c[ind][0]
        y1 = transform_bbox_c[ind][1]
        x2 = transform_bbox_c[ind][2]
        y2 = transform_bbox_c[ind][3]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none'))
        ax.plot(idx[1], idx[0], 'ro', markersize=2)
        ax.axis('off')
        ax.set_title(f'det_token#n{vis_index}-attention')


    fcenter_ax = fig.add_subplot(gs[:, 1:-1])
    fcenter_ax.imshow(img)
    for (y, x) in raw_idxs:
        fcenter_ax.add_patch(plt.Circle((x,y), fact // 2, color='r'))
        fcenter_ax.axis('off')
    
    for bbox in bbox_scaled_c: 
        xmin, ymin, xmax, ymax, p = bbox
        fcenter_ax.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none'))

    plt.tight_layout()
    save_image_name = image_path.split('/')[-1]
    plt.savefig('results/MOT15_val_mean_det_token/' + save_image_name, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    return bbox_scaled_c,reference_image_id

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
        reference_bbox_c, reference_id= attention_inference(model,image,args)
        
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