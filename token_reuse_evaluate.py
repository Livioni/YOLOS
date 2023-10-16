# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch,warnings,argparse,random,os
from tqdm.contrib import tzip
from time import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image
import numpy as np
from tools.patch_reuse_tools import TRANSFORM_tiny, TRANSFORM_base, MOT_CLASSES, COLORS, extract_bboxes_from_coco, rescale_bboxes, resize_bbox, detections_to_coco_format
import util.misc as utils
from models.detector import ReuseDetector
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 忽略UserWarning
warnings.simplefilter(action='ignore', category=UserWarning)

TRANSFORM_tiny = transforms.Compose([
            transforms.Resize((432,768)),
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
    parser.add_argument('--drop_proportion', default=1.0, type=float,
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
    parser.add_argument("--max_reuse_frame", default=5, type=int,
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

def visualization(image_path:str,\
                  reuse_image_path:str,\
                  reuse_image_bbox:list,\
                  prediction_result:list,\
                  output_dir:str):
    fig, ax = plt.subplots(2, 2, figsize=(30, 20))
    #左上放reuse image
    reuse_image = Image.open(reuse_image_path)
    ax[0,0].imshow(reuse_image)
    ax[0,0].set_title('Reuse Image',fontsize=50)
    ax[0,0].axis('off')
    #左下放reuse image的bbox
    ax[1,0].imshow(reuse_image)
    for bbox in reuse_image_bbox:
        x1, y1, x2, y2, c = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax[1,0].add_patch(rect)
        text = 'Person'
        ax[1,0].text(x1, y1, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    ax[1,0].set_title('Reuse Image Prediction',fontsize=50)
    #右上放image (patch-reuse)
    image = Image.open(image_path)
    image_copy = image.copy()
    patch_dim_w, patch_dim_h = image.size[0] // 40, image.size[1] // 40
    patch_num = patch_dim_h * patch_dim_w
    all_indices = set(range(patch_num))
    keep_patch_indices = []
    for bbox in reuse_image_bbox:
        x1, y1, x2, y2, c = bbox
        # 计算与bounding box相关的patch的开始和结束索引
        start_row_idx = y1 // (16 * 2.5)
        start_col_idx = x1 // (16 * 2.5)
        end_row_idx = y2 // (16 * 2.5)
        end_col_idx = x2 // (16 * 2.5)
        
        # 从所有的patch中移除当前bounding box内的patch
        for i in range(int(start_row_idx), int(end_row_idx)+1):
            for j in range(int(start_col_idx), int(end_col_idx)+1):
                patch_idx = i * patch_dim_w + j
                all_indices.discard(patch_idx)
                keep_patch_indices.append(patch_idx)
    row = list(all_indices)
    transformed_img_reuse = TRANSFORM_tiny.transforms[1](reuse_image)
    transformed_img = TRANSFORM_tiny.transforms[1](image)
    reuse_image_patches = rearrange(transformed_img_reuse, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=40, p2=40)
    image_patches = rearrange(transformed_img, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=40, p2=40)
    replacement_patches = reuse_image_patches[row, :]
    image_patches[row, :] = replacement_patches
    image = rearrange(image_patches, '(h w) (p1 p2 c) -> c (h p1) (w p2)', p1=40, p2=40, h=patch_dim_h, w=patch_dim_w)
    image = transforms.ToPILImage()(image)
    ax[0,1].imshow(image)
    ax[0,1].set_title('Token Embedding Reuse Image',fontsize=50)
    ax[0,1].axis('off')
    #右下放image的bbox
    ax[1,1].imshow(image_copy)
    for bbox in prediction_result:
        x1, y1, x2, y2, c = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax[1,1].add_patch(rect)
        text = f'Person: {c:0.2f}'
        ax[1,1].text(x1, y1, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    ax[1,1].set_title('Token Embedding Reuse Image Prediction',fontsize=50)
    ax[1,1].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir+'/result.png')
    return

def token_reuse_inference(model, image_path : str, reuse_image_path : str, args):
    img = Image.open(image_path)
    reuse_image = Image.open(reuse_image_path)
    TRANSFORM = TRANSFORM_tiny if args.backbone_name == 'tiny' else TRANSFORM_base
    reference_tensor = TRANSFORM(reuse_image).unsqueeze(0)  # tensor数据格式是torch(C,H,W)
    input_tensor = TRANSFORM(img).unsqueeze(0)  # tensor数据格式是torch(C,H,W)

    # 将参考的图像的patch tokenized
    reuse_image_name = reuse_image_path.split('/')[-1]
    val_image_name = image_path.split('/')[-1]
    bboxes,reuse_image_id = extract_bboxes_from_coco(args.vals_json, reuse_image_name) 
    _, image_id = extract_bboxes_from_coco(args.vals_json, val_image_name)
    ground_truth_bbox = [resize_bbox(bbox, reuse_image.size, reference_tensor.shape[-2:]) for bbox in bboxes]
    ###############reference_inference#########
    with torch.no_grad():
        reuse_embedding = None
        reuse_region = None
        reference_tensor = reference_tensor.to(args.device)
        outputs_reference,saved_embedding,_ = model(reference_tensor,reuse_embedding,reuse_region,args.drop_proportion)
    probas_reference = outputs_reference['pred_logits'].softmax(-1)[0, :, :-1].to('cpu')
    keep_reference = probas_reference.max(-1).values > 0.9
    # convert boxes from [0; 1] to image scales
    bboxes_scaled_reference = rescale_bboxes(outputs_reference['pred_boxes'][0, keep_reference], img.size)
    bboxes_feed_back = rescale_bboxes(outputs_reference['pred_boxes'][0, keep_reference], [input_tensor.shape[-1],input_tensor.shape[-2]])
    bboxes_scale_reference_add_confidence = torch.cat((bboxes_scaled_reference, probas_reference[keep_reference]), dim=1).tolist()
    ################drop_inference#############
    with torch.no_grad():
        reuse_embedding = saved_embedding
        reuse_region = bboxes_feed_back
        input_tensor = input_tensor.to(args.device)
        outputs,_,debug_data = model(input_tensor,reuse_embedding,reuse_region,args.drop_proportion)
    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1].to('cpu')
    keep = probas.max(-1).values > 0.9
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], img.size)
    bboxes_scaled_add_confidence = torch.cat((bboxes_scaled, probas[keep]), dim=1).tolist()
    return bboxes_scale_reference_add_confidence, bboxes_scaled_add_confidence, reuse_image_id, image_id, debug_data

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

    # image_paths, reuse_image_paths = val_dataset_preporcess(args)
    image_paths = ['/home/livion/Documents/github/dataset/MOT15_coco/val/ADL-Rundle-8-000495.jpg']
    reuse_image_paths = ['/home/livion/Documents/github/dataset/MOT15_coco/val/ADL-Rundle-8-000491.jpg']
    reference_prediction = []
    reuse_prediction = []
    reference_id_pool = []
    image_id_pool = []
    reuse_propotion = []
    for image, reuse_image in tzip(image_paths,reuse_image_paths):
        reference_bbox_c, reuse_bbox_c, reference_id, image_id, debug_data = token_reuse_inference(model,image,reuse_image,args)
        ################visualization#############
        visualization(image_path=image,\
                  reuse_image_path=reuse_image,\
                  reuse_image_bbox=reference_bbox_c,\
                  prediction_result=reuse_bbox_c,\
                  output_dir=args.output_dir)
        reference_prediction.append(reference_bbox_c)
        reuse_prediction.append(reuse_bbox_c)
        reference_id_pool.append(reference_id)
        image_id_pool.append(image_id)
        reuse_propotion.append(debug_data['reuse_proportion'])
        # print('image_id: ', image_id,'inference time: ', round(end_time - start_time,4), 's')

    reference_result = detections_to_coco_format(reference_prediction, reference_id_pool)
    reuse_result = detections_to_coco_format(reuse_prediction, image_id_pool)

    # 加载Ground Truth数据
    reference_coco_gt = COCO(args.vals_json)  # 替换为你的ground truth的COCO格式json文件的路径
    reuse_coco_gt = COCO(args.vals_json)  # 替换为你的ground truth的COCO格式json文件的路径

    # 使用检测结果来加载COCO detections
    reference_coco_dt = reference_coco_gt.loadRes(reference_result['annotations'])
    reuse_coco_dt = reuse_coco_gt.loadRes(reuse_result['annotations'])

    reference_detected_image_ids = set([detection["image_id"] for detection in reference_result['annotations']])
    reuse_detected_image_ids = set([detection["image_id"] for detection in reuse_result['annotations']])

    # 执行评估
    print("##########raw_inference_evaluation#########")
    raw_coco_eval = COCOeval(reference_coco_gt, reference_coco_dt, 'bbox')
    raw_coco_eval.params.imgIds = list(reference_detected_image_ids)  # 仅评估实际进行了预测的图像
    raw_coco_eval.evaluate()
    raw_coco_eval.accumulate()
    raw_coco_eval.summarize()

    # 执行评估
    print("##########token_reuse_inference_evaluation#########")
    reuse_coco_eval = COCOeval(reuse_coco_gt, reuse_coco_dt, 'bbox')
    reuse_coco_eval.params.imgIds = list(reuse_detected_image_ids)  # 仅评估实际进行了预测的图像
    reuse_coco_eval.evaluate()
    reuse_coco_eval.accumulate()
    reuse_coco_eval.summarize()

    print('Reuse Proportion: ', np.mean(reuse_propotion))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('YOLOS token reuse evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)        
    main(args)