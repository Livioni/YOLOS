# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch,warnings,argparse,random,os
from time import time
from tqdm.contrib import tzip
from pathlib import Path
from PIL import Image
import numpy as np
from tools.patch_reuse_tools import TRANSFORM_tiny, TRANSFORM_base, MOT_CLASSES, COLORS, extract_bboxes_from_coco, rescale_bboxes, resize_bbox, detections_to_coco_format
import util.misc as utils
from models.detector import ReuseDetector
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 忽略UserWarning
warnings.simplefilter(action='ignore', category=UserWarning)

def get_args_parser():
    parser = argparse.ArgumentParser('Set YOLOS', add_help=False)

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

def token_reuse_inference(model, image_path : str, reuse_image_path : str, args):
    img = Image.open(image_path)
    reuse_image = Image.open(reuse_image_path)
    TRANSFORM = TRANSFORM_tiny if args.backbone_name == 'tiny' else TRANSFORM_base
    reference_tensor = TRANSFORM(reuse_image).unsqueeze(0)  # tensor数据格式是torch(C,H,W)
    input_tensor = TRANSFORM(img).unsqueeze(0)  # tensor数据格式是torch(C,H,W)

    # 将参考的图像的patch tokenized
    image_name = reuse_image_path.split('/')[-1]
    bboxes,reference_image_id = extract_bboxes_from_coco(args.vals_json, image_name) 
    ground_truth_bbox = [resize_bbox(bbox, reuse_image.size, reference_tensor.shape[-2:]) for bbox in bboxes]
    ###############reference_inference#########
    with torch.no_grad():
        reuse_embedding = None
        reuse_region = None
        reference_tensor = reference_tensor.to(args.device)
        outputs_reference,saved_embedding,_ = model(reference_tensor,reuse_embedding,reuse_region,args.drop_proportion)
        attention = model.forward_return_attention(reference_tensor)
    probas_reference = outputs_reference['pred_logits'].softmax(-1)[0, :, :-1].to('cpu')
    keep_reference = probas_reference.max(-1).values > 0.9
    # convert boxes from [0; 1] to image scales
    bboxes_scaled_reference = rescale_bboxes(outputs_reference['pred_boxes'][0, keep_reference], img.size)
    bboxes_scale_reference_add_confidence = torch.cat((bboxes_scaled_reference, probas_reference[keep_reference]), dim=1).tolist()
    new_bbox = bboxes_scaled_reference
    ################drop_inference#############
    with torch.no_grad():
        reuse_embedding = saved_embedding
        reuse_region = new_bbox
        input_tensor = input_tensor.to(args.device)
        outputs,_,debug_data = model(input_tensor,reuse_embedding,reuse_region,args.drop_proportion)
    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1].to('cpu')
    keep = probas.max(-1).values > 0.9
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], img.size)
    bboxes_scaled_add_confidence = torch.cat((bboxes_scaled, probas[keep]), dim=1).tolist()
    image_id = reference_image_id + args.max_reuse_frame
    return bboxes_scale_reference_add_confidence, bboxes_scaled_add_confidence, reference_image_id, image_id, debug_data

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

    image_paths, reuse_image_paths = val_dataset_preporcess(args)
    reference_prediction = []
    reuse_prediction = []
    reference_id_pool = []
    image_id_pool = []
    reuse_propotion = []
    for image, reuse_image in tzip(image_paths,reuse_image_paths):
        start_time = time()
        reference_bbox_c, reuse_bbox_c, reference_id, image_id, debug_data = token_reuse_inference(model,image,reuse_image,args)
        end_time = time()
        reference_prediction.append(reference_bbox_c)
        reuse_prediction.append(reuse_bbox_c)
        reference_id_pool.append(reference_id)
        image_id_pool.append(image_id)
        reuse_propotion.append(debug_data['reuse_proportion'])
        print('image_id: ', image_id,'inference time: ', round(end_time - start_time,4), 's')

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