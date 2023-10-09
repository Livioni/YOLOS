# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch,warnings,argparse,copy,random,os,json,tempfile
from time import time
from pathlib import Path
from PIL import Image
import numpy as np
from tools.token_reuse_tools import TRANSFORM_tiny, TRANSFORM_base, MOT_CLASSES, COLORS, extract_bboxes_from_coco, rescale_bboxes, detections_to_cocojson, detections_to_coco_format
import util.misc as utils
from models.detector import Detector
from einops import rearrange
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 忽略UserWarning
warnings.simplefilter(action='ignore', category=UserWarning)

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
    parser.add_argument('--drop_porpotion', default=0.0, type=float,
                        help='the porpotion of the patch to drop')
    parser.add_argument('--dataset_file', default='mot15', type=str,
                        help='the dataset to train on')
    parser.add_argument('--vals_folder', default='/home/livion/Documents/github/dataset/MOT15_coco/val', type=str,
                        help='the folder of the validation set')
    parser.add_argument('--vals_json', default='/home/livion/Documents/github/dataset/MOT15_coco/annotations/MOT15_instances_vals.json', type=str,
                        help='the json file of the validation set')

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
    parser.add_argument('--resume', default='', help='resume from checkpoint')


    return parser

def val_dataset_preporcess(args, max_reuse_frame):
    image_paths = []
    reuse_image_paths = []
    files = sorted(os.listdir(args.vals_folder))
    for file in files:
        file_part = file.split('-')
        file_id = file_part[-1][0:-4]
        reuse_frame_id = int(file_id) - max_reuse_frame
        reuse_file_name = file[:-7] + str(reuse_frame_id) + '.jpg'
        if os.path.exists(os.path.join(args.vals_folder, reuse_file_name)):
            image_paths.append(os.path.join(args.vals_folder, file))
            reuse_image_paths.append(os.path.join(args.vals_folder, reuse_file_name))
        else:
            continue

    return image_paths, reuse_image_paths

def token_reuse_inference(model, image_path : str, reuse_image_path : str, device : str, args):
    img = Image.open(image_path)
    TRANSFORM = TRANSFORM_tiny if args.backbone_name == 'tiny' else TRANSFORM_base
    input_tensor = TRANSFORM(img).unsqueeze(0)  # tensor数据格式是torch(C,H,W)
    original_img = copy.deepcopy(input_tensor)
    patch_dim_w, patch_dim_h = input_tensor.shape[3] // 16, input_tensor.shape[2] // 16
    patch_num = patch_dim_h * patch_dim_w

    image_name = image_path.split('/')[-1]
    bboxes,image_id = extract_bboxes_from_coco(args.vals_json, image_name)   
    all_indices = set(range(patch_num))
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
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

    drop_num = int(len(all_indices) * args.drop_porpotion)
    # 从除所有bounding boxes外的patches中随机选择要drop的patches
    row = np.random.choice(list(all_indices), size=drop_num, replace=False)
    # 假设你有一个ref_tensor与input_tensor尺寸相同
    reuse_image = Image.open(reuse_image_path)
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

    ###############drop_inference#########
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        outputs = model(input_tensor)
    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-2]
    keep = probas.max(-1).values > 0.9
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], img.size)

    bboxes_scaled_add_confidence = torch.cat((bboxes_scaled, probas[keep]), dim=1).tolist()
    bboxes_scale_raw_add_confidence = torch.cat((bboxes_scaled_raw, probas_raw[keep_raw]), dim=1).tolist()

    return bboxes_scale_raw_add_confidence, bboxes_scaled_add_confidence,image_id

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
    model.eval()

    image_paths, reuse_image_paths = val_dataset_preporcess(args, 5)
    raw_prediction = []
    reuse_prediction = []
    image_id_pool = []
    for image, reuse_image in zip(image_paths,reuse_image_paths):
        start_time = time()
        raw_bbox_c, reuse_bbox_c, image_id = token_reuse_inference(model,image,reuse_image,device,args)
        end_time = time()
        raw_prediction.append(raw_bbox_c)
        reuse_prediction.append(reuse_bbox_c)
        image_id_pool.append(image_id)
        print('image_id: ', image_id,'inference time: ', end_time - start_time, 's')

    raw_result = detections_to_coco_format(raw_prediction, image_id_pool)
    reuse_result = detections_to_coco_format(reuse_prediction, image_id_pool)

     # 加载Ground Truth数据
    raw_coco_gt = COCO(args.vals_json)  # 替换为你的ground truth的COCO格式json文件的路径
    reuse_coco_gt = COCO(args.vals_json)  # 替换为你的ground truth的COCO格式json文件的路径

    # 使用检测结果来加载COCO detections
    raw_coco_dt = raw_coco_gt.loadRes(raw_result['annotations'])
    reuse_coco_dt = reuse_coco_gt.loadRes(reuse_result['annotations'])

    detected_image_ids = set([detection["image_id"] for detection in raw_result['annotations']])

    # 执行评估
    print("##########raw_inference_evaluation#########")
    raw_coco_eval = COCOeval(raw_coco_gt, raw_coco_dt, 'bbox')
    raw_coco_eval.params.imgIds = list(detected_image_ids)  # 仅评估实际进行了预测的图像
    raw_coco_eval.evaluate()
    raw_coco_eval.accumulate()
    raw_coco_eval.summarize()

    # 执行评估
    print("##########token_reuse_inference_evaluation#########")
    reuse_coco_eval = COCOeval(reuse_coco_gt, reuse_coco_dt, 'bbox')
    reuse_coco_eval.params.imgIds = list(detected_image_ids)  # 仅评估实际进行了预测的图像
    reuse_coco_eval.evaluate()
    reuse_coco_eval.accumulate()
    reuse_coco_eval.summarize()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('YOLOS token reuse evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)        
    main(args)