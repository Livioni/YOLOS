# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch,warnings,argparse,random,os,copy
from tqdm.contrib import tzip
from pathlib import Path
import numpy as np
from tools.patch_reuse_tools import rescale_bboxes,box_cxcywh_to_xyxy
import util.misc as utils
from einops import rearrange
from models.detector import PostProcess,Detector
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
    parser.add_argument('--drop_proportion', default=0.0, type=float,
                        help='the proportion of the patch to drop')
    parser.add_argument('--dataset_file', default='mot15', type=str,
                        help='the dataset to train on')
    parser.add_argument('--vals_folder', default='/home/livion/Documents/github/dataset/MOT15_coco/val', type=str,
                        help='the folder of the validation set')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--eval_size', default=512, type=int,
                        help='the size of the validation set')

    # * model setting
    parser.add_argument("--det_token_num", default=100, type=int,
                        help="Number of det token in the deit backbone")
    parser.add_argument("--max_reuse_frame", default=5, type=int,
                        help="reuse frame interval")
    parser.add_argument('--backbone_name', default='tiny', type=str,
                        help="Name of the deit backbone to use")
    parser.add_argument('--pre_trained', default='',
                        help="set imagenet pretrained model path if not train yolos from scratch")

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

def token_reuse_inference(model,image_tensor,target,\
                          reuse_image_tensor,target_reuse,args):
    image_tensor = image_tensor.unsqueeze(0)
    reuse_image_tensor = reuse_image_tensor.unsqueeze(0)
    original_img = copy.deepcopy(image_tensor)

    patch_dim_w, patch_dim_h = image_tensor.shape[3] // 16, image_tensor.shape[2] // 16
    patch_num = patch_dim_h * patch_dim_w
    bboxes = target['boxes']
    ground_truth_bboxes = rescale_bboxes(torch.tensor(bboxes), [image_tensor.shape[-1],image_tensor.shape[-2]]).tolist()
    all_indices = set(range(patch_num))
    if args.drop_proportion > 0:
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

        drop_num = int(len(all_indices) * args.drop_proportion)
        # 从除所有bounding boxes外的patches中随机选择要drop的patches
        row = np.random.choice(list(all_indices), size=drop_num, replace=False)
        # 将ref_tensor重新整形为patch形式
        ref_tensor_patches = rearrange(reuse_image_tensor, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)
        # 使用row中的索引从ref_tensor_patches中选取patches
        replacement_patches = ref_tensor_patches[:, row, :]
        # 将input_tensor重新整形为patch形式
        input_tensor_patches = rearrange(image_tensor, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)
        # 使用replacement_patches替换input_tensor_patches中的patches
        input_tensor_patches[:, row, :] = replacement_patches
        # 将input_tensor_patches重新整形回原始尺寸
        image_tensor = rearrange(input_tensor_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=16, p2=16, h=patch_dim_h, w=patch_dim_w)

    ###############raw_inference#########
    with torch.no_grad():
        original_img = original_img.to(args.device)
        outputs_raw = model(original_img)
    ###############drop_inference#########
    with torch.no_grad():
        image_tensor = image_tensor.to(args.device)
        outputs = model(image_tensor)
    return outputs,outputs_raw

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
    checkpoint = torch.load(resume, map_location=args.device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    ######build_dataset########
    dataset_val = build_dataset(image_set='val', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    base_ds = get_coco_api_from_dataset(dataset_val)
    postprocessors = {'bbox': PostProcess()}
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    image_paths, reuse_image_paths = val_dataset_preporcess(args)

    for image, reuse_image in tzip(image_paths,reuse_image_paths):
        files = sorted(os.listdir(args.vals_folder))
        index = files.index(image.split('/')[-1])
        image_tensor, target = dataset_val[index]
        reuse_index = files.index(reuse_image.split('/')[-1])
        reuse_image_tensor, target_reuse = dataset_val[reuse_index]
        outputs,outputs_raw = token_reuse_inference(model,image_tensor,target,\
                                                    reuse_image_tensor,target_reuse,args)
        orig_target_sizes = torch.stack([target["orig_size"]], dim=0).to(args.device)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): results[0]}
        coco_evaluator.update(res)


    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('YOLOS token reuse evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)        
    main(args)