# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch,warnings,argparse,random,os
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import util.misc as utils
from models.detector import TokenReorganizations,PostProcess
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
    parser.add_argument('--keep_rate', default=0.9, type=float,
                        help='the rate of the tokens to keep')
    parser.add_argument('--dataset_file', default='mot15', type=str,
                        help='the dataset to train on')
    parser.add_argument('--vals_folder', default='/home/livion/Documents/github/dataset/MOT15_coco/val', type=str,
                        help='the folder of the validation set')
    parser.add_argument('--coco_path', default='/home/livion/Documents/github/dataset/MOT15_coco', type=str)
    parser.add_argument('--eval_size', default=512, type=int,
                        help='the size of the validation set')

    # * model setting
    parser.add_argument("--det_token_num", default=100, type=int,
                        help="Number of det token in the deit backbone")
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
    parser.add_argument('--resume', default='results/MOT15Det_tiny/checkpoint0299.pth', help='resume from checkpoint')


    return parser

def get_attention(model,image_tensor,target,args):
    image_tensor = image_tensor.to(args.device)
    attention = model.forward_return_attention(image_tensor)
    attention = attention[-1].detach().cpu()
    nh = attention.shape[1] # number of head
    attention = attention[0, :, -args.det_token_num:, 1:-args.det_token_num]
    return 

def inference(model, image_tensor, target, args):
    image_tensor = image_tensor.to(args.device)
    output = model(image_tensor)
    return output


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
        elif args.backbone_name == 'tiny':
            init_pe_size = [800, 1333]
            mid_pe_size = None
        else:
            raise('backbone_name not supported')
    else:
        raise('dataset_file not supported')
        
    model = TokenReorganizations(
        num_classes=args.num_class, #类别数91
        pre_trained= args.pre_trained, #pre_train模型pth文件
        det_token_num=args.det_token_num, #100个det token
        backbone_name=args.backbone_name, #vit backbone的型号
        init_pe_size=init_pe_size, #初始化position embedding大小
        mid_pe_size=mid_pe_size, #mid position embedding 大小
        use_checkpoint=args.use_checkpoint, #是否使用checkpoint
        keep_rate=args.keep_rate #token保留率
    )
    model.to(device)
    checkpoint = torch.load(args.resume, map_location=args.device)
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

    reuse_proportion = []

    for image_tensor,target in tqdm(data_loader_val):
        # outputs, debug_data = get_attention(model,image_tensor,target,args)
        outputs = inference(model, image_tensor, target, args)
        orig_target_sizes = torch.stack([target[0]["orig_size"]], dim=0).to(args.device)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target[0]['image_id'].item(): results[0]}
        coco_evaluator.update(res)
        # reuse_proportion.append(debug_data['reuse_proportion'])

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # print('Reuse Proportion: ', np.mean(reuse_proportion))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('YOLOS token reuse evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)        
    main(args)