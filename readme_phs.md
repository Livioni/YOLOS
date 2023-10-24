# Usage
## List
- **patch_reuse.py**: 使用（选择）base帧和当前帧，提取base帧的真值信息，当前帧reuse base frame 一定比例的patch。还包括patc_mask功能 随机mask和非ROI mask

- **patch_reuse_evaluate.py**: 同patch_reuse.py 测试整个测试集在reuse patch情况下准确度的变化。base frame 和current frame 设置为 current frame reuse X 帧之前的 base frame。
- **token_reuse_evaluate.py**: --max_reuse_frame 表示在X帧内，后X-1帧 reuse最开始一帧的patchs。测试整个测试集的精度变化。输出为
- **tools/det_token_attention_bbox.py**: 使用YOLOS文章中的技巧，可视化置信度大于阈值的det_tokens的self-attention热力图。
- **tools/detr_attenton_visualizer.py**: 使用detr文章的技巧，可视化encoder中最后一层的self-attention，并以bboxes框中心作为索引值（reference point）可视化当前的热力图。
- **patch_drop.py**: 直接丢弃（不是mask）一定比例的非ROI区域的patch，ROI由真值提供。


## Result 
### YOLOS Base
1. 使用raw YOLOS_B，results/MOT15Det_base/checkpoint0199.pth：
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.467
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.847
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.466
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.241
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.422
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.544
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.088
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.527
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.635
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.496
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.619
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.670

2. 使用 YOLOS_B, reuse_frame = 2, drop_proportion = 1.0\
        results/MOT15Det_base/checkpoint0199.pth，并且用上一帧检测的反馈



### YOLOS Tiny
1. 使用raw YOLOS_T，results/MOT15Det_tiny/checkpoint0299.pth：
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.433
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.803
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.415
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.236
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.387
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.516
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.090
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.480
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.612
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.412
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.578
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.677

    