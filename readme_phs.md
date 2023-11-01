# Usage
## List
- **patch_reuse.py**: 使用（选择）base帧和当前帧，提取base帧的真值信息，当前帧reuse base frame 一定比例的patch。还包括patc_mask功能 随机mask和非ROI mask

- **patch_reuse_evaluate.py**: 同patch_reuse.py 测试整个测试集在reuse patch情况下准确度的变化。base frame 和current frame 设置为 current frame reuse X 帧之前的 base frame。
- **token_reuse_evaluate.py**: --max_reuse_frame 表示在X帧内，后X-1帧 reuse最开始一帧的patchs。测试整个测试集的精度变化。输出为
- **tools/det_token_attention_bbox.py**: 使用YOLOS文章中的技巧，可视化置信度大于阈值的det_tokens的self-attention热力图。
- **tools/detr_attenton_visualizer.py**: 使用detr文章的技巧，可视化encoder中最后一层的self-attention，并以bboxes框中心作为索引值（reference point）可视化当前的热力图。
- **patch_drop.py**: 直接丢弃（不是mask）一定比例的非ROI区域的patch，ROI由真值提供。
- **token_merge.py**: TOKEN Merging : Your ViT But Faster
- **token_reorganizations** : token_reorganizations ICLR'22
  
## History
- 2023.11.1-1 保存token_reuse_evaluate.py 其中 self.merge self.replace功能，merge表示丢弃的一部分token经过average weights后重新形成一个token加入到patch token后面，replace功能表示丢弃的一部分attention weight不太重要的token并用base frame 的token代替，效果都不是太好，探索学习的方法。


## Result 
### YOLOS Base
1. 使用raw YOLOS_B，results/MOT15Det_base/checkpoint0199.pth：
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.467
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.847
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.466

2. 使用 YOLOS_B, reuse_frame = 2, drop_proportion = 1.0\
        results/MOT15Det_base/checkpoint0199.pth，并且用真值反馈

### YOLOS Tiny
1. 使用raw YOLOS_T，results/MOT15Det_tiny/checkpoint0299.pth： [848 x 480]
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.433
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.804
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.416

2. 使用 **token_reuse_evaluate.py**, reuse_frame = 2, drop_proportion = 1.0\
        results/MOT15Det_tiny/checkpoint0299.pth，并且用真值反馈

    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.432
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.802
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.415
     
3. 使用 **token_reuse_evaluate.py**, reuse_frame = 3, drop_proportion = 1.0\
        results/MOT15Det_tiny/checkpoint0299.pth，并且用真值反馈

    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.430
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.797
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.415
    
4. 使用 **token_reuse_evaluate.py**, reuse_frame = 4, drop_proportion = 1.0\
        results/MOT15Det_tiny/checkpoint0299.pth，并且用真值反馈

    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.425
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.793
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.413

Reuse Proportion:  0.8033412173652144
由此可见他们等价

8. 使用 **token_reuse_evaluate.py**, reuse_frame = 5, drop_proportion = 1.0\
        results/MOT15Det_tiny/checkpoint0299.pth，并且用真值反馈

    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.421
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.785
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.403

9. 使用 **token_reuse_evaluate.py**, reuse_frame = 6, drop_proportion = 1.0\
        results/MOT15Det_tiny/checkpoint0299.pth，并且用真值反馈

    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.415
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.777
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.397

10. 使用 **token_reuse_evaluate.py**, reuse_frame = 7, drop_proportion = 1.0\
        results/MOT15Det_tiny/checkpoint0299.pth，并且用真值反馈

    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.410
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.766
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.391



1. 使用 **token_reuse_evaluate.py**, reuse_frame = 2, drop_proportion = 1.0\
        results/MOT15Det_tiny/checkpoint0299.pth，base frame 检测值反馈
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.430
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.800
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.411

2. 使用 **token_reuse_evaluate.py**, reuse_frame = 3, drop_proportion = 1.0\
        results/MOT15Det_tiny/checkpoint0299.pth，base frame 检测值反馈
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.426
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.796
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.408
    
3. 使用 **token_reuse_evaluate.py**, reuse_frame = 4, drop_proportion = 1.0\
        results/MOT15Det_tiny/checkpoint0299.pth，base frame 检测值反馈
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.419
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.788
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.400

4. 使用 **token_reuse_evaluate.py**, reuse_frame = 5, drop_proportion = 1.0\
        results/MOT15Det_tiny/checkpoint0299.pth，base frame 检测值反馈
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.415
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.782
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.398

Reuse Proportion:  0.8203114300504226

### Token Drop
1.  python patch_drop.py --drop_proportion 0.1
    reuse_proportion: 0.09972677595628415

    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.427
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.804
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.406

2.  python patch_drop.py --drop_proportion 0.2
    reuse_proportion: 0.1994535519125683
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.409
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.801
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.371

3. python patch_drop.py --drop_proportion 0.3
    reuse_proportion: 0.2998633879781421
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.376
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.784
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.316

4. python patch_drop.py --drop_proportion 0.4
    reuse_proportion: 0.39959016393442626
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.332
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.747
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.239

5. python patch_drop.py --drop_proportion 0.5
    reuse_proportion: 0.5
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.277
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.699
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.153

6. python patch_drop.py --drop_proportion 0.6
    reuse_proportion: 0.5997267759562842
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.210
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.605
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.087

7. python patch_drop.py --drop_proportion 0.7
    reuse_proportion: 0.6994535519125683
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.138
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.465
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.033

8. python patch_drop.py --drop_proportion 0.8
    reuse_proportion: 0.8
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.074
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.295
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.010

9. python patch_drop.py --drop_proportion 0.9
    reuse_proportion: 0.8995901639344263
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.033
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.147
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.004

10. python patch_drop.py --drop_proportion 1.0
    reuse_proportion: 1.0
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.013
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.058
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.002

### Token Progressively Drop
1.  10%
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.432
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.804
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.415

2. 20%
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.429
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.806
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.407

3. 30%
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.426
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.809
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.404

4. 40%
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.418
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.806
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.391

5. 50%
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.409
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.806
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.367

6. 60%
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.393
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.797
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.339

7. 70%
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.376
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.790
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.309

8. 80%
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.351
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.773
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.265

9. 90%
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.323
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.757
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.213

10. 100%
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.286
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.723
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.155
    
### Token Merging

1. 10% 分12次Merging 10%的所有token
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.422
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.799
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.392

2. 100% 分12次Merging 100%的所有token


### Evit
1. keep_rate = 1.0： [848 x 480]
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.433
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.804
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.416

2. keep_rate = 0.9:
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.421
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.800
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.401


### Token_reuse_evaluate 
python token_reuse_evaluate.py --max_reuse_frame 5 --drop_proportion 1.0 block [3,6,9] 保留前95% topk token
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.399
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.769
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.371

python token_reuse_evaluate.py --max_reuse_frame 5 --drop_proportion 1.0 block [3,6,9] 保留前95% topk token 并且merge 剩下的5% token
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.400
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.770
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.371

python token_reuse_evaluate.py --max_reuse_frame 5 --drop_proportion 1.0 block [3,6,9] replace 剩下的10%不重要的token
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.400
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.766
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.372

python token_reuse_evaluate.py --max_reuse_frame 5 --drop_proportion 1.0 block [3,6,9] 保留前90% topk token
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.367
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.731
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.331
