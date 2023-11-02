import torch,math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from functools import partial
from einops import rearrange
from .layers import DropPath, to_2tuple, trunc_normal_
import torch.utils.checkpoint as checkpoint 
from models.tome_merge import bipartite_soft_matching, merge_source, merge_wavg
from models.utils import batch_index_select

def complement_idx(idx, dim):
    """
    Compute the complement: set(range(dim)) - set(idx).
    idx is a multi-dimensional tensor, find the complement for its trailing dimension,
    all other dimension is considered batched.
    Args:
        idx: input index, shape: [N, *, K]
        dim: the max index for complement
    """
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl

class PredictorLG(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=192):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, policy):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:,:, :C//2]
        global_x = (x[:,:, C//2:] * policy).sum(dim=1, keepdim=True) / torch.sum(policy, dim=1, keepdim=True)
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)
        return self.out_conv(x)
    
class SViTPredictor(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=192):
        super().__init__()
        self.predict = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//4),
            nn.Linear(embed_dim//4, 2)
        )

    def forward(self, x):
        return self.predict(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention = False, det_token_index : list = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 3, B, num_head, N, c
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # if det_token_index is not None:
        #     det_token_index += (N-100) * np.ones_like(det_token_index)
        #     # det_token_index = sorted(det_token_index)
        #     keep_token = torch.tensor(np.concatenate((np.arange(0,N-100), det_token_index), axis=0))
        #     keep_token = keep_token.to(x.device)
        #     attn = torch.index_select(attn, 3, keep_token)
        #     v = torch.index_select(v, 2, keep_token)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attention:
            return x, attn
        else:
            return x

class DynamicAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, N, _ = policy.size()
        B, H, N, N = attn.size()
        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # attn = attn.exp_() * attn_policy
        # return attn / attn.sum(dim=-1, keepdim=True)

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(self, x, policy):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if policy is None:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn, policy)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False, det_token_index : list = None):
        if det_token_index is not None:
            y, attn = self.attn(self.norm1(x), det_token_index = det_token_index,return_attention=return_attention)
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            if det_token_index is not None:
                det_token_index += (x.shape[1]-100) * np.ones_like(det_token_index)
                keep_token = torch.tensor(np.concatenate((np.arange(0,x.shape[1]-100), det_token_index), axis=0))
                keep_token = keep_token.to(x.device)
                x = torch.index_select(x, 1, keep_token)
            
            return x, attn
        elif return_attention:
            y, attn = self.attn(self.norm1(x), return_attention=return_attention)
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

class DynamicBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = DynamicAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, policy=None):
        x = x + self.drop_path(self.attn(self.norm1(x), policy=policy))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x        

class AttentivenessBlock(Block):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0, attn_drop=0, drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm,\
                 fuse_token=False, det_token_num=100, keep_rate=0.9):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer, norm_layer)
        self.fuse_token = fuse_token
        self.det_token_num = det_token_num
        self.keep_rate = keep_rate
        
    def forward(self, x, return_attention=False):
        B, N, C = x.shape        
        y, attn = self.attn(self.norm1(x),return_attention=True)
        if self.keep_rate < 1:
            left_tokens = math.ceil(self.keep_rate * (N - 101))
            det_attn = attn[:, :, -self.det_token_num:, 1:-self.det_token_num,]  # [B, H, N-1]
            det_attn = torch.mean(det_attn, dim=1)  # [B, N-1]
            attention_value = torch.mean(det_attn, dim=1)

            _, idx = torch.topk(attention_value, left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens]

            index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]
            x = x + self.drop_path(y)
            if index is not None:
                # B, N, C = x.shape
                non_cls = x[:, 1:-self.det_token_num, :]  # [B, N-1, C]
                # x_others = torch.gather(non_cls, dim=1, index=index)  # [B, left_tokens, C]
                idx = torch.sort(idx, dim=1)[0][0]
                x_others = non_cls[:, idx, :]

                if self.fuse_token:
                    compl = complement_idx(idx, N - 101)  # [B, N-1-left_tokens]
                    # non_topk = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]
                    non_topk = non_cls[:, compl, :]                    
                    non_topk_attn = torch.gather(attention_value, dim=1, index=compl.unsqueeze(0))  # [B, N-1-left_tokens]
                    extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
                    x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)
                else:
                    x = torch.cat([x[:, 0:1], x_others, x[:,-self.det_token_num:]], dim=1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        else:
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

class MergeBlock(Block):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0, attn_drop=0, drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, det_token_num=100):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer, norm_layer)
        self.det_token_num = det_token_num    
        self.tome_size = None

    def forward(self, x, merging_proportion = 0.9/12, return_attention=False):
        if return_attention:
            y, attn = self.attn(self.norm1(x), return_attention=return_attention)
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn
        else:

            y = self.attn(self.norm1(x), return_attention=return_attention)
            x = x + self.drop_path(y)
            #extract patch tokens
            cls_tokens, patch_token, det_token = x[:, 0:1, :], x[:, 1:-self.det_token_num, :], x[:, -self.det_token_num:, :]
            #calculate the number of patch tokens to be merged
            r = int(merging_proportion * 1590)
            #init the merge function
            merge, _ = bipartite_soft_matching(
                patch_token,
                r,
                False,
                False,
            )
            #merge the patch tokens
            patch_token, self.tome_size = merge_wavg(merge, patch_token, None)
            #concat the cls tokens, patch tokens and det tokens
            x = torch.cat((cls_tokens, patch_token, det_token), dim=1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=[512,864], patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        embeddings = self.proj(x).flatten(2).transpose(1, 2)
        return embeddings

class ReuseEmbed(PatchEmbed):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=[512,864], patch_size=16, in_chans=3, embed_dim=384):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

    def forward(self, x, reuse_embedding, reuse_region, drop_proportion=0.1):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        patch_dim_w, patch_dim_h = W//16, H//16
        all_indices = set(range(patch_dim_h * patch_dim_w))
        row = None
        reuse_proportion = 0
        if reuse_embedding is not None or reuse_region is not None:
            for bbox in reuse_region:
                x1, y1, x2, y2 = bbox
                # 计算与bounding box相关的patch的开始和结束索引
                start_row_idx = y1 //16
                start_col_idx = x1 // 16
                end_row_idx = y2 // 16
                end_col_idx = x2 // 16
                
                # 从所有的patch中移除当前bounding box内的patch
                for i in range(int(start_row_idx), int(end_row_idx)+1):
                    for j in range(int(start_col_idx), int(end_col_idx)+1):
                        patch_idx = i * patch_dim_w + j
                        all_indices.discard(patch_idx)
            drop_num = int(len(all_indices) * drop_proportion)
            row = np.random.choice(list(all_indices), size=drop_num, replace=False)
            reuse_proportion = len(row)/len(set(range(patch_dim_h * patch_dim_w)))
        embeddings_to_save = self.proj(x)
        embeddings_to_save = embeddings_to_save.flatten(2)
        if reuse_embedding is not None and len(row) > 0:
            replace_embedding = reuse_embedding[:,:,row]
            embeddings_to_save[:,:,row] = replace_embedding
        embeddings = embeddings_to_save.transpose(1, 2)
        intermediate_data = {'reuse_proportion': reuse_proportion}
        return embeddings, embeddings_to_save, intermediate_data

class DropEmbed(PatchEmbed):
    def __init__(self, img_size=[512, 864], patch_size=16, in_chans=3, embed_dim=384):
        super().__init__(img_size, patch_size, in_chans, embed_dim)
    
    def forward(self, x, row):
        B, C, H, W = x.shape
        x1 = self.proj(x).flatten(2)

        # 创建一个包含所有索引的array，然后取补集
        all_indices = np.arange(x1.size(2))
        indices_to_keep = torch.tensor(np.setdiff1d(all_indices, row)).to("cuda")

        # 使用index_select来选择希望保留的索引
        x2 = x1.index_select(2, indices_to_keep)

        embeddings = x2.transpose(1, 2)
        return embeddings

class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, is_distill=False):
        super().__init__()
        
        if isinstance(img_size,tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        
        self.depth = depth
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if is_distill:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 2, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # set finetune flag
        self.has_mid_pe = False

    def bboxes_to_row(self, bboxes_mask : list, input_size : tuple, drop_proportion : float = 1.0,):
        patch_dim_w, patch_dim_h = input_size[0] // self.patch_size, input_size[1] // self.patch_size
        patch_num = patch_dim_h * patch_dim_w
        all_indices = set(range(patch_num))
        for bbox in bboxes_mask:
            x1, y1, x2, y2 = bbox
            # 计算与bounding box相关的patch的开始和结束索引
            start_row_idx = y1 // self.patch_size
            start_col_idx = x1 // self.patch_size
            end_row_idx = y2 // self.patch_size
            end_col_idx = x2 // self.patch_size
            
            # 从所有的patch中移除当前bounding box内的patch
            for i in range(int(start_row_idx), int(end_row_idx)+1):
                for j in range(int(start_col_idx), int(end_col_idx)+1):
                    patch_idx = i * patch_dim_w + j
                    all_indices.discard(patch_idx)

            mask_num = int(len(all_indices) * drop_proportion)
            # 从除所有bounding boxes外的patches中随机选择要mask的patches
            row = np.random.choice(list(all_indices), size=mask_num, replace=False)
            return row
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def finetune_det(self, img_size=[800, 1344], det_token_num=100, mid_pe_size=None, use_checkpoint=False):
        # import pdb;pdb.set_trace()

        import math
        g = math.pow(self.pos_embed.size(1) - 1, 0.5)
        if int(g) - g != 0:
            self.pos_embed = torch.nn.Parameter(self.pos_embed[:, 1:, :])

        self.det_token_num = det_token_num
        self.det_token = nn.Parameter(torch.zeros(1, det_token_num, self.embed_dim))
        self.det_token = trunc_normal_(self.det_token, std=.02)
        cls_pos_embed = self.pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:,None]
        det_pos_embed = torch.zeros(1, det_token_num, self.embed_dim)
        det_pos_embed = trunc_normal_(det_pos_embed, std=.02)
        patch_pos_embed = self.pos_embed[:, 1:, :]
        patch_pos_embed = patch_pos_embed.transpose(1,2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)
        H, W = img_size
        new_P_H, new_P_W = H//self.patch_size, W//self.patch_size
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H,new_P_W), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        self.pos_embed = torch.nn.Parameter(torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=1))
        self.img_size = img_size
        if mid_pe_size == None:
            self.has_mid_pe = False
            print('No mid pe')
        else:
            print('Has mid pe')
            self.mid_pos_embed = nn.Parameter(torch.zeros(self.depth - 1, 1, 1 + (mid_pe_size[0] * mid_pe_size[1] // self.patch_size ** 2) + 100, self.embed_dim))
            trunc_normal_(self.mid_pos_embed, std=.02)
            self.has_mid_pe = True
            self.mid_pe_size = mid_pe_size
        self.use_checkpoint=use_checkpoint

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'det_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def InterpolateInitPosEmbed(self, pos_embed, img_size=(800, 1344)):
        # import pdb;pdb.set_trace()
        cls_pos_embed = pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:,None]
        det_pos_embed = pos_embed[:, -self.det_token_num:,:]
        patch_pos_embed = pos_embed[:, 1:-self.det_token_num, :]
        patch_pos_embed = patch_pos_embed.transpose(1,2)
        B, E, Q = patch_pos_embed.shape


        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        # patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        H, W = img_size
        new_P_H, new_P_W = H//self.patch_size, W//self.patch_size
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H,new_P_W), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=1)
        return scale_pos_embed

    def InterpolateMidPosEmbed(self, pos_embed, img_size=(800, 1344)):
        # import pdb;pdb.set_trace()
        cls_pos_embed = pos_embed[:, :, 0, :]
        cls_pos_embed = cls_pos_embed[:,None]
        det_pos_embed = pos_embed[:, :, -self.det_token_num:,:]
        patch_pos_embed = pos_embed[:, :, 1:-self.det_token_num, :]
        patch_pos_embed = patch_pos_embed.transpose(2,3)
        D, B, E, Q = patch_pos_embed.shape

        P_H, P_W = self.mid_pe_size[0] // self.patch_size, self.mid_pe_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(D*B, E, P_H, P_W)
        H, W = img_size
        new_P_H, new_P_W = H//self.patch_size, W//self.patch_size
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H,new_P_W), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2).contiguous().view(D,B,new_P_H*new_P_W,E)
        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=2)
        return scale_pos_embed

    def forward_features(self, x):
        # import pdb;pdb.set_trace()
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        # if (H,W) != self.img_size:
        #     self.finetune = True

        x = self.patch_embed(x)
        # interpolate init pe
        if (self.pos_embed.shape[1] - 1 - self.det_token_num) != x.shape[1]:
            temp_pos_embed = self.InterpolateInitPosEmbed(self.pos_embed, img_size=(H,W))
        else:
            temp_pos_embed = self.pos_embed
        # interpolate mid pe
        if self.has_mid_pe:
            # temp_mid_pos_embed = []
            if (self.mid_pos_embed.shape[2] - 1 - self.det_token_num) != x.shape[1]:
                temp_mid_pos_embed = self.InterpolateMidPosEmbed(self.mid_pos_embed, img_size=(H,W))
            else:
                temp_mid_pos_embed = self.mid_pos_embed


        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        det_token = self.det_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x, det_token), dim=1)
        x = x + temp_pos_embed
        x = self.pos_drop(x)

        for i in range(len((self.blocks))):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(self.blocks[i], x)    # saves mem, takes time
            else:
                x = self.blocks[i](x)
            if self.has_mid_pe:
                if i < (self.depth - 1):
                    x = x + temp_mid_pos_embed[i]

        x = self.norm(x)

        return x[:, -self.det_token_num:, :]

    def forward_return_all_selfattention(self, x):
        # import pdb;pdb.set_trace()
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        # if (H,W) != self.img_size:
        #     self.finetune = True

        x = self.patch_embed(x)
        # interpolate init pe
        if (self.pos_embed.shape[1] - 1 - self.det_token_num) != x.shape[1]:
            temp_pos_embed = self.InterpolateInitPosEmbed(self.pos_embed, img_size=(H,W))
        else:
            temp_pos_embed = self.pos_embed
        # interpolate mid pe
        if self.has_mid_pe:
            # temp_mid_pos_embed = []
            if (self.mid_pos_embed.shape[2] - 1 - self.det_token_num) != x.shape[1]:
                temp_mid_pos_embed = self.InterpolateMidPosEmbed(self.mid_pos_embed, img_size=(H,W))
            else:
                temp_mid_pos_embed = self.mid_pos_embed


        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        det_token = self.det_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x, det_token), dim=1)
        x = x + temp_pos_embed
        x = self.pos_drop(x)
        output = []
        for i in range(len((self.blocks))):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(self.blocks[i], x)    # saves mem, takes time
            else:
                x, attn = self.blocks[i](x, return_attention=True)

            if i == len(self.blocks)-1:
                output.append(attn)
            if self.has_mid_pe:
                if i < (self.depth - 1):
                    x = x + temp_mid_pos_embed[i]

        x = self.norm(x)

        return output

    def forward(self, x, return_attention=False):
        if return_attention == True:
            # return self.forward_selfattention(x)
            return self.forward_return_all_selfattention(x)
        else:
            x = self.forward_features(x)
            return x

class SViT(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,\
                  drop_rate=0, attn_drop_rate=0, drop_path_rate=0, hybrid_backbone=None, norm_layer=nn.LayerNorm, is_distill=False, pruning_loc = [3,4,5,6,7,8,9,10,11],\
                  keep_rate = [0.7,0.7,0.7,0.49,0.49,0.49,0.343,0.343,0.343]):
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, hybrid_backbone, norm_layer, is_distill)
        self.pruning_loc = pruning_loc
        self.token_ratio = keep_rate
        predictor_list = [SViTPredictor(embed_dim) for _ in range(len(pruning_loc))]
        self.score_predictor = nn.ModuleList(predictor_list)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            DynamicBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

    def forward_features(self, x):
        # import pdb;pdb.set_trace()
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        # if (H,W) != self.img_size:
        #     self.finetune = True

        x = self.patch_embed(x)
        # interpolate init pe
        if (self.pos_embed.shape[1] - 1 - self.det_token_num) != x.shape[1]:
            temp_pos_embed = self.InterpolateInitPosEmbed(self.pos_embed, img_size=(H,W))
        else:
            temp_pos_embed = self.pos_embed
        # interpolate mid pe
        if self.has_mid_pe:
            # temp_mid_pos_embed = []
            if (self.mid_pos_embed.shape[2] - 1 - self.det_token_num) != x.shape[1]:
                temp_mid_pos_embed = self.InterpolateMidPosEmbed(self.mid_pos_embed, img_size=(H,W))
            else:
                temp_mid_pos_embed = self.mid_pos_embed

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        det_token = self.det_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x, det_token), dim=1)
        x = x + temp_pos_embed
        x = self.pos_drop(x)

        ##############Dynamic ViT################
        p_count = 0
        out_pred_prob = []
        init_n = H//self.patch_size * W//self.patch_size
        policy = torch.ones(B, init_n + 101, 1, dtype=x.dtype, device=x.device)
        for i, blk in enumerate(self.blocks):
            if i in self.pruning_loc:
                spatial_x = x[:, 1:-self.det_token_num, :]
                pred_score = self.score_predictor[p_count](spatial_x)
                if self.training:
                    hard_keep_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1]
                    out_pred_prob.append(hard_keep_decision.reshape(B, init_n))
                    cls_policy = torch.ones(B, 1, 1, dtype=hard_keep_decision.dtype, device=hard_keep_decision.device)
                    det_policy = torch.ones(B, 100, 1, dtype=hard_keep_decision.dtype, device=hard_keep_decision.device)
                    policy = torch.cat([cls_policy, hard_keep_decision, det_policy], dim=1)
                    reuse_token = x.clone()
                    selected_x = blk(x, policy=policy)
                    mask = policy.bool().expand(-1, -1, 192)
                    x = torch.where(mask, selected_x, reuse_token)
                else:
                    score = pred_score[:,:,0]
                    num_keep_node = int(init_n * self.token_ratio[p_count])
                    keep_policy = torch.argsort(score, dim=1, descending=True)[:, :num_keep_node]
                    cls_policy = torch.ones(B, 1, dtype=keep_policy.dtype, device=keep_policy.device)
                    keep_det = torch.arange(start=x.shape[1]-self.det_token_num, end=x.shape[1], dtype=keep_policy.dtype, device=keep_policy.device)
                    det_policy = keep_det.unsqueeze(0).expand(B, -1)
                    now_policy = torch.cat([cls_policy, keep_policy + 1, det_policy], dim=1)
                    selected_x = batch_index_select(x, now_policy)
                    # 对选中的token应用blk
                    processed_x = blk(selected_x)
                    # 创建一个新的tensor，大小与原始x相同，用于存放最终结果
                    final_x = torch.empty_like(x)
                    # 将经过处理的token插入到最终结果中
                    # 创建一个布尔掩码，用于确定now_policy中的位置
                    mask = torch.zeros_like(x, dtype=torch.bool)
                    mask.scatter_(1, now_policy.unsqueeze(-1).expand(-1, -1, x.size(-1)), 1)
                    # 将未经处理的token放入final_x
                    final_x[~mask] = x[~mask]
                    # 将经过处理的token放入final_x
                    # 需要保留mask中True的索引
                    mask_indices = mask.nonzero(as_tuple=True)
                    # 创建一个序列，包含0到B-1的数字，用于索引批次
                    batch_indices = torch.arange(B, device=x.device)
                    # 为每个批次应用对应的now_policy中的indices
                    for i in range(B):
                        batch_mask = batch_indices == i
                        selected_indices = now_policy[i]
                        final_x[batch_mask, selected_indices] = processed_x[i]
                    x = final_x
                p_count += 1
            else:
                if self.training:
                    x = blk(x, policy)
                else:
                    x = blk(x)

        x = self.norm(x)

        if self.training:
            return x[:, -self.det_token_num:, :], out_pred_prob
        else:
            return x[:, -self.det_token_num:, :], None
        
    def forward(self, x, return_attention=False):
        if return_attention == True:
            # return self.forward_selfattention(x)
            return self.forward_return_all_selfattention(x)
        else:
            x, out_pred_prob = self.forward_features(x)
            return x, out_pred_prob


class DynamicVisionTransformer(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,\
                  drop_rate=0, attn_drop_rate=0, drop_path_rate=0, hybrid_backbone=None, norm_layer=nn.LayerNorm, is_distill=False, pruning_loc = [3,6,9],\
                  keep_rate = 0.9):
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, hybrid_backbone, norm_layer, is_distill)
        self.pruning_loc = pruning_loc
        self.token_ratio = [keep_rate, keep_rate ** 2, keep_rate ** 3]
        predictor_list = [PredictorLG(embed_dim) for _ in range(len(pruning_loc))]
        self.score_predictor = nn.ModuleList(predictor_list)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            DynamicBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

    def forward_features(self, x):
        # import pdb;pdb.set_trace()
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        # if (H,W) != self.img_size:
        #     self.finetune = True

        x = self.patch_embed(x)
        # interpolate init pe
        if (self.pos_embed.shape[1] - 1 - self.det_token_num) != x.shape[1]:
            temp_pos_embed = self.InterpolateInitPosEmbed(self.pos_embed, img_size=(H,W))
        else:
            temp_pos_embed = self.pos_embed
        # interpolate mid pe
        if self.has_mid_pe:
            # temp_mid_pos_embed = []
            if (self.mid_pos_embed.shape[2] - 1 - self.det_token_num) != x.shape[1]:
                temp_mid_pos_embed = self.InterpolateMidPosEmbed(self.mid_pos_embed, img_size=(H,W))
            else:
                temp_mid_pos_embed = self.mid_pos_embed

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        det_token = self.det_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x, det_token), dim=1)
        x = x + temp_pos_embed
        x = self.pos_drop(x)

        ##############Dynamic ViT################
        p_count = 0
        out_pred_prob = []
        init_n = H//self.patch_size * W//self.patch_size
        prev_decision = torch.ones(B, init_n, 1, dtype=x.dtype, device=x.device)
        policy = torch.ones(B, init_n + 101, 1, dtype=x.dtype, device=x.device)
        for i, blk in enumerate(self.blocks):
            if i in self.pruning_loc:
                spatial_x = x[:, 1:-self.det_token_num, :]
                pred_score = self.score_predictor[p_count](spatial_x, prev_decision).reshape(B, -1, 2)
                if self.training:
                    hard_keep_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1] * prev_decision
                    out_pred_prob.append(hard_keep_decision.reshape(B, init_n))
                    cls_policy = torch.ones(B, 1, 1, dtype=hard_keep_decision.dtype, device=hard_keep_decision.device)
                    det_policy = torch.ones(B, 100, 1, dtype=hard_keep_decision.dtype, device=hard_keep_decision.device)
                    policy = torch.cat([cls_policy, hard_keep_decision, det_policy], dim=1)
                    x = blk(x, policy=policy)
                    prev_decision = hard_keep_decision
                else:
                    score = pred_score[:,:,0]
                    num_keep_node = int(init_n * self.token_ratio[p_count])
                    keep_policy = torch.argsort(score, dim=1, descending=True)[:, :num_keep_node]
                    cls_policy = torch.zeros(B, 1, dtype=keep_policy.dtype, device=keep_policy.device)
                    keep_det = torch.arange(start=x.shape[1]-self.det_token_num, end=x.shape[1], dtype=keep_policy.dtype, device=keep_policy.device)
                    det_policy = keep_det.unsqueeze(0).expand(B, -1)
                    now_policy = torch.cat([cls_policy, keep_policy + 1, det_policy], dim=1)
                    x = batch_index_select(x, now_policy)
                    prev_decision = batch_index_select(prev_decision, keep_policy)
                    x = blk(x)
                p_count += 1
            else:
                if self.training:
                    x = blk(x, policy)
                else:
                    x = blk(x)

        x = self.norm(x)

        if self.training:
            return x[:, -self.det_token_num:, :], out_pred_prob
        else:
            return x[:, -self.det_token_num:, :], None
        
    def forward(self, x, return_attention=False):
        if return_attention == True:
            # return self.forward_selfattention(x)
            return self.forward_return_all_selfattention(x)
        else:
            x, out_pred_prob = self.forward_features(x)
            return x, out_pred_prob
        
class VisionTransformerTokenMerging(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop_rate=0, attn_drop_rate=0, drop_path_rate=0, hybrid_backbone=None, norm_layer=nn.LayerNorm, is_distill=False, det_token_num=100):
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, hybrid_backbone, norm_layer, is_distill)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
        MergeBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, det_token_num=det_token_num)
        for i in range(depth)])

class VisionTransformerTokenReuse(VisionTransformer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, is_distill=False):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, embed_dim=embed_dim, depth=depth,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate, hybrid_backbone=hybrid_backbone, norm_layer=norm_layer, is_distill=is_distill)
        self.patch_embed = ReuseEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.merge = False
        self.replace = False

    def forward_features(self, x, additional_data):
        reuse_embedding = additional_data['reuse_embedding']
        reuse_region = additional_data['reuse_region']
        drop_proportion = additional_data['drop_proportion']
        det_token_index = additional_data['det_token_index']
        reuse_block_token = None if 'block_token' not in additional_data else additional_data['block_token']
        # import pdb;pdb.set_trace()
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        # if (H,W) != self.img_size:
        #     self.finetune = True
        x, saved_embedding,intermediate_data = self.patch_embed(x, reuse_embedding, reuse_region, drop_proportion)
        # interpolate init pe
        if (self.pos_embed.shape[1] - 1 - self.det_token_num) != x.shape[1]:
            temp_pos_embed = self.InterpolateInitPosEmbed(self.pos_embed, img_size=(H,W))
        else:
            temp_pos_embed = self.pos_embed
        # interpolate mid pe
        if self.has_mid_pe:
            # temp_mid_pos_embed = []
            if (self.mid_pos_embed.shape[2] - 1 - self.det_token_num) != x.shape[1]:
                temp_mid_pos_embed = self.InterpolateMidPosEmbed(self.mid_pos_embed, img_size=(H,W))
            else:
                temp_mid_pos_embed = self.mid_pos_embed

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        det_token = self.det_token.expand(B, -1, -1)
        patch_num =  deepcopy(x.shape[1])
        x = torch.cat((cls_tokens, x, det_token), dim=1)
        x = x + temp_pos_embed
        x = self.pos_drop(x)
        block_attn = []
        block_token = []
        for i in range(len((self.blocks))):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(self.blocks[i], x)    # saves mem, takes time
            else:
                x,attn = self.blocks[i](x, return_attention=True)
                if reuse_embedding is not None and reuse_region is not None:
                    if i in [3,6,9]:
                        cls_attn, patch_attn, det_attn = attn[:,:,0,:], attn[:,:,1:-self.det_token_num,:], attn[:,:,-self.det_token_num:,:]
                        patch_det_weight = patch_attn[:,:,:,-self.det_token_num:]
                        patch_det_mean_weight = torch.mean(patch_det_weight,dim=1)
                        patch_det_mean_weight = torch.mean(patch_det_mean_weight,dim=-1)
                        drop_num = int(patch_num * 0.1)
                        keep_num = patch_attn.shape[2] - drop_num
                        _,topk_index = torch.topk(patch_det_mean_weight, keep_num, dim=1)
                        topk_index = (torch.sort(topk_index,dim=1)[0][0]) 
                        cls_token, patch_token, det_token = x[:,0,:], x[:,1:-self.det_token_num,:], x[:,-self.det_token_num:,:]
                        new_patch_token = patch_token.clone()
                        if self.replace:
                            all_index = torch.arange(patch_det_weight.shape[2])
                            mask = torch.ones(all_index.shape[0], dtype=torch.bool)
                            mask[topk_index] = False
                            # 使用该 mask 从 all_index 中选出补集
                            complement_set = all_index[mask]
                            reuse_token = reuse_block_token[i][:,complement_set,:]
                            new_patch_token[:,complement_set,:] = reuse_token
                        else:
                            new_patch_token = patch_token[:,topk_index,:]
                        if self.merge:
                            all_index = torch.arange(patch_det_weight.shape[2])
                            complement_set = torch.tensor([x for x in all_index if x not in topk_index])
                            merge_token = patch_token[:,complement_set,:]
                            merge_token = torch.mean(merge_token,dim=1).unsqueeze(0)
                            x = torch.cat((cls_token.unsqueeze(1), new_patch_token, merge_token, det_token), dim=1)
                        else:
                            x = torch.cat((cls_token.unsqueeze(1), new_patch_token, det_token), dim=1)
            if self.has_mid_pe:
                if i < (self.depth - 1):
                    x = x + temp_mid_pos_embed[i]
            block_attn.append(attn)
            block_token.append(x)
        x = self.norm(x)
        intermediate_data['patch_embedding'] = saved_embedding
        intermediate_data['block_attn'] = block_attn
        intermediate_data['block_token'] = block_token
        return x[:, -self.det_token_num:, :], saved_embedding, intermediate_data
    
    def forward_return_all_selfattention(self, x):
        # import pdb;pdb.set_trace()
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        # if (H,W) != self.img_size:
        #     self.finetune = True
        reuse_embedding = None
        reuse_region = None
        x,_,_ = self.patch_embed(x,reuse_embedding,reuse_region)
        # interpolate init pe
        if (self.pos_embed.shape[1] - 1 - self.det_token_num) != x.shape[1]:
            temp_pos_embed = self.InterpolateInitPosEmbed(self.pos_embed, img_size=(H,W))
        else:
            temp_pos_embed = self.pos_embed
        # interpolate mid pe
        if self.has_mid_pe:
            # temp_mid_pos_embed = []
            if (self.mid_pos_embed.shape[2] - 1 - self.det_token_num) != x.shape[1]:
                temp_mid_pos_embed = self.InterpolateMidPosEmbed(self.mid_pos_embed, img_size=(H,W))
            else:
                temp_mid_pos_embed = self.mid_pos_embed


        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        det_token = self.det_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x, det_token), dim=1)
        x = x + temp_pos_embed
        x = self.pos_drop(x)
        output = []
        for i in range(len((self.blocks))):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(self.blocks[i], x)    # saves mem, takes time
            else:
                x, attn = self.blocks[i](x, return_attention=True)

            if i == len(self.blocks)-1:
                output.append(attn)
            if self.has_mid_pe:
                if i < (self.depth - 1):
                    x = x + temp_mid_pos_embed[i]

        x = self.norm(x)

        return output

    def forward(self, x, additional_data=None, return_attention=False):
        if return_attention == True:
            # return self.forward_selfattention(x)
            return self.forward_return_all_selfattention(x, additional_data)
        else:
            x,saved_embedding,intermediate_data = self.forward_features(x, additional_data)
            return x, saved_embedding, intermediate_data

class VisionTransformerPatchDrop(VisionTransformer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, is_distill=False):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, embed_dim=embed_dim, depth=depth,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate, hybrid_backbone=hybrid_backbone, norm_layer=norm_layer, is_distill=is_distill)
        self.patch_embed = DropEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

    def InterpolateInitPosEmbed(self, pos_embed, img_size=(800, 1344),row=[]):
        # import pdb;pdb.set_trace()
        cls_pos_embed = pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:,None]
        det_pos_embed = pos_embed[:, -self.det_token_num:,:]
        patch_pos_embed = pos_embed[:, 1:-self.det_token_num, :]
        patch_pos_embed = patch_pos_embed.transpose(1,2)
        B, E, Q = patch_pos_embed.shape


        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        # patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        H, W = img_size
        new_P_H, new_P_W = H//self.patch_size, W//self.patch_size
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H,new_P_W), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2)
        # 创建一个包含所有索引的array，然后取补集
        all_indices = np.arange(patch_pos_embed.size(2))
        indices_to_keep = torch.tensor(np.setdiff1d(all_indices, row)).to("cuda")

        # 使用index_select来选择希望保留的索引
        patch_pos_embed = patch_pos_embed.index_select(2, indices_to_keep)

        patch_pos_embed = patch_pos_embed.transpose(1, 2)

        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=1)
        return scale_pos_embed

    def forward_features(self, x, row):
        # import pdb;pdb.set_trace()
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        # if (H,W) != self.img_size:
        #     self.finetune = True
        x = self.patch_embed(x, row)
        # interpolate init pe
        if (self.pos_embed.shape[1] - 1 - self.det_token_num) != x.shape[1]:
            temp_pos_embed = self.InterpolateInitPosEmbed(self.pos_embed, img_size=(H,W), row=row)
        else:
            temp_pos_embed = self.pos_embed
        # interpolate mid pe
        if self.has_mid_pe:
            # temp_mid_pos_embed = []
            if (self.mid_pos_embed.shape[2] - 1 - self.det_token_num) != x.shape[1]:
                temp_mid_pos_embed = self.InterpolateMidPosEmbed(self.mid_pos_embed, img_size=(H,W))
            else:
                temp_mid_pos_embed = self.mid_pos_embed

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        det_token = self.det_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x, det_token), dim=1)
        x = x + temp_pos_embed
        x = self.pos_drop(x)

        for i in range(len((self.blocks))):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(self.blocks[i], x)    # saves mem, takes time
            else:
                x = self.blocks[i](x)
            if self.has_mid_pe:
                if i < (self.depth - 1):
                    x = x + temp_mid_pos_embed[i]

        x = self.norm(x)

        return x[:, -self.det_token_num:, :]
    
    def forward(self, x, row, return_attention=False):
        if return_attention == True:
            # return self.forward_selfattention(x)
            return self.forward_return_all_selfattention(x)
        else:
            x = self.forward_features(x,row)
            return x

class VisionTransformerPatchDropProgressively(VisionTransformerPatchDrop):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop_rate=0, attn_drop_rate=0, drop_path_rate=0, hybrid_backbone=None, norm_layer=nn.LayerNorm, is_distill=False):
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate,            attn_drop_rate, drop_path_rate, hybrid_backbone, norm_layer, is_distill)
    
        self.drop_proportion_step = np.linspace(0.0, 1.0, len(self.blocks)) 

    def forward_features(self, x, row):
        # import pdb;pdb.set_trace()
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        # if (H,W) != self.img_size:
        #     self.finetune = True
        # row = self.bboxes_to_row(bboxes_mask, input_size=(W,H))
        patch_to_drop = 0 if row is None else len(row)
        row = [] if row is None else row
        drop_num = self.drop_proportion_step * patch_to_drop
        self.drop_num_per_step = []
        for i in range(len(drop_num)):
            if i > 0:
                self.drop_num_per_step.append(int(drop_num[i]-drop_num[i-1]))
            else:
                self.drop_num_per_step.append(int(drop_num[i]))
        self.drop_num_per_step[-1] += patch_to_drop - sum(self.drop_num_per_step) 
        x = self.patch_embed(x, row=[])
        # interpolate init pe
        if (self.pos_embed.shape[1] - 1 - self.det_token_num) != x.shape[1]:
            temp_pos_embed = self.InterpolateInitPosEmbed(self.pos_embed, row=[], img_size=(H,W))
        else:
            temp_pos_embed = self.pos_embed
        # interpolate mid pe
        if self.has_mid_pe:
            # temp_mid_pos_embed = []
            if (self.mid_pos_embed.shape[2] - 1 - self.det_token_num) != x.shape[1]:
                temp_mid_pos_embed = self.InterpolateMidPosEmbed(self.mid_pos_embed, row=[], img_size=(H,W))
            else:
                temp_mid_pos_embed = self.mid_pos_embed

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        det_token = self.det_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x, det_token), dim=1)
        x = x + temp_pos_embed
        x = self.pos_drop(x)
        for i in range(len((self.blocks))):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(self.blocks[i], x)    # saves mem, takes time
            else:
                x, row = self.progressively_drop(x, row, self.drop_num_per_step[i])
                x = self.blocks[i](x)
            if self.has_mid_pe:
                if i < (self.depth - 1):
                    x = x + temp_mid_pos_embed[i]

        x = self.norm(x)

        return x[:, -self.det_token_num:, :]
    
    def progressively_drop(self, x : torch.Tensor, row : list, drop_num_per_step : int): 
        cls_tokens, patch_token, det_token = x[:, 0:1, :], x[:, 1:-self.det_token_num, :], x[:, -self.det_token_num:, :]
        all_indexes = set(range(patch_token.size(1)))
        drop_indexes = set(np.random.choice(row, size=drop_num_per_step, replace=False))
        keep_indexes = all_indexes - drop_indexes
        patch_token = patch_token[:, list(keep_indexes), :]
        row = self.__update_row(row, drop_indexes)
        x = torch.cat((cls_tokens, patch_token, det_token), dim=1)
        return x, row

    def __update_row(self, row : list, drop_indexes : set):
        if len(drop_indexes) == 0:
            return row
        row = sorted(row)
        cnt = 0
        for i in (range(len(row))):
            if row[i] in drop_indexes:
                row[i] = -1
                cnt += 1
            else:
                row[i] -= cnt
        row = [x for x in row if x != -1]
        return row

class VisionTransformerTokenReorganizations(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop_rate=0, attn_drop_rate=0, drop_path_rate=0, hybrid_backbone=None, norm_layer=nn.LayerNorm, is_distill=False,\
                 keep_rate=1.0):
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, hybrid_backbone, norm_layer, is_distill)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        keep_rates = [1.0,1.0,1.0,keep_rate,1.0,1.0,keep_rate,1.0,1.0,keep_rate,1.0,1.0]
        # whether_fuse = [False,False,False,True,False,False,True,False,False,True,False,False]

        self.blocks = nn.ModuleList([
            AttentivenessBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, keep_rate=keep_rates[i],fuse_token = False)    
            for i in range(depth)])
    
def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

def reuse_tiny(pretrained=None, **kwargs):
    model = VisionTransformerTokenReuse(
                patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6))
    if pretrained: 
        # checkpoint = torch.load('deit_tiny_patch16_224-a1311bcf.pth', map_location="cpu")
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
        #     map_location="cpu", check_hash=True
        # )
        checkpoint = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model, 192

def droptiny(pretrained=None, **kwargs):
    model = VisionTransformerPatchDrop(
                patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6))
    if pretrained: 
        # checkpoint = torch.load('deit_tiny_patch16_224-a1311bcf.pth', map_location="cpu")
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
        #     map_location="cpu", check_hash=True
        # )
        checkpoint = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model, 192

def progressively_drop_tiny(pretrained=None, **kwargs):
    model = VisionTransformerPatchDropProgressively(
                patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6))
    if pretrained: 
        # checkpoint = torch.load('deit_tiny_patch16_224-a1311bcf.pth', map_location="cpu")
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
        #     map_location="cpu", check_hash=True
        # )
        checkpoint = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model, 192

def token_merging_tiny(pretrained=None, **kwargs):
    model = VisionTransformerTokenMerging(
                patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), det_token_num=100)
    if pretrained: 
        # checkpoint = torch.load('deit_tiny_patch16_224-a1311bcf.pth', map_location="cpu")
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
        #     map_location="cpu", check_hash=True
        # )
        checkpoint = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model, 192

def token_reorganizations_tiny(pretrained=None, keep_rate = 1.0, **kwargs):
    model = VisionTransformerTokenReorganizations(
                patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),keep_rate=keep_rate)
    if pretrained: 
        # checkpoint = torch.load('deit_tiny_patch16_224-a1311bcf.pth', map_location="cpu")
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
        #     map_location="cpu", check_hash=True
        # )
        checkpoint = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model, 192

def dynamic_tiny(pretrained=None, **kwargs):
    model = DynamicVisionTransformer(
                patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), pruning_loc=[3,6,9], keep_rate=0.9,\
                )
    if pretrained: 
        # checkpoint = torch.load('deit_tiny_patch16_224-a1311bcf.pth', map_location="cpu")
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
        #     map_location="cpu", check_hash=True
        # )
        checkpoint = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model, 192


def svit_tiny(pretrained=None, **kwargs):
    model = SViT(
                patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), pruning_loc=[3,4,5,6,7,8,9,10,11],\
                keep_rate=[0.7,0.7,0.7,0.49,0.49,0.49,0.343,0.343,0.343],\
                )
    if pretrained: 
        # checkpoint = torch.load('deit_tiny_patch16_224-a1311bcf.pth', map_location="cpu")
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
        #     map_location="cpu", check_hash=True
        # )
        checkpoint = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model, 192

def tiny(pretrained=None, **kwargs):
    model = VisionTransformer(
                patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6))
    if pretrained: 
        # checkpoint = torch.load('deit_tiny_patch16_224-a1311bcf.pth', map_location="cpu")
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
        #     map_location="cpu", check_hash=True
        # )
        checkpoint = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model, 192

def small(pretrained=None, **kwargs):
    model = VisionTransformerTokenReuse(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-12), **kwargs)
    if pretrained:
        # checkpoint = torch.load('deit_small_patch16_224-cd65a155.pth', map_location="cpu")
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
        #     map_location="cpu", check_hash=True
        # )
        checkpoint = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model, 384

def small_dWr(pretrained=None, **kwargs):
    model = VisionTransformer(
        img_size=240, 
        patch_size=16, embed_dim=330, depth=14, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if pretrained:
        # checkpoint = torch.load('fa_deit_ldr_14_330_240.pth', map_location="cpu")
        checkpoint = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model, 330

def base(pretrained=None, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),is_distill=True, **kwargs)
    if pretrained:
        # checkpoint = torch.load('deit_base_distilled_patch16_384-d0272ac0.pth', map_location="cpu")
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
        #     map_location="cpu", check_hash=True
        # )
        checkpoint = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model, 768

def dropbase(pretrained=None, **kwargs):
    model = VisionTransformerPatchDrop(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),is_distill=True, **kwargs)
    if pretrained:
        # checkpoint = torch.load('deit_base_distilled_patch16_384-d0272ac0.pth', map_location="cpu")
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
        #     map_location="cpu", check_hash=True
        # )
        checkpoint = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model, 768

def reuse_base(pretrained=None, **kwargs):
    model = VisionTransformerTokenReuse(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),is_distill=True, **kwargs)
    if pretrained:
        # checkpoint = torch.load('deit_base_distilled_patch16_384-d0272ac0.pth', map_location="cpu")
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
        #     map_location="cpu", check_hash=True
        # )
        checkpoint = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model, 768

def progressively_drop_drop_base(pretrained=None, **kwargs):
    model = VisionTransformerPatchDropProgressively(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),is_distill=True, **kwargs)
    if pretrained:
        # checkpoint = torch.load('deit_base_distilled_patch16_384-d0272ac0.pth', map_location="cpu")
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
        #     map_location="cpu", check_hash=True
        # )
        checkpoint = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model, 768

if __name__ == '__main__':
    model = VisionTransformerTokenReuse(
                patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6))
     