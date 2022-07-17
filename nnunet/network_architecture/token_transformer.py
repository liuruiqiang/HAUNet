# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
Take the standard Transformer as T2T Transformer
"""
import torch.nn as nn
import copy
# from timm.models.layers import DropPath
from nnunet.network_architecture.transformer_block import Mlp,position_enc,learned_position_enc,position_embedding
import numpy as np
# from visualizer import get_local
import torch

# get_local.activate()
# get_local.clear()

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim = None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim

        self.head_dim = in_dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, in_dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, in_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    # @get_local("attn_map")
    def forward(self, x):
        B, N, C = x.shape
        # print('x shape', x.shape)
        # self.rel_h = nn.Parameter(torch.randn([1, n_dims, 1, height]), requires_grad=True)
        # self.rel_w = nn.Parameter(torch.randn([1, n_dims, width, 1]), requires_grad=True)
        if self.sr_ratio > 1:
            q = self.q(x).reshape(B, N, self.num_heads, self.head_dim ).permute(0, 2, 1, 3)
            x_ = x.permute(0, 2, 1).reshape(B, C,int(np.sqrt(N)), int(np.sqrt(N)))
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            # print(x_.shape)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim ).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn_ = attn.clone()
        attn_map = attn.softmax(dim=-1)
        attn = self.attn_drop(attn_map)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.sr_ratio == 1:
            v = v.transpose(1,2).reshape(B, N, self.in_dim)

            x = v.squeeze(1) + x   # because the original x has different size with current x, use v to do skip connection

        return x

class Z_Attention(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

class Focal_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim = None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim

        self.head_dim = in_dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, in_dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, in_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    # @get_local("attn_map")
    def forward(self, q_x,kv_x):
        B, N, C = q_x.shape
        q = self.q(q_x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        x_ = q_x.permute(0, 2, 1).reshape(B, C, int(np.sqrt(N)), int(np.sqrt(N)))
        kv_ = self.kv(kv_x).reshape(B, -1, 2, self.num_heads, self.head_dim ).permute(2, 0, 3, 1, 4)
        k, v = kv_[0], kv_[1]
        # print('x shape', x.shape)
        # self.rel_h = nn.Parameter(torch.randn([1, n_dims, 1, height]), requires_grad=True)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn_ = attn.clone()
        attn_map = attn.softmax(dim=-1)
        attn = self.attn_drop(attn_map)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.sr_ratio == 1:
            q = q.transpose(1,2).reshape(B, N, self.in_dim)

            x = q.squeeze(1) + x   # because the original x has different size with current x, use v to do skip connection

        return x

class Token_transformer(nn.Module):

    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, sr_ratio=1,act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,sr_ratio=sr_ratio)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # print('xshape',x.shape)
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Token_transformer2d(nn.Module):

    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, sr_ratio=1,input_resolution=(56,56),act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.input_resolution = input_resolution
        h, w = input_resolution
        self.position_embedding = position_embedding(h * w, dim, pos_type="learned")
        self.attn = Attention(
            dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,sr_ratio=sr_ratio)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def reshape_out(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(b, h * w, c)
        return x

    def reshape_back(self, x):
        b, n,c = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape((b,c) + self.input_resolution)
        return x

    def forward(self, x):
        # print('xshape',x.shape)
        x = self.reshape_out(x)
        x = self.position_embedding(x)
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.reshape_back(x)
        return x



class Efficient_transformer(nn.Module):

    def __init__(self, dim, in_dim, num_heads,unfold_level, input_resolution,mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, sr_ratio=1,act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        self.unfold_level = unfold_level
        self.input_resolution = input_resolution
        kerner_size = 2 ** unfold_level
        stride = 2 ** unfold_level
        self.norm1 = norm_layer(dim*(stride**2))
        self.unfolds = nn.Unfold(kernel_size=(kerner_size, kerner_size), stride=stride)
        self.folds = nn.Fold(output_size=(input_resolution, input_resolution), kernel_size=(kerner_size, kerner_size),
                    stride=stride)
        self.attn = Attention(
            dim=dim*(stride**2), in_dim=in_dim*(stride**2), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,sr_ratio=sr_ratio)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def reshape_out(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(b, h * w, c)
        return x

    def forward(self, x):
        # print('xshape',x.shape)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        split_tmp = self.unfolds(x).permute(0, 2, 1)
        # print('split_tmp',split_tmp.shape)
        attntion_tmp = self.attn(self.norm1(split_tmp))
        attntion_tmp = attntion_tmp.permute(0, 2, 1)
        x=self.folds(attntion_tmp)
        x = self.reshape_out(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Multi_Gran_Transformer_improved(nn.Module):
    """
    先做不同尺度的unfold，然后做pool成不同尺度的特征进行堆积成为key和value
    """
    def __init__(self, dim, in_dim, num_heads,num_levels,input_resolution, keep_init=True,multi_gran_opt='add',mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, sr_ratio=1,act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.unfolds = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.keep_init = keep_init
        self.multi_gran_opt = multi_gran_opt
        self.norms = nn.ModuleList()
        self.folds = nn.ModuleList()
        # print('heads:...',in_dim,num_heads,type(in_dim),type(num_heads))
        if keep_init:
            self.init_attn = Attention(
                dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, sr_ratio=sr_ratio)
        self.num_levels = num_levels
        for i in range(1,num_levels):
            kerner_size = 2**i
            stride = 2**i
            self.unfolds.append(nn.Unfold(kernel_size=(kerner_size, kerner_size), stride=stride))
            self.folds += [
                nn.Fold(output_size=(input_resolution,input_resolution),kernel_size=(kerner_size, kerner_size), stride=stride)
            ]
            self.attentions += [
                Attention(
                    dim*(stride**2), in_dim=in_dim*(stride**2), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                    proj_drop=drop, sr_ratio=sr_ratio)
            ]
            self.norms += [
                norm_layer(dim*(stride**2))
            ]
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        if self.multi_gran_opt == 'cat' and self.keep_init:
            self.norm2 = norm_layer(in_dim*(num_levels))
            self.mlp = Mlp(in_features=in_dim*(num_levels), hidden_features=int(in_dim *(num_levels)* mlp_ratio),
                           out_features=in_dim*(num_levels),
                           act_layer=act_layer, drop=drop)
            self.mlp2 = nn.Linear(in_dim*(num_levels),in_dim)
            self.drop_path2 = nn.Dropout(0.1)
            self.norm3 = norm_layer(in_dim*(num_levels))
        elif self.multi_gran_opt == 'cat' and not self.keep_init:
            self.norm2 = norm_layer(in_dim * (num_levels-1))
            self.mlp = Mlp(in_features=in_dim * (num_levels-1), hidden_features=int(in_dim*(num_levels-1) * mlp_ratio),
                           out_features=in_dim*(num_levels-1),
                           act_layer=act_layer, drop=drop)
            self.mlp2 = nn.Linear(in_dim * (num_levels-1), in_dim)
            self.drop_path2 = nn.Dropout(0.1)
            self.norm3 = norm_layer(in_dim * (num_levels-1))
        else:
            self.norm2 = norm_layer(in_dim)
            self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def reshape_out(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(b, h * w, c)
        return x

    # @get_local("attention_tmp")
    def forward(self, x):
        # print(self.unfolds)
        x_tmp = x.clone()
        B, new_HW, C = x.shape
        x_tmp = x_tmp.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        x_splits = []
        if self.num_levels >1:
            for i in range(1,self.num_levels):
                split_tmp = self.unfolds[i-1](x_tmp).permute(0,2,1)
                # print('split_tmp',split_tmp.shape)
                attntion_tmp = self.attentions[i-1](self.norms[i-1](split_tmp))
                attntion_tmp = attntion_tmp.permute(0,2,1)
                fold_tmp = self.folds[i-1](attntion_tmp)
                x_splits.append(self.reshape_out(fold_tmp))

        if self.keep_init:
            x = self.init_attn(self.norm1(x))
            attention_tmp = x
        if len(x_splits) > 0:
            if self.multi_gran_opt == 'add':
                cat_tmp = torch.stack(x_splits, dim=0)
                split_result = torch.sum(cat_tmp,dim=0)
                if not self.keep_init:
                    x = split_result
                else:
                    x += split_result

            elif self.multi_gran_opt == 'cat':
                split_cat = torch.cat(x_splits,dim=2)
                if not self.keep_init:
                    x = split_cat
                else:
                    x = torch.cat((x, split_cat), dim=2)
                # if len(x_splits) > 1:
                #     for split in x_splits[1:]:
                #         x = torch.cat((x,split),dim=2)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.multi_gran_opt == "cat":
            x = self.drop_path2(self.mlp2(self.norm3(x)))
        return x

class Multi_Gran_Cross_Transformers(nn.Module):
    def __init__(self,num_layers,dim, in_dim, num_heads, num_levels,input_resolution,keep_init=True,mlp_ratio=1,multi_gran_opt='cat'):
        super().__init__()
        self.num_layer = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers-1):
            block = Multi_Gran_Cross_Transformer(dim,dim,num_heads,num_levels,input_resolution,keep_init=keep_init,mlp_ratio=mlp_ratio,)
            self.layers.append(copy.deepcopy(block))
        self.last_layer = Multi_Gran_Transformer_improved(dim,in_dim,num_heads,num_levels,input_resolution,keep_init=keep_init,mlp_ratio=mlp_ratio)

    def forward(self,x ):
        if self.num_layer > 1 :
            for block in self.layers:
                x = block(x)

        x = self.last_layer(x)
        return x




class Multi_Gran_Cross_Transformer(nn.Module):
    """
    先做不同尺度的unfold，然后做pool成不同尺度的特征进行堆积成为key和value
    """
    def __init__(self, dim, ref_dim,  num_heads,num_levels,input_resolution,start_level=1,
                 keep_init=True,multi_gran_opt='add',mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, sr_ratio=1,act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.unfolds = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.keep_init = keep_init
        self.multi_gran_opt = multi_gran_opt
        self.norms = nn.ModuleList()
        self.folds = nn.ModuleList()
        self.cross_attentions = nn.ModuleList()
        # print('heads:...',in_dim,num_heads,type(in_dim),type(num_heads))
        if keep_init:
            self.init_attn = CrossAttention(
                    dim, ref_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            # self.init_attn = Attention(
            #     dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            #     proj_drop=drop, sr_ratio=sr_ratio)
        self.num_levels = num_levels
        self.start_level = start_level
        for i in range(start_level, num_levels):
        # for i in range(1,num_levels):
        #     print('kenelsize:',2**i)
            kerner_size = 2**i
            stride = 2**i
            self.unfolds.append(nn.Unfold(kernel_size=(kerner_size, kerner_size), stride=stride))
            self.folds += [
                nn.Fold(output_size=(input_resolution,input_resolution),kernel_size=(kerner_size, kerner_size), stride=stride)
            ]
            self.cross_attentions += [
                CrossAttention(
                    dim*(stride**2), ref_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

            ]
            self.norms += [
                norm_layer(dim*(stride**2))
            ]
        self.ref_norm = norm_layer(ref_dim)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        if self.multi_gran_opt == 'cat' and self.keep_init:
            self.norm2 = norm_layer(dim * (num_levels - start_level + 1))
            self.mlp = Mlp(in_features=dim * (num_levels - start_level + 1), hidden_features=int(dim * (num_levels) * mlp_ratio),
                           out_features=dim * (num_levels - start_level + 1),
                           act_layer=act_layer, drop=drop)
            self.mlp2 = nn.Linear(dim * (num_levels - start_level + 1), dim)
            self.drop_path2 = nn.Dropout(0.1)
            self.norm3 = norm_layer(dim * (num_levels - start_level + 1))
        elif self.multi_gran_opt == 'cat' and not self.keep_init:
            self.norm2 = norm_layer(dim * (num_levels - start_level))
            self.mlp = Mlp(in_features=dim * (num_levels - start_level), hidden_features=int(dim * (num_levels - start_level) * mlp_ratio),
                           out_features=dim * (num_levels - start_level),
                           act_layer=act_layer, drop=drop)
            self.mlp2 = nn.Linear(dim * (num_levels - start_level), dim)
            self.drop_path2 = nn.Dropout(0.1)
            self.norm3 = norm_layer(dim * (num_levels - start_level))
        else:
            self.norm2 = norm_layer(dim)
            self.mlp = Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio), out_features=dim, act_layer=act_layer, drop=drop)

    def reshape_out(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(b, h * w, c)
        return x

    # @get_local("attention_tmp")
    def forward(self, x,ref_x,return_ref=False):
        # print(self.unfolds)
        ref_x = self.ref_norm(ref_x)
        x_tmp = x.clone()
        B, new_HW, C = x.shape
        x_tmp = x_tmp.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        x_splits = []
        if self.num_levels >1:
            for i in range(self.start_level, self.num_levels):
                split_tmp = self.unfolds[i - self.start_level](x_tmp).permute(0, 2, 1)
                # print('split_tmp',split_tmp.shape)
                attntion_tmp = self.cross_attentions[i-self.start_level](self.norms[i-self.start_level](split_tmp),ref_x)
                attntion_tmp = attntion_tmp.permute(0,2,1)
                fold_tmp = self.folds[i-self.start_level](attntion_tmp)
                x_splits.append(self.reshape_out(fold_tmp))

        if self.keep_init:
            x = self.init_attn(self.norm1(x),ref_x)
            attention_tmp = x
        if len(x_splits) > 0:
            if self.multi_gran_opt == 'add':
                cat_tmp = torch.stack(x_splits, dim=0)
                split_result = torch.sum(cat_tmp,dim=0)
                if not self.keep_init:
                    x = split_result
                else:
                    x += split_result

            elif self.multi_gran_opt == 'cat':
                split_cat = torch.cat(x_splits,dim=2)
                if not self.keep_init:
                    x = split_cat
                else:
                    x = torch.cat((x, split_cat), dim=2)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.multi_gran_opt == "cat":
            x = self.drop_path2(self.mlp2(self.norm3(x)))
        if return_ref:
            return x,ref_x
        return x
    
class Focal_Transformer(nn.Module):
    """
    先做不同尺度的unfold，然后做pool成不同尺度的特征进行堆积成为key和value
    """
    def __init__(self, dim, in_dim, num_heads,num_levels,input_resolution, keep_init=True,mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, sr_ratio=1,act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_level1 = norm_layer(dim)
        self.norm_level2 = norm_layer(dim*4)
        self.norm_level3 = norm_layer(dim * 16)
        self.fc2 = nn.Linear(dim*4,dim)
        self.fc3 = nn.Linear(dim*16,dim)
        self.norm_level3 = norm_layer(dim*16)
        self.unfolds = nn.ModuleList()
        # print('heads:...',in_dim,num_heads,type(in_dim),type(num_heads))
        self.attn = Focal_Attention(dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, sr_ratio=sr_ratio)
        self.fold =nn.Fold(output_size=(input_resolution,input_resolution),kernel_size=(4, 4), stride=4)
        self.conv = nn.Conv2d(in_dim//16,in_dim,kernel_size=1,)
        for i in range(1,num_levels):
            kerner_size = 2**i
            stride = 2**i
            self.unfolds += [
                nn.Unfold(kernel_size=(kerner_size, kerner_size), stride=stride)
            ]
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim//16)
        self.mlp = Mlp(in_features=in_dim//16, hidden_features=int(in_dim//16*mlp_ratio), out_features=in_dim//16, act_layer=act_layer, drop=drop)

    def reshape_out(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(b, h * w, c)
        return x

    # @get_local("attention_tmp")
    def forward(self, x):
        # print('xshape',x.shape)
        x_tmp = x.clone()
        B, new_HW, C = x.shape
        x = self.norm_level1(x)
        x_tmp = x_tmp.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        split2 = self.unfolds[0](x_tmp).permute(0,2,1)
        split2 = self.norm_level2(split2)
        split2 = self.fc2(split2)
        split3 = self.unfolds[1](x_tmp).permute(0, 2, 1)
        split3 = self.norm_level3(split3)
        split3 = self.fc3(split3)
        kv_ = torch.cat((split2,x),dim=1)

        attn_result = self.attn(split3,kv_)
        attn_result = attn_result.permute(0,2,1)
        fold_result = self.reshape_out(self.fold(attn_result))
        x = fold_result + self.drop_path(self.mlp(self.norm2(fold_result)))


        return x

# model = Focal_Transformer(32,32,num_heads=4,num_levels=3,input_resolution=224)
# model.cuda()
# data = torch.rand(4,224*224,32).cuda()
# out = model(data)
# print(out.shape)


class Multi_Gran_Transformer_posed(nn.Module):
    """
    先做不同尺度的unfold，然后做pool成不同尺度的特征进行堆积成为key和value
    """
    def __init__(self, dim, in_dim, num_heads,num_levels,input_resolution, keep_init=True,multi_gran_opt='add',mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 pos_type="learned",drop_path=0.0, sr_ratio=1,act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.unfolds = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.keep_init = keep_init
        self.multi_gran_opt = multi_gran_opt
        self.norms = nn.ModuleList()
        self.folds = nn.ModuleList()
        self.pos_module = nn.ModuleList()
        if keep_init:
            self.init_attn = Attention(
                dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, sr_ratio=sr_ratio)
            self.init_posemb = position_embedding(input_resolution*input_resolution,dim,pos_type=pos_type)
        self.num_levels = num_levels
        for i in range(1,num_levels):
            kerner_size = 2**i
            stride = 2**i
            self.unfolds += [
                nn.Unfold(kernel_size=(kerner_size, kerner_size), stride=stride)
            ]
            self.pos_module += [
                    position_embedding(input_resolution // kerner_size * input_resolution // kerner_size, dim*(stride**2),pos_type=pos_type)
                ]
            self.folds += [
                nn.Fold(output_size=(input_resolution,input_resolution),kernel_size=(kerner_size, kerner_size), stride=stride)
            ]
            self.attentions += [
                Attention(
                    dim*(stride**2), in_dim=in_dim*(stride**2), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                    proj_drop=drop, sr_ratio=sr_ratio)
            ]
            self.norms += [
                norm_layer(dim*(stride**2))
            ]
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        if self.multi_gran_opt == 'cat' and self.keep_init:
            self.norm2 = norm_layer(in_dim * (num_levels))
            self.mlp = Mlp(in_features=in_dim * (num_levels), hidden_features=int(in_dim * (num_levels) * mlp_ratio),
                           out_features=in_dim * (num_levels),
                           act_layer=act_layer, drop=drop)
            self.mlp2 = nn.Linear(in_dim * (num_levels), in_dim)
            self.drop_path2 = nn.Dropout(0.1)
            self.norm3 = norm_layer(in_dim * (num_levels))
        elif self.multi_gran_opt == 'cat' and not self.keep_init:
            self.norm2 = norm_layer(in_dim * (num_levels - 1))
            self.mlp = Mlp(in_features=in_dim * (num_levels - 1), hidden_features=int(in_dim * (num_levels - 1) * mlp_ratio),
                           out_features=in_dim * (num_levels - 1),
                           act_layer=act_layer, drop=drop)
            self.mlp2 = nn.Linear(in_dim * (num_levels - 1), in_dim)
            self.drop_path2 = nn.Dropout(0.1)
            self.norm3 = norm_layer(in_dim * (num_levels - 1))
        else:
            self.norm2 = norm_layer(in_dim)
            self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def reshape_out(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(b, h * w, c)
        return x

    def forward(self, x):
        x_tmp = x.clone()
        B, new_HW, C = x.shape
        x_tmp = x_tmp.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        x_splits = []
        if self.num_levels >1:
            for i in range(1,self.num_levels):
                split_tmp = self.unfolds[i-1](x_tmp).permute(0,2,1)
                split_tmp = self.pos_module[i-1](split_tmp)
                # print('split_tmp',split_tmp.shape)
                attntion_tmp = self.attentions[i-1](self.norms[i-1](split_tmp))
                attntion_tmp = attntion_tmp.permute(0,2,1)
                fold_tmp = self.folds[i-1](attntion_tmp)
                x_splits.append(self.reshape_out(fold_tmp))

        if self.keep_init:
            x = self.init_posemb(x)
            x = self.init_attn(self.norm1(x))
        if len(x_splits) > 0:
            if self.multi_gran_opt == 'add':
                cat_tmp = torch.stack(x_splits, dim=0)
                split_result = torch.sum(cat_tmp,dim=0)
                if not self.keep_init:
                    x = split_result
                else:
                    x += split_result

            elif self.multi_gran_opt == 'cat':
                if not self.keep_init:
                    x = x_splits[0]
                else:
                    x = torch.cat((x, x_splits[0]), dim=2)
                if len(x_splits) >= 1:
                    for split in x_splits[1:]:
                        x = torch.cat((x,split),dim=2)
        # print('x:',x.shape)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.multi_gran_opt == "cat":
            x = self.drop_path2(self.mlp2(self.norm3(x)))
        return x


class Adap_Gran_Transformer_posed(nn.Module):
    """
    先做不同尺度的unfold，然后做pool成不同尺度的特征进行堆积成为key和value
    """
    def __init__(self, dim, in_dim, num_heads,num_levels,input_resolution,start_level=1,  keep_init=True,multi_gran_opt='add',mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 pos_type="learned",drop_path=0.0, sr_ratio=1,act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.unfolds = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.keep_init = keep_init
        self.multi_gran_opt = multi_gran_opt
        self.start_level = start_level
        self.norms = nn.ModuleList()
        self.folds = nn.ModuleList()
        self.pos_module = nn.ModuleList()
        if keep_init:
            self.init_attn = Attention(
                dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, sr_ratio=sr_ratio)
            self.init_posemb = position_embedding(input_resolution*input_resolution,dim,pos_type=pos_type)
        self.num_levels = num_levels
        for i in range(start_level,num_levels):
            kerner_size = 2**i
            stride = 2**i
            self.unfolds += [
                nn.Unfold(kernel_size=(kerner_size, kerner_size), stride=stride)
            ]
            self.pos_module += [
                    position_embedding(input_resolution // kerner_size * input_resolution // kerner_size, dim*(stride**2),pos_type=pos_type)
                ]
            self.folds += [
                nn.Fold(output_size=(input_resolution,input_resolution),kernel_size=(kerner_size, kerner_size), stride=stride)
            ]
            self.attentions += [
                Attention(
                    dim*(stride**2), in_dim=in_dim*(stride**2), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                    proj_drop=drop, sr_ratio=sr_ratio)
            ]
            self.norms += [
                norm_layer(dim*(stride**2))
            ]
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        if self.multi_gran_opt == 'cat' and self.keep_init:
            self.norm2 = norm_layer(in_dim * (num_levels-start_level+1))
            self.mlp = Mlp(in_features=in_dim * (num_levels-start_level+1), hidden_features=int(in_dim * (num_levels-start_level+1) * mlp_ratio),
                           out_features=in_dim * (num_levels-start_level+1),
                           act_layer=act_layer, drop=drop)
            self.mlp2 = nn.Linear(in_dim * (num_levels-start_level+1), in_dim)
            self.drop_path2 = nn.Dropout(0.1)
            self.norm3 = norm_layer(in_dim * (num_levels-start_level+1))
        elif self.multi_gran_opt == 'cat' and not self.keep_init:
            self.norm2 = norm_layer(in_dim * (num_levels -start_level))
            self.mlp = Mlp(in_features=in_dim * (num_levels-start_level), hidden_features=int(in_dim * (num_levels - -start_level) * mlp_ratio),
                           out_features=in_dim * (num_levels-start_level),
                           act_layer=act_layer, drop=drop)
            self.mlp2 = nn.Linear(in_dim * (num_levels -start_level), in_dim)
            self.drop_path2 = nn.Dropout(0.1)
            self.norm3 = norm_layer(in_dim * (num_levels  -start_level))
        else:
            self.norm2 = norm_layer(in_dim)
            self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def reshape_out(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(b, h * w, c)
        return x

    def forward(self, x):
        x_tmp = x.clone()
        B, new_HW, C = x.shape
        x_tmp = x_tmp.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        x_splits = []
        if self.num_levels >1:
            for i in range(self.start_level,self.num_levels):
                split_tmp = self.unfolds[i- self.start_level](x_tmp).permute(0,2,1)
                split_tmp = self.pos_module[i- self.start_level](split_tmp)
                # print('split_tmp',split_tmp.shape)
                attntion_tmp = self.attentions[i- self.start_level](self.norms[i- self.start_level](split_tmp))
                attntion_tmp = attntion_tmp.permute(0,2,1)
                fold_tmp = self.folds[i- self.start_level](attntion_tmp)
                x_splits.append(self.reshape_out(fold_tmp))

        if self.keep_init:
            x = self.init_posemb(x)
            x = self.init_attn(self.norm1(x))
        if len(x_splits) > 0:
            if self.multi_gran_opt == 'add':
                cat_tmp = torch.stack(x_splits, dim=0)
                split_result = torch.sum(cat_tmp,dim=0)
                if not self.keep_init:
                    x = split_result
                else:
                    x += split_result

            elif self.multi_gran_opt == 'cat':
                if not self.keep_init:
                    x = x_splits[0]
                if len(x_splits) >= 1:
                    for split in x_splits[1:]:
                        x = torch.cat((x,split),dim=2)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.multi_gran_opt == "cat":
            x = self.drop_path2(self.mlp2(self.norm3(x)))
        return x
# e = 0
# a = torch.rand((2,3))
# print(a)
# b = torch.rand((2,3))
# g = torch.rand((2,3))
# print(b)
# print(g)
# c = torch.stack([a,b,g],dim=0)
# d = torch.cat((a,b),dim=1)
# print(d,d.shape)
# print(c,c.shape)
# f = torch.sum(c,dim=0)
# print(f)
# c = c.view(2,6)
# print(c)


class Multi_Gran_Transformer(nn.Module):
    """
    先做不同尺度的unfold，然后做pool成不同尺度的特征进行堆积成为key和value
    """
    def __init__(self, dim, in_dim, num_heads,num_levels,input_resolution, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, sr_ratio=1,act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.unfolds = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.folds = nn.ModuleList()
        self.attentions +=[
            Attention(
                dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, sr_ratio=sr_ratio)
        ]
        self.num_levels = num_levels
        for i in range(1,num_levels):
            kerner_size = 2**i
            stride = 2**i
            self.unfolds += [
                nn.Unfold(kernel_size=(kerner_size, kerner_size), stride=stride)
            ]
            self.folds += [
                nn.Fold(output_size=(input_resolution,input_resolution),kernel_size=(kerner_size, kerner_size), stride=stride)
            ]
            self.attentions += [
                Attention(
                    dim*(stride**2), in_dim=in_dim*(stride**2), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                    proj_drop=drop, sr_ratio=sr_ratio)
            ]
            self.norms += [
                norm_layer(dim*(stride**2))
            ]
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.attn = Attention(
            dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def reshape_out(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(b, h * w, c)
        return x

    def forward(self, x):
        x_tmp = x.clone()
        B, new_HW, C = x.shape
        x_tmp = x_tmp.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        x_splits = []
        if self.num_levels >1:
            for i in range(1,self.num_levels):
                split_tmp = self.unfolds[i-1](x_tmp).permute(0,2,1)
                # print('split_tmp',split_tmp.shape)
                attntion_tmp = self.attentions[i](self.norms[i-1](split_tmp))
                attntion_tmp = attntion_tmp.permute(0,2,1)
                x_splits.append(self.folds[i-1](attntion_tmp))
        x = self.attn(self.norm1(x))
        # print('x.shape',x.shape)
        if len(x_splits) > 0:
            for split in x_splits:
                x += self.reshape_out(split)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# model = Multi_Gran_Transformer_posed(dim=16,in_dim=16,num_heads=4,num_levels=3,input_resolution=128)
# data = torch.rand((1,128*128,16)
#                   )
# out = model(data)
# print(out.shape)

class transformer_bloack(nn.Module):

    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # print('xshape',x.shape)
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class transformers(nn.Module):
    def __init__(self,num_layers,dim, in_dim, num_heads, mlp_ratio=1):
        super().__init__()
        self.num_layer = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers-1):
            block = transformer_bloack(dim,dim,num_heads,mlp_ratio,)
            self.layers.append(copy.deepcopy(block))
        self.last_layer = transformer_bloack(dim,in_dim,num_heads,mlp_ratio,)

    def forward(self,x ):
        if self.num_layer > 1 :
            for block in self.layers:
                x = block(x)

        x = self.last_layer(x)
        return x

class MultiGran_transformers(nn.Module):
    def __init__(self,num_layers,dim, in_dim, num_heads, num_levels,input_resolution,keep_init=True,mlp_ratio=1):
        super().__init__()
        self.num_layer = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers-1):
            block = Multi_Gran_Transformer_improved(dim,dim,num_heads,num_levels,input_resolution,keep_init=keep_init,mlp_ratio=mlp_ratio,)
            self.layers.append(copy.deepcopy(block))
        self.last_layer = Multi_Gran_Transformer_improved(dim,in_dim,num_heads,num_levels,input_resolution,keep_init=keep_init,mlp_ratio=mlp_ratio)

    def forward(self,x ):
        if self.num_layer > 1 :
            for block in self.layers:
                x = block(x)

        x = self.last_layer(x)
        return x


class Multi_Gran_Transformer_adapGran(nn.Module):
    """
    先做不同尺度的unfold，然后做pool成不同尺度的特征进行堆积成为key和value
    """
    def __init__(self, dim, in_dim, num_heads,num_levels,input_resolution,start_level=1, keep_init=True,multi_gran_opt='add',mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, sr_ratio=1,act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.unfolds = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.keep_init = keep_init
        self.multi_gran_opt = multi_gran_opt
        self.norms = nn.ModuleList()
        self.folds = nn.ModuleList()
        # print('heads:...',in_dim,num_heads,type(in_dim),type(num_heads))
        if keep_init:
            self.init_attn = Attention(
                dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, sr_ratio=sr_ratio)
        self.num_levels = num_levels
        self.start_level = start_level
        for i in range(start_level,num_levels):
            kerner_size = 2**i
            stride = 2**i
            self.unfolds += [
                nn.Unfold(kernel_size=(kerner_size, kerner_size), stride=stride)
            ]
            self.folds += [
                nn.Fold(output_size=(input_resolution,input_resolution),kernel_size=(kerner_size, kerner_size), stride=stride)
            ]
            self.attentions += [
                Attention(
                    dim*(stride**2), in_dim=in_dim*(stride**2), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                    proj_drop=drop, sr_ratio=sr_ratio)
            ]
            self.norms += [
                norm_layer(dim*(stride**2))
            ]
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        if self.multi_gran_opt == 'cat' and self.keep_init:
            self.norm2 = norm_layer(in_dim*(num_levels-start_level+1))
            self.mlp = Mlp(in_features=in_dim*(num_levels-start_level+1), hidden_features=int(in_dim *(num_levels)* mlp_ratio),
                           out_features=in_dim*(num_levels-start_level+1),
                           act_layer=act_layer, drop=drop)
            self.mlp2 = nn.Linear(in_dim*(num_levels-start_level+1),in_dim)
            self.drop_path2 = nn.Dropout(0.1)
            self.norm3 = norm_layer(in_dim*(num_levels-start_level+1))
        elif self.multi_gran_opt == 'cat' and not self.keep_init:
            self.norm2 = norm_layer(in_dim * (num_levels-start_level))
            self.mlp = Mlp(in_features=in_dim * (num_levels-start_level), hidden_features=int(in_dim*(num_levels-start_level) * mlp_ratio),
                           out_features=in_dim*(num_levels-start_level),
                           act_layer=act_layer, drop=drop)
            self.mlp2 = nn.Linear(in_dim * (num_levels-start_level), in_dim)
            self.drop_path2 = nn.Dropout(0.1)
            self.norm3 = norm_layer(in_dim * (num_levels-start_level))
        else:
            self.norm2 = norm_layer(in_dim)
            self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def reshape_out(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(b, h * w, c)
        return x

    # @get_local("attention_tmp")
    def forward(self, x):
        x_tmp = x.clone()
        B, new_HW, C = x.shape
        x_tmp = x_tmp.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        x_splits = []
        if self.num_levels >1:
            for i in range(self.start_level,self.num_levels):
                split_tmp = self.unfolds[i-self.start_level](x_tmp).permute(0,2,1)
                # print('split_tmp',split_tmp.shape)
                attntion_tmp = self.attentions[i-self.start_level](self.norms[i-self.start_level](split_tmp))
                attntion_tmp = attntion_tmp.permute(0,2,1)
                fold_tmp = self.folds[i-self.start_level](attntion_tmp)
                x_splits.append(self.reshape_out(fold_tmp))

        if self.keep_init:
            x = self.init_attn(self.norm1(x))
            # attention_tmp = x
        if len(x_splits) > 0:
            if self.multi_gran_opt == 'add':
                cat_tmp = torch.stack(x_splits, dim=0)
                split_result = torch.sum(cat_tmp,dim=0)
                if not self.keep_init:
                    x = split_result
                else:
                    x += split_result

            elif self.multi_gran_opt == 'cat':
                if not self.keep_init:
                    x = x_splits[0]
                else:
                    x = torch.cat((x, x_splits[0]), dim=2)
                if len(x_splits) > 1:
                    for split in x_splits[1:]:
                        x = torch.cat((x,split),dim=2)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.multi_gran_opt == "cat":
            x = self.drop_path2(self.mlp2(self.norm3(x)))
        return x



# model = Multi_Gran_Cross_Transformer(dim=16,ref_dim=256,input_resolution=32,num_levels=3,num_heads=8,sr_ratio=2,mlp_ratio=4)
# data = torch.rand(4,1024,16)
# ref = torch.rand(4,14*14,256)
# out = model(data,ref)
# print(out.shape)

# x = torch.arange(1,10).view(3,3)
# y = torch.arange(10,19).view(3,3)
# res = [x,y]
# res = torch.cat(res,dim=1)
# print(x)
# print(y)
# print(res)