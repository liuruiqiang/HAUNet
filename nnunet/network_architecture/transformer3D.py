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
from nnunet.network_architecture.transformer_block import Mlp,position_enc,\
    learned_position_enc,position_embedding
import numpy as np
import torch.nn.functional as F
# F.unfold()
import torch

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
        attn = attn.softmax(dim=-1)
        # print('attn shape',attn.shape)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.sr_ratio == 1:
            v = v.transpose(1,2).reshape(B, N, self.in_dim)

            x = v.squeeze(1) + x   # because the original x has different size with current x, use v to do skip connection

        return x

class extract_patch_3D(nn.Module):
    def __init__(self,kernel_size,stride,padding=0,dilation=1):
        super().__init__()
        if isinstance(kernel_size,int):
            self.kernel_size = (kernel_size,kernel_size,kernel_size)
        if isinstance(stride,int):
            self.stride = (stride,stride,stride)
        if isinstance(padding,int):
            self.padding = (padding,padding,padding)
        if isinstance(dilation,int):
            self.dilation = (dilation,dilation,dilation)

    def get_dim_blocks(self,dim_in, dim_kernel_size, dim_padding=0, dim_stride=1, dim_dilation=1):
        dim_out = (dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1) // dim_stride + 1
        return dim_out

    def forward(self,x):

        channels =x.shape[1]

        d_dim_in = x.shape[2]
        h_dim_in = x.shape[3]
        w_dim_in = x.shape[4]
        d_dim_out = self.get_dim_blocks(d_dim_in, self.kernel_size[0], self.padding[0], self.stride[0], self.dilation[0])
        h_dim_out = self.get_dim_blocks(h_dim_in, self.kernel_size[1], self.padding[1], self.stride[1], self.dilation[1])
        w_dim_out = self.get_dim_blocks(w_dim_in, self.kernel_size[2], self.padding[2], self.stride[2], self.dilation[2])
        # print(d_dim_in, h_dim_in, w_dim_in, d_dim_out, h_dim_out, w_dim_out)

        # (B, C, D, H, W)
        x = x.view(-1, channels, d_dim_in, h_dim_in * w_dim_in)
        # (B, C, D, H * W)

        x = torch.nn.functional.unfold(x, kernel_size=(self.kernel_size[0], 1), padding=(self.padding[0], 0), stride=(self.stride[0], 1), dilation=(self.dilation[0], 1))
        # (B, C * kernel_size[0], d_dim_out * H * W)

        x = x.view(-1, channels * self.kernel_size[0] * d_dim_out, h_dim_in, w_dim_in)
        # (B, C * kernel_size[0] * d_dim_out, H, W)

        x = torch.nn.functional.unfold(x, kernel_size=(self.kernel_size[1], self.kernel_size[2]), padding=(self.padding[1], self.padding[2]), stride=(self.stride[1], self.stride[2]), dilation=(self.dilation[1], self.dilation[2]))
        # (B, C * kernel_size[0] * d_dim_out * kernel_size[1] * kernel_size[2], h_dim_out, w_dim_out)

        x = x.view(-1, channels, self.kernel_size[0], d_dim_out, self.kernel_size[1], self.kernel_size[2], h_dim_out, w_dim_out)
        # (B, C, kernel_size[0], d_dim_out, kernel_size[1], kernel_size[2], h_dim_out, w_dim_out)

        x = x.permute(0, 1, 3, 6, 7, 2, 4, 5)
        # x = x.permute(0,1,2,4,5,3,6,7)
        # (B, C,kernel_size[0], kernel_size[1], kernel_size[2], d_dim_out, h_dim_out, w_dim_out)
        # (B, C, d_dim_out, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1], kernel_size[2])

        x = x.contiguous().view(-1, channels, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2])
        # x = x.contiguous().view(B,channels*self.kernel_size[0]*self.kernel_size[1]*self.kernel_size[2],
        #                         d_dim_out*h_dim_out*w_dim_out)
        # (B * d_dim_out * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1], kernel_size[2])

        return x

class combine_patch_3D(nn.Module):
    def __init__(self,kernel_size,output_shape,stride,padding=0,dilation=1):
        super().__init__()
        if isinstance(kernel_size,int):
            self.kernel_size = (kernel_size,kernel_size,kernel_size)
        if isinstance(stride,int):
            self.stride = (stride,stride,stride)
        if isinstance(padding,int):
            self.padding = (padding,padding,padding)
        if isinstance(dilation,int):
            self.dilation = (dilation,dilation,dilation)
        if isinstance(output_shape, int):
            self.output_shape = (output_shape,output_shape,output_shape)
        else:
            self.output_shape = output_shape

    def get_dim_blocks(self,dim_in, dim_kernel_size, dim_padding=0, dim_stride=1, dim_dilation=1):
        dim_out = (dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1) // dim_stride + 1
        return dim_out

    def forward(self,x):

        channels = x.shape[1]

        d_dim_out, h_dim_out, w_dim_out = self.output_shape
        d_dim_in = self.get_dim_blocks(d_dim_out, self.kernel_size[0], self.padding[0], self.stride[0], self.dilation[0])
        h_dim_in = self.get_dim_blocks(h_dim_out, self.kernel_size[1], self.padding[1], self.stride[1], self.dilation[1])
        w_dim_in = self.get_dim_blocks(w_dim_out, self.kernel_size[2], self.padding[2], self.stride[2], self.dilation[2])
        # print(d_dim_in, h_dim_in, w_dim_in, d_dim_out, h_dim_out, w_dim_out)

        # (B, C, D, H, W)
        x = x.view(-1, channels, d_dim_in, h_dim_in, w_dim_in, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2])
        # (B, C, d_dim_in, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1], kernel_size[2])

        x = x.permute(0, 1, 5, 2, 6, 7, 3, 4)
        # (B, C, kernel_size[0], d_dim_in, kernel_size[1], kernel_size[2], h_dim_in, w_dim_in)

        x = x.contiguous().view(-1, channels * self.kernel_size[0] * d_dim_in *self. kernel_size[1] * self.kernel_size[2], h_dim_in * w_dim_in)
        # (B, C * kernel_size[0] * d_dim_in * kernel_size[1] * kernel_size[2], h_dim_in * w_dim_in)

        x = torch.nn.functional.fold(x, output_size=(h_dim_out, w_dim_out), kernel_size=(self.kernel_size[1], self.kernel_size[2]), padding=(self.padding[1], self.padding[2]), stride=(self.stride[1], self.stride[2]),
                                     dilation=(self.dilation[1], self.dilation[2]))
        # (B, C * kernel_size[0] * d_dim_in, H, W)

        x = x.view(-1, channels * self.kernel_size[0], d_dim_in * h_dim_out * w_dim_out)
        # (B, C * kernel_size[0], d_dim_in * H * W)

        x = torch.nn.functional.fold(x, output_size=(d_dim_out, h_dim_out * w_dim_out), kernel_size=(self.kernel_size[0], 1), padding=(self.padding[0], 0), stride=(self.stride[0], 1), dilation=(self.dilation[0], 1))
        # (B, C, D, H * W)

        x = x.view(-1, channels, d_dim_out, h_dim_out, w_dim_out)

        return x


class extract_patch_2D(nn.Module):
    def __init__(self,kernel_size,stride,padding=0,dilation=1):
        super().__init__()
        if isinstance(kernel_size,int):
            self.kernel_size = (kernel_size,kernel_size)
        if isinstance(stride,int):
            self.stride = (stride,stride)
        if isinstance(padding,int):
            self.padding = (padding,padding)
        if isinstance(dilation,int):
            self.dilation = (dilation,dilation)

    def get_dim_blocks(self,dim_in, dim_kernel_size, dim_padding=0, dim_stride=1, dim_dilation=1):
        dim_out = (dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1) // dim_stride + 1
        return dim_out

    def forward(self,x):

        channels =x.shape[1]
        h_dim_in = x.shape[2]
        w_dim_in = x.shape[3]
        h_dim_out = self.get_dim_blocks(h_dim_in, self.kernel_size[0], self.padding[0], self.stride[0], self.dilation[0])
        w_dim_out = self.get_dim_blocks(w_dim_in, self.kernel_size[1], self.padding[1], self.stride[1], self.dilation[1])

        # (B, C, H, W)
        x = torch.nn.functional.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride, dilation=self.dilation)
        # (B, C * kernel_size[0] * kernel_size[1], h_dim_out * w_dim_out)
        x = x.view(-1, channels, self.kernel_size[0], self.kernel_size[1], h_dim_out, w_dim_out)
        # (B, C, kernel_size[0], kernel_size[1], h_dim_out, w_dim_out)
        x = x.permute(0, 1, 4, 5, 2, 3)
        # (B, C, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1])
        x = x.contiguous().view(-1, channels, self.kernel_size[0], self.kernel_size[1])
        # (B * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1])
        return x

class combine_patch_2D(nn.Module):
    def __init__(self,kernel_size,output_shape,stride,padding=0,dilation=1):
        super().__init__()
        if isinstance(kernel_size,int):
            self.kernel_size = (kernel_size,kernel_size)
        if isinstance(stride,int):
            self.stride = (stride,stride)
        if isinstance(padding,int):
            self.padding = (padding,padding)
        if isinstance(dilation,int):
            self.dilation = (dilation,dilation)
        if isinstance(output_shape, int):
            self.output_shape = (output_shape,output_shape)
        else:
            self.output_shape = output_shape

    def get_dim_blocks(self,dim_in, dim_kernel_size, dim_padding=0, dim_stride=1, dim_dilation=1):
        dim_out = (dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1) // dim_stride + 1
        return dim_out

    def forward(self,x):

        channels = x.shape[1]
        h_dim_out, w_dim_out = self.output_shape[2:]
        h_dim_in = self.get_dim_blocks(h_dim_out, self.kernel_size[0], self.padding[0], self.stride[0], self.dilation[0])
        w_dim_in = self.get_dim_blocks(w_dim_out, self.kernel_size[1], self.padding[1], self.stride[1], self.dilation[1])
        x = x.view(-1, channels, h_dim_in, w_dim_in, self.kernel_size[0], self.kernel_size[1])
        # (B, C, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1])
        x = x.permute(0, 1, 4, 5, 2, 3)
        # (B, C, kernel_size[0], kernel_size[1], h_dim_in, w_dim_in)
        x = x.contiguous().view(-1, channels * self.kernel_size[0] * self.kernel_size[1], h_dim_in * w_dim_in)
        # (B, C * kernel_size[0] * kernel_size[1], h_dim_in * w_dim_in)
        x = torch.nn.functional.fold(x, (h_dim_out, w_dim_out), kernel_size=(self.kernel_size[0], self.kernel_size[1]), padding=self.padding, stride=self.stride, dilation=self.dilation)
        # (B, C, H, W)
        return x

# a = torch.arange(1, 65, dtype=torch.float).view(2,2,4,4)
# print(a.shape)
# print(a)
# extrac_func = extract_patch_2D(2, padding=1, stride=2, dilation=1)
# b = extrac_func(a)
# # b = extract_patches_2ds(a, 2, padding=1, stride=2)
# print(b.shape)
# print(b)
# combine_func = combine_patch_2D( 2, (2,2,4,4), padding=1, stride=2, dilation=1)
# c = combine_func(b)
# print(c.shape)
# print(c)
# print(torch.all(a==c))
# a = torch.arange(1, 129, dtype=torch.float).view(2,2,2,4,4)
# print(a.shape)
# print(a)
# # b = extract_patches_3d(a, 2, padding=1, stride=2)
# extract_func = extract_patch_3D(2, padding=1, stride=2)
# b = extract_func(a)
# print(b.shape)
# print(b)
# combine_func = combine_patch_3D(2, (2,2,2,4,4), padding=1, stride=2)
# c = combine_func(b)
# print(c.shape)
# print(c)
# print(torch.all(a==c))

# map = torch.rand((2,64,48,96,96)
#                  )
# extract_func = extract_patch_3D(2, padding=0, stride=2)
# b = extract_func(map)
# print(b.shape,24*48*48)
# b_attn = b.view(1,16*16*16,2*2*2*2)
# b_attn = b.view(1*16*16*16,2,2,2,2)
# combine_func = combine_patch_3D(2, (1,2,32,32,32), padding=0, stride=2)
# c = combine_func(b_attn)
# print(c.shape)
# print(torch.all(c==map))
# map = torch.rand((1,2,224,224)
#                  )
# unfold_result = nn.Unfold(kernel_size=(8,8),padding=1,stride=6)(map)
# print('unfold:',unfold_result.shape)
# # unfold_result = unfold_result.permute(0,2,1)
# fold_result = nn.Fold(output_size=(224,224),kernel_size=(8,8),padding=1,stride=6)(unfold_result)
# print(fold_result.shape)
# print(torch.all(map==fold_result))


class Token_transformer(nn.Module):

    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, sr_ratio=1,act_layer=nn.ReLU,input_resolution=(48,192,192), norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        t,h,w = input_resolution
        self.position_embedding = position_embedding(t*h*w,dim,pos_type="learned")
        self.input_resolution = input_resolution
        self.attn = Attention(
            dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,sr_ratio=sr_ratio)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def reshape_out(self,x):
        b, c, t,h, w = x.shape
        # print(h,w,t,h*w*t)
        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape(b, h * w * t, c)
        return x

    def reshape_back(self,x):
        b,n,c = x.shape
        x = x.permute(0,2,1)
        x = x.reshape((b,c,)+self.input_resolution)
        return x

    def forward(self, x):
        x = self.reshape_out(x)
        x = self.position_embedding(x)
        # print('xshape',x.shape)
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.reshape_back(x)
        return x

# data = torch.rand(2,320,6,12,12).cuda()
# model = Token_transformer(dim=320,in_dim=320,num_heads=8,input_resolution=(6,12,12))
# model.cuda()
# out = model(data)
# print(out.shape)
# print(tuple([1,2,3]))

class Efficient_transformer(nn.Module):

    def __init__(self, dim, in_dim, num_heads,unfold_level, input_resolution,mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, sr_ratio=1,act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        self.unfold_level = unfold_level
        self.input_resolution = input_resolution
        t, h, w = input_resolution
        kerner_size = 2 ** unfold_level
        self.position_embedding = position_embedding(t * h * w//(kerner_size**3) , dim*kerner_size**3, pos_type="learned")
        stride = 2 ** unfold_level
        self.norm1 = norm_layer(dim*(stride**3))
        self.unfolds = extract_patch_3D(kernel_size=kerner_size, stride=stride)
        self.folds = combine_patch_3D(kernel_size=kerner_size,output_shape=input_resolution, stride=stride)
        self.attn = Attention(
                    dim*(stride**3), in_dim=in_dim*(stride**3), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                    proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def reshape_out(self, x):
        b, c, t, h, w = x.shape
        # print(h, w, t, h * w * t)
        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape(b, h * w * t, c)
        return x

    def reshape_back(self, x):
        b, n, c = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape((b, c,) + self.input_resolution)
        return x

    def forward(self, x):
        # print('xshape',x.shape)
        # B, new_HW, C = x.shape
        b,c,t,h,w = x.shape
        # x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        split_tmp = self.unfolds(x)
        split_tmp = split_tmp.view(b, split_tmp.shape[0] // b, c * self.unfolds.kernel_size[0] *
                                   self.unfolds.kernel_size[1] * self.unfolds.kernel_size[2])
        # print('split_tmp',split_tmp.shape)
        # print('split_tmp',split_tmp.shape)
        split_tmp = self.position_embedding(split_tmp)
        attntion_tmp = self.attn(self.norm1(split_tmp))
        attntion_tmp = attntion_tmp.view(-1, c, self.unfolds.kernel_size[0],
                                         self.unfolds.kernel_size[1], self.unfolds.kernel_size[2])

        # attntion_tmp = attntion_tmp.permute(0, 2, 1)
        x=self.folds(attntion_tmp)
        x = self.reshape_out(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.reshape_back(x)
        return x

# data = torch.rand(2,160,48,24,24).cuda()
# model = Efficient_transformer(dim=160,in_dim=160,num_heads=8,unfold_level=2,input_resolution=(48,24,24))
# model.cuda()
# out = model(data)
# print(out.shape)


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
        # print(self.multi_gran_opt)
        # print('heads:...',in_dim,num_heads,type(in_dim),type(num_heads))
        if keep_init:
            self.init_attn = Attention(
                dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, sr_ratio=sr_ratio)
        self.num_levels = num_levels
        for i in range(1,num_levels):
            kerner_size = 2**i
            stride = 2**i
            self.unfolds += [
                extract_patch_3D(kernel_size=kerner_size, stride=stride)
            ]
            self.folds += [
                combine_patch_3D(kernel_size=kerner_size,output_shape=input_resolution, stride=stride)
            ]
            self.attentions += [
                Attention(
                    dim*(stride**3), in_dim=in_dim*(stride**3), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                    proj_drop=drop, sr_ratio=sr_ratio)
            ]
            self.norms += [
                norm_layer(dim*(stride**3))
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
        b, c, h, w,z = x.shape
        x = x.permute(0, 2, 3,4, 1)
        x = x.reshape(b, h * w*z, c)
        return x

    def forward(self, x,shape):
        # print('xshape',x.shape)
        x_tmp = x.clone()
        x_tmp = x_tmp.transpose(1, 2).reshape(shape)
        x_splits = []
        b,c,d,h,w = x_tmp.shape
        if self.num_levels >1:
            for i in range(1,self.num_levels):
                split_tmp = self.unfolds[i-1](x_tmp)
                split_tmp = split_tmp.view(b,split_tmp.shape[0]//b,c*self.unfolds[i-1].kernel_size[0]*
                                           self.unfolds[i-1].kernel_size[1]*self.unfolds[i-1].kernel_size[2])
                # print('split_tmp',split_tmp.shape)
                attntion_tmp = self.attentions[i-1](self.norms[i-1](split_tmp))
                attntion_tmp = attntion_tmp.view(-1,c,self.unfolds[i-1].kernel_size[0],
                                           self.unfolds[i-1].kernel_size[1],self.unfolds[i-1].kernel_size[2])
                fold_tmp = self.folds[i-1](attntion_tmp)
                x_splits.append(self.reshape_out(fold_tmp))

        if self.keep_init:
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
                split_cat = torch.cat(x_splits, dim=2)
                if not self.keep_init:
                    x = split_cat
                else:
                    x = torch.cat((x, split_cat), dim=2)
        # print(x.shape,self.drop_path(self.mlp(self.norm2(x))).shape)
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.multi_gran_opt == "cat":
            x = self.drop_path2(self.mlp2(self.norm3(x)))
        return x


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
                torch.nn.
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
            self.norm2 = norm_layer(in_dim*(num_levels))
            self.mlp = Mlp(in_features=in_dim*(num_levels), hidden_features=int(in_dim * mlp_ratio), out_features=in_dim,
                           act_layer=act_layer, drop=drop)
        elif self.multi_gran_opt == 'cat' and not self.keep_init:
            self.norm2 = norm_layer(in_dim * (num_levels-1))
            self.mlp = Mlp(in_features=in_dim * (num_levels-1), hidden_features=int(in_dim * mlp_ratio),
                           out_features=in_dim,
                           act_layer=act_layer, drop=drop)
        else:
            self.norm2 = norm_layer(in_dim)
            self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def reshape_out(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(b, h * w, c)
        return x

    def forward(self, x,shape):
        x_tmp = x.clone()
        B, new_HW, C = x.shape
        # hwz = round(new_HW**(1/3))
        x_tmp = x_tmp.transpose(1, 2).reshape(shape)
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
                x += split_result

            elif self.multi_gran_opt == 'cat':
                if not self.keep_init:
                    x = x_splits[0]
                if len(x_splits) >= 1:
                    for split in x_splits[1:]:
                        x = torch.cat((x,split),dim=2)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
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
# a = (48,192,129)
# print((1,2,)+a)
# model = Token_transformer(dim=16,in_dim=32,num_heads=8,sr_ratio=2,mlp_ratio=4)
# data = torch.rand(4,1024,16)
# out = model(data)
# print(out.shape)
# a = 28*28*28
# b = a**(1/3)
# print(round(b))