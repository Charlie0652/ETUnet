import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Callable, Tuple, Optional, Set, List, Union
from skimage import data, filters
import torch.utils.checkpoint as checkpoint
import numpy as np
from thop import profile
__all__ = ['ETUNET']
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from functools import partial
import math



class DecoderBlock(nn.Module):
    def __init__(self, dim):
        super(DecoderBlock, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.Bn = nn.InstanceNorm2d(dim)
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        input = x
        x1 = self.dwconv(x)
        x2 = self.Bn(x1)
        x3 = self.relu(x2)
        x4 = self.dwconv(x3)
        x5 = self.Bn(x4)
        x6 = self.relu(x5)
        out = x6 + input
        return out

class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel//2, 3, 1, 1)
    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        out1 = self.layer(up)
        out2 = torch.cat((out1, feature_map), 1)
        out3 = self.layer(out2)
        return out3

class Up(nn.Module):
    def __init__(self, channel, out_c):
        super(Up, self).__init__()
        self.layer = nn.Conv2d(channel, out_c, 3, 1, 1)
        self.d = nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, padding=1)
    def forward(self, x, feature_map):
        x1 = F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        x1 = self.d(x1)
        x2 = self.d(feature_map)
        out2 = torch.cat((x1, x2), 1)
        out3 = self.layer(out2)
        return out3
class OutSample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(OutSample, self).__init__()
        # self.up = nn.ConvTranspose2d(in_dim, in_dim//2, 2, 2)
        self.layer = nn.Conv2d(in_dim, out_dim, 1, 1)
    def forward(self, x):
        # x1 = F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        # out = self.up(x)
        return self.layer(x)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class PWTConv2d(nn.Module):
    def __init__(self, pdc, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(PWTConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        # self.pdc = pdc
        self.pdc = createConvFunc(pdc)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return self.pdc(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

## sx, sy convolutions
def createConvFunc(op_type):
    assert op_type in ['sx', 'sy'], 'unknown op type: %s' % str(op_type)
    if op_type == 'sx':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding == dilation, 'padding for ad_conv set wrong'

            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = (weights - weights[:, :, [2, 1, 0, 5, 4, 3, 8, 7, 6]]).view(shape)  # clock-wise
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    elif op_type == 'sy':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding == dilation, 'padding for ad_conv set wrong'

            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = (weights - weights[:, :, [6, 7, 8, 3, 4, 5, 0, 1, 2]]).view(shape)  # clock-wise
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    else:
        print('impossible to be here unless you force that')
        return None

class prewitconv(nn.Module):
    def __init__(self, sx, sy, indim, outdim):
        super(prewitconv, self).__init__()
        self.pwtx = PWTConv2d(sx, indim, outdim, kernel_size=3, padding=1, groups=1, bias=False)
        self.pwty = PWTConv2d(sy, indim, outdim, kernel_size=3, padding=1, groups=1, bias=False)
        self.conv = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, groups=outdim)
        self.con = nn.Conv2d(indim, outdim, kernel_size=1, padding=0)
        self.bn = nn.InstanceNorm2d(indim)
        self.relu = nn.LeakyReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.pwtx(x)
        x3 = self.pwty(x)
        x4 = x2 + x3
        x5 = self.sig(self.con(x4)) * x1
        x6 = x1 + x5
        x7 = self.relu(self.bn(self.conv(x6)))
        return x7

class down(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(down, self).__init__()
        self.proj = nn.Conv2d(in_chan, out_chan, kernel_size=2, stride=2)
        self.Bn = LayerNorm(out_chan, eps=1e-6, data_format="channels_first")
    def forward(self, x):
        x1 = self.proj(x)
        x2 = self.Bn(x1)
        return x2

class ConvPatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        kernel_size = to_2tuple(patch_size[0])
        print(f"ConvPatchEmbed: @input_channels: {in_chans}, output_channel: {embed_dim},\
             @kernel_size: {kernel_size}, @stride: {patch_size}")
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size, patch_size)
        self.norm = norm_layer(embed_dim)
        # self.fg = nn.Conv2d(embed_dim, embed_dim// 2, kernel_size=3, padding=1)
        self.Bn = LayerNorm(embed_dim, eps=1e-6,  data_format="channels_first")


    def forward(self, x):
        B, C, H, W = x.shape
        x2 = self.proj(x)
        x_GT = x2.flatten(2).transpose(1, 2)
        x_GT = self.norm(x_GT)
        x_LT = self.Bn(x2)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x_GT, x_LT, H, W


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = in_features
        hidden_features = hidden_features
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

class CMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x

def grid_partition(
        input: torch.Tensor,
        grid_size: Tuple[int, int] = (4, 4)
) -> torch.Tensor:
    # Get size of input
    B, C, H, W = input.shape
    # Unfold input
    grid = input.view(B, C, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1])
    # Permute and reshape [B * (H // grid_size[0]) * (W // grid_size[1]), grid_size[0], window_size[1], C]
    grid = grid.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return grid


def grid_reverse(
        grid: torch.Tensor,
        original_size: Tuple[int, int],
        grid_size: Tuple[int, int] = (4, 4)
) -> torch.Tensor:
    # Get height, width, and channels
    (H, W), C = original_size, grid.shape[-1]
    # Compute original batch size
    B = int(grid.shape[0] / (H * W / grid_size[0] / grid_size[1]))
    # Fold grid tensor
    output = grid.view(B, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    output = output.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, C, H, W)
    return output

def get_relative_position_index(
        win_h: int,
        win_w: int
) -> torch.Tensor:
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)


class RelativeSelfAttention(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_heads: int = 16,
            grid_window_size: Tuple[int, int] = (2, 2),
            attn_drop: float = 0.,
            drop: float = 0.
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(RelativeSelfAttention, self).__init__()
        # Save parameters
        self.in_channels: int = in_channels
        self.num_heads: int = num_heads
        self.grid_window_size: Tuple[int, int] = grid_window_size
        self.scale: float = num_heads ** -0.5
        self.attn_area: int = grid_window_size[0] * grid_window_size[1]
        # Init layers
        self.qkv_mapping = nn.Linear(in_features=in_channels, out_features=3 * in_channels, bias=True)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.proj_drop = nn.Dropout(p=drop)
        self.softmax = nn.Softmax(dim=-1)
        # Define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * grid_window_size[0] - 1) * (2 * grid_window_size[1] - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(grid_window_size[0],
                                                                                    grid_window_size[1]))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(
            self,
            input: torch.Tensor
    ) -> torch.Tensor:
        # Get shape of input
        B_, N, C = input.shape
        # Perform query key value mapping
        qkv = self.qkv_mapping(input).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # Scale query
        q = q * self.scale
        # Compute attention maps
        attn = self.softmax(q @ k.transpose(-2, -1) + self._get_relative_positional_bias())
        # Map value with attention maps
        output = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        # Perform final projection and dropout
        output = self.proj(output)
        output = self.proj_drop(output)
        return output

class MaxViTTransformerBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            partition_function: Callable,
            reverse_function: Callable,
            num_heads: int = 16,
            grid_window_size: Tuple[int, int] = (2, 2),
            attn_drop: float = 0.,
            drop: float = 0.,
            drop_path: float = 0.,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        """ Constructor method """
        super(MaxViTTransformerBlock, self).__init__()
        # Save parameters
        self.partition_function: Callable = partition_function
        self.reverse_function: Callable = reverse_function
        self.grid_window_size: Tuple[int, int] = grid_window_size
        # Init layers
        self.norm_1 = norm_layer(in_channels)
        self.attention = RelativeSelfAttention(
            in_channels=in_channels,
            num_heads=num_heads,
            grid_window_size=grid_window_size,
            attn_drop=attn_drop,
            drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_2 = norm_layer(in_channels)
        self.mlp = Mlp(
            in_features=in_channels,
            hidden_features=int(mlp_ratio * in_channels),
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Save original shape
        B, C, H, W = input.shape
        # Perform partition
        input_partitioned = self.partition_function(input, self.grid_window_size)
        input_partitioned = input_partitioned.view(-1, self.grid_window_size[0] * self.grid_window_size[1], C)
        # Perform normalization, attention, and dropout
        output = input_partitioned + self.drop_path(self.attention(self.norm_1(input_partitioned)))
        # Perform normalization, MLP, and dropout
        output = output + self.drop_path(self.mlp(self.norm_2(output)))
        # Reverse partition
        output = self.reverse_function(output, (H, W), self.grid_window_size)
        return output


class MaxViTBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_heads: int = 16,
            grid_window_size: Tuple[int, int] = (2, 2),
            attn_drop: float = 0.,
            drop: float = 0.,
            drop_path: float = 0.,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer_transformer: Type[nn.Module] = nn.LayerNorm
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(MaxViTBlock, self).__init__()
        self.grid_transformer = MaxViTTransformerBlock(
            in_channels=in_channels,
            partition_function=grid_partition,
            reverse_function=grid_reverse,
            num_heads=num_heads,
            grid_window_size=grid_window_size,
            attn_drop=attn_drop,
            drop=drop,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer_transformer
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.grid_transformer(input)
        return output

class LA(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, c_ratio=1):
        super(LA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.sr_ratio = sr_ratio
        self.c_new = int(dim * c_ratio)  # scaled channel dimension
        print(f'@ dim: {dim}, dim_new: {self.c_new}, c_ratio: {c_ratio}, sr_ratio: {sr_ratio}\n')

        # self.scale = qk_scale or head_dim ** -0.5
        self.scale = qk_scale or (head_dim * c_ratio) ** -0.5

        if sr_ratio > 1:
            self.q = nn.Linear(dim, self.c_new, bias=qkv_bias)
            self.reduction = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim),
                nn.Conv2d(dim, self.c_new, kernel_size=1, stride=1))
            self.norm_act = nn.Sequential(
                nn.LayerNorm(self.c_new),
                nn.GELU())
            self.k = nn.Linear(self.c_new, self.c_new, bias=qkv_bias)
            self.v = nn.Linear(self.c_new, dim, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.k = nn.Linear(dim, dim, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sig = nn.Sigmoid()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W, y):
        B, N, C = x.shape
        if self.sr_ratio > 1:
            # reduction
            _x = x.permute(0, 2, 1).reshape(B, C, H, W)
            _x = self.reduction(_x).reshape(B, self.c_new, -1).permute(0, 2, 1)  # shape=(B, N', C')
            _x = self.norm_act(_x)
            # q, k, v, where q_shape=(B, N, C'), k_shape=(B, N', C'), v_shape=(N, N', C)
            q = self.q(x).reshape(B, N, self.num_heads, int(self.c_new / self.num_heads)).permute(0, 2, 1, 3)
            k = self.k(_x).reshape(B, -1, self.num_heads, int(self.c_new / self.num_heads)).permute(0, 2, 1, 3)
            v = self.v(_x).reshape(B, -1, self.num_heads, int(C / self.num_heads)).permute(0, 2, 1, 3)
        else:

            # print((",", x.shape))
            y1 = y.flatten(2).transpose(1, 2)
            y2 = self.norm(y1)
            # print((",", y2.shape))
            yq = self.sig(self.q(y2))
            ykv = self.sig(self.kv(y2))
            q0 = self.q(x)
            q1 = q0 * yq
            q2 = q0 + q1
            # q = self.q(x).reshape(B, N, self.num_heads, int(C / self.num_heads)).permute(0, 2, 1, 3)
            q = q2.reshape(B, N, self.num_heads, int(C / self.num_heads)).permute(0, 2, 1, 3)
            kv0 = self.kv(x)
            kv1 = kv0 * ykv
            kv2 = kv1 + ykv
            # kv = self.kv(x).reshape(B, -1, 2, self.num_heads, int(C / self.num_heads)).permute(2, 0, 3, 1, 4)
            kv = kv2.reshape(B, -1, 2, self.num_heads, int(C / self.num_heads)).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, c_ratio=1, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = LA(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            c_ratio=c_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * 4)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W, y):
        # TODO: add scale like PoolFormer
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, y))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim // 2, dim // 2, kernel_size=7, padding=3, groups=dim // 2)
        self.dconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.conv = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, groups=dim // 2)
        self.conv1 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1)

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.Bn = nn.InstanceNorm2d(dim // 2)
        self.relu = nn.LeakyReLU(inplace=True)
        self.L = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.drop = nn.Dropout2d(0)
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax2d()

    def forward(self, x):
        b, c, h, w = x.shape
        m = self.conv2(x)
        x1 = self.relu(self.Bn(self.dwconv(m)))
        x2 = self.relu(self.Bn(self.dwconv(x1)))

        x3 = self.relu(self.Bn(self.conv1(m)))
        x4 = self.relu(self.Bn(self.conv(x3)))

        x5 = torch.cat((x2, x4), dim=1)
        # x5 = x2 + x4
        x6 = self.dconv(x5)
        x7 = x6.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x7 = self.norm(x7)
        x7 = self.pwconv1(x7)
        x7 = self.act(x7)
        x7 = self.pwconv2(x7)
        # x = self.act2(x)
        x7 = x7.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x8 = x6 + x7
        return x8


class LocalTransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.Bn = nn.InstanceNorm2d(dim)
        self.relu = nn.LeakyReLU(inplace=True)
        # self.Bn = nn.BatchNorm2d(dim)
        # self.relu = nn.ReLU(inplace=True)
        self.BN = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.attn = CA(dim)
        self.mlp = CMLP(dim)

    def forward(self, x):
        # TODO: add scale like PoolFormer
        # input = x
        # x = self.relu(self.Bn(self.conv(x)))
        # x = self.relu(self.Bn(self.dwconv(x)))
        # x = input + x
        # x = x + self.attn(self.BN(x))
        # x = x + self.mlp(self.BN(x))
        x = self.attn(x)
        return x


class LMT(nn.Module):
    def __init__(self, dim, num_heads, sr_ratios):
        super().__init__()
        nh = num_heads
        sr_ra = sr_ratios
        self.GT1 = GlobalTransformerBlock(dim=dim, num_heads=nh, qkv_bias=False, drop=0., attn_drop=0.,
                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, c_ratio=1, sr_ratio=sr_ra)
        # self.GT1 = MaxViTBlock(in_channels=dim, num_heads=nh, grid_window_size=(4, 4), drop=0., attn_drop=0.,
        #                       drop_path=0., mlp_ratio=4., act_layer=nn.GELU, norm_layer_transformer=nn.LayerNorm)
        self.GT = MaxViTBlock(in_channels=dim, num_heads=nh, grid_window_size=(2, 2), drop=0., attn_drop=0.,
                 drop_path=0., mlp_ratio=4., act_layer=nn.GELU, norm_layer_transformer=nn.LayerNorm)
        self.LT = LocalTransformerBlock(dim=dim)
        self.conv = nn.Conv2d(dim, dim//2, kernel_size=3, padding=1, groups=dim//2)
        self.con = nn.Conv2d(dim*2, dim, 3, padding=1, groups=dim)
        self.L1 = nn.Conv2d(dim*4, dim, kernel_size=3, padding=1, groups=dim)
        self.L2 = nn.Conv2d(dim, dim//2, kernel_size=3, padding=1, groups=dim//2)
        self.stem1 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
        self.stem2 = nn.Conv2d(dim, dim, kernel_size=2, stride=2, padding=1, dilation=2)
        self.stem3 = nn.MaxPool2d(2)
        self.stem4 = nn.AvgPool2d(2)
        self.Bn = nn.InstanceNorm2d(dim)
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, x1, x11, H, W, Y):
        b, c, h, w = x11.shape
        # h = int(math.sqrt(n))
        # w = h
        x2 = self.GT1(x1, H, W, Y)
        x2 = x2.transpose(2, 1).reshape(b, c, h, w)
        # a = x2
        # print("**", a.shape)
        b1 = self.stem1(x2)
        b2 = self.stem2(x2)
        # print("**", b2.shape)
        b3 = self.stem3(x2)
        b4 = self.stem4(x2)
        c1 = torch.cat((b1, b2), 3)
        c2 = torch.cat((b3, b4), 3)
        c3 = torch.cat((c1, c2), 2)
        x3 = self.GT(c3)
        e, f = x3.chunk(2, dim=2)
        e1, e2 = e.chunk(2, dim=3)
        f1, f2 = f.chunk(2, dim=3)
        g = torch.cat((torch.cat((torch.cat((e1, f1), 1), e2), 1), f2), 1)
        up = F.interpolate(g, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        # print("11", up.shape)
        up1 = self.L1(up)
        # print("&&",up2.shape)
        # up3 = up1 + a
        # x2 = self.L2(up1)
        x22 = self.LT(Y)
        # x22 = self.L2(x22)
        out = torch.cat((up1, x22), 1)
        out = self.relu(self.Bn(self.con(out)))
        return out

def out_maps(x, max_num=30, out_path='E:/lijiawei/ETUnet-main/feature-maps/'):
    for i in range(0, x.shape[1]):
        if i >= max_num:
            break
        feature = x[0, i, :, :].view(x.shape[-2], x.shape[-1])
        feature = feature.cpu().detach().numpy()
        feature = 1.0 / (1 + np.exp(-1 * feature))
        feature = np.round(feature * 255)
        feature1 = feature.astype(np.uint8)
        feature_img = cv2.applyColorMap(feature1, cv2.COLORMAP_JET)
        dst_path = os.path.join(out_path, str(i) + '.png')
        cv2.imwrite(dst_path, feature_img)     # 热力图
        # cv2.imwrite(dst_path, feature1)      # 直接输出特征图


class ETUNET(nn.Module):
    def __init__(
            self,
            img_size=256,
            patch_size=[2, 2, 2, 2],
            in_chans=3,
            num_classes=1,
            dims=[96, 192, 384, 768],
            num_heads=[4, 8, 16, 32],
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            sr_ratios=[1, 1, 1, 1],
            pwt=['sx', 'sy'],
            c_ratios=[1.25, 1.25, 1.25, 1.25],
            plain_head=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, dims[0], kernel_size=3, padding=1)
        self.bn = nn.InstanceNorm2d(dims[0])
        self.relu = nn.LeakyReLU(inplace=True)
        # self.DAM = DAMBlock(dims[0], dims[0])
        self.prewitt1 = prewitconv(pwt[0], pwt[1], dims[0], dims[0])
        self.prewitt2 = prewitconv(pwt[0], pwt[1], dims[0], dims[0])
        self.prewitt3 = prewitconv(pwt[0], pwt[1], dims[1], dims[1])
        self.prewitt4 = prewitconv(pwt[0], pwt[1], dims[2], dims[2])
        self.prewitt5 = prewitconv(pwt[0], pwt[1], dims[3], dims[3])

        # self.CPE1 = down(dims[0], dims[0])
        # self.CPE2 = down(dims[0], dims[1])
        # self.CPE3 = down(dims[1], dims[2])
        # self.CPE4 = down(dims[2], dims[3])
        self.CPE1 = ConvPatchEmbed(patch_size[0], dims[0], dims[0], norm_layer=norm_layer)
        self.CPE2 = ConvPatchEmbed(patch_size[1], dims[0], dims[1], norm_layer=norm_layer)
        self.CPE3 = ConvPatchEmbed(patch_size[2], dims[1], dims[2], norm_layer=norm_layer)
        self.CPE4 = ConvPatchEmbed(patch_size[3], dims[2], dims[3], norm_layer=norm_layer)

        self.MCA1 = LocalTransformerBlock(dims[0])
        self.MCA2 = LocalTransformerBlock(dims[1])
        # self.LMT1 = LMT(dims[0], num_heads[0], sr_ratios[0])
        # self.LMT2 = LMT(dims[1], num_heads[1], sr_ratios[1])
        self.LMT3 = LMT(dims[2], num_heads[2], sr_ratios[2])
        self.LMT4 = LMT(dims[3], num_heads[3], sr_ratios[3])

        self.up1 = UpSample(dims[3])
        self.up2 = UpSample(dims[2])
        self.up3 = UpSample(dims[1])
        self.up4 = Up(dims[0], dims[0])
        self.out = OutSample(dims[0], num_classes)

        self.C1 = DecoderBlock(dims[2])
        self.C2 = DecoderBlock(dims[1])
        self.C3 = DecoderBlock(dims[0])
        self.C4 = DecoderBlock(dims[0])

    def forward(self, x):
        x1 = self.relu(self.bn(self.conv1(x)))
        x2 = self.prewitt1(x1)
        x3, x33, H1, W1 = self.CPE1(x2)
        # x3 = self.CPE1(x2)
        # x3 = self.prewitt2(x3)
        x3 = self.MCA1(x33)

        # x3 = self.DAM(x2)
        x4, x44, H2, W2= self.CPE2(x3)
        # print(",", x44.shape)

        # x5 = self.LMT1(x4, x44, H1, W1)
        # x5 = self.prewitt3(x44)
        # x6, x66, H2, W2= self.CPE2(x5)
        # print(",", x66.shape)
        # x7 = self.LMT2(x6, x66, H2, W2)
        # x5 = self.prewitt2(x4)
        x7 = self.MCA2(x44)
        # x7 = self.prewitt3(x7)
        x8, x88, H3, W3 = self.CPE3(x7)
        # print(",", x88.shape)
        # x8 = self.CPE3(x7)
        x80 = self.prewitt4(x88)
        x9 = self.LMT3(x8, x88, H3, W3, x80)
        # x9 = self.prewitt4(x9)
        # out_maps(x9, max_num=96, out_path='E:/lijiawei/ETUnet-main/feature-maps/')
        x10, x110, H4, W4 = self.CPE4(x9)
        # x10 = self.CPE4(x9)
        x111 = self.prewitt5(x110)
        # print(",", x10.shape)
        x11 = self.LMT4(x10, x110, H4, W4, x111)

        x16 = self.up1(x11, x9)
        # print(",,,,", x16.shape)
        x17 = self.C1(x16)
        x18 = self.up2(x17, x7)
        # print(",,,,", x18.shape)
        x19 = self.C2(x18)
        x20 = self.up3(x19, x3)
        # print(",,,,", x20.shape)
        x21 = self.C3(x20)
        x22 = self.up4(x21, x2)
        # print(",,,,", x22.shape)
        x23 = self.C4(x22)
        x24 = self.out(x23)
        # print(",,,,", x24.shape)
        return x24

# if __name__ == '__main__':
#     x = torch.randn(1, 3, 256, 256)
#     net = ETUNET()
#     net(x)
#     flops, params = profile(net, inputs=(x,))
#     print(flops/1e9, params/1e6)




