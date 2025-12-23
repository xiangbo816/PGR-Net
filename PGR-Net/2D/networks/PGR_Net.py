from functools import partial
from torch.nn.modules.utils import _pair
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from timm.models.layers import DropPath


class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels,
                             kernel_size=scale[i],
                             padding=scale[i] // 2,
                             groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x


class PGR_Net(nn.Module):

    def __init__(self, in_chan, base_chan, num_classes=9, aux_loss=False, num_block=2, embed_dims=[32, 64, 128, 256]):
        super().__init__()

        self.aux_loss = aux_loss
        self.inc = [BasicBlock(in_chan, base_chan)]
        self.inc.append(BasicBlock(base_chan, base_chan))
        self.inc = nn.Sequential(*self.inc)
        self.down1 = down_block(base_chan, 2 * base_chan, (2, 2), num_block=num_block, embed_dims=embed_dims[0])
        self.down2 = down_block(2 * base_chan, 4 * base_chan, (2, 2), num_block=num_block, embed_dims=embed_dims[1])
        self.down3 = down_block(4 * base_chan, 8 * base_chan, (2, 2), num_block=num_block, embed_dims=embed_dims[2])
        self.down4 = down_block(8 * base_chan, 16 * base_chan, (2, 2), num_block=num_block, embed_dims=embed_dims[3])
        self.up4 = up_block(2 * base_chan, base_chan, scale=(2, 2), num_block=2)
        self.up3 = up_block(4 * base_chan, 2 * base_chan, scale=(2, 2), num_block=2)
        self.up2 = up_block(8 * base_chan, 4 * base_chan, scale=(2, 2), num_block=2)
        self.up1 = up_block(16 * base_chan, 8 * base_chan, scale=(2, 2), num_block=2)

        self.outc = nn.Conv2d(base_chan, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.outc(out)
        return out


class TransBlock(nn.Module):
    def __init__(self, dim, num_heads=1, mlp_ratio=4., sr_ratio=1, drop=0.,
                 drop_path=0., norm_layer=nn.GroupNorm):
        super().__init__()
        self.norm1 = norm_layer(num_groups=1, num_channels=dim)
        self.norm3 = norm_layer(num_groups=1, num_channels=dim)
        self.attn = Attention(dim=dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(num_groups=1, num_channels=dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # MLP
        self.mlp = fft_Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       drop=drop)

    def forward(self, x, cnn_feature):
        if cnn_feature is not None:
            x = x + self.drop_path(self.attn(self.norm1(x), self.norm3(cnn_feature)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Trans_CNNBlock(nn.Module):

    def __init__(self, inplanes, planes, embed_dim, stride=1, in_patch_size=7, in_stride=2, in_pad=3):
        super().__init__()
        self.planes = planes
        self.inplanes = inplanes
        self.conv1 = conv3x3(planes//2, planes//2, stride)
        self.bn1 = nn.BatchNorm2d(planes//2)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes//2, planes//2)
        self.bn2 = nn.BatchNorm2d(planes//2)
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(planes//2),
                self.relu,
                nn.Conv2d(planes//2, planes//2, kernel_size=1, stride=stride, bias=False)
            )
        self.patch_embed = PatchEmbed(patch_size=in_patch_size,
                                      stride=in_stride,
                                      padding=in_pad,
                                      in_chans=planes//2,
                                      embed_dim=planes//2)
        self.trans_block = TransBlock(dim=planes//2)
        self.pos_embed = nn.Conv2d(planes//2, planes//2, kernel_size=7, padding=3, groups=planes//2)
        self.proj = nn.Sequential(
            nn.BatchNorm2d(planes),
            self.relu,
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=planes//2),

            nn.BatchNorm2d(planes),
            self.relu,
            nn.Conv2d(planes, planes//2, kernel_size=1),

            nn.BatchNorm2d(planes//2),
            self.relu,
            nn.Conv2d(planes//2, planes, kernel_size=1))

        self.conv0 = nn.Sequential(
            nn.BatchNorm2d(inplanes),
            self.relu,
            nn.Conv2d(inplanes, planes, kernel_size=1))

    def forward(self, x):
        x = self.conv0(x)
        x = torch.split(x, self.planes//2, dim=1)
        CNN_x = x[0]
        Trans_x = x[1]

        residue = CNN_x
        out = self.bn1(CNN_x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += self.shortcut(residue)
        _,_,H,W = out.size()
        x = self.patch_embed(Trans_x)
        # x = x + self.pos_embed(x)
        x = self.trans_block(x,out)
        x = F.interpolate(x, size=(H,W))
        x = torch.cat([out, x], dim=1)
        out = self.proj(x) + x
        return out


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(inplanes),
                self.relu,
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
            )
    def forward(self, x):
        residue = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += self.shortcut(residue)

        return out


class down_block(nn.Module):
    def __init__(self, in_ch, out_ch, scale, num_block, embed_dims, bottleneck=False, pool=True):
        super().__init__()

        block_list = []

        if bottleneck:
            block = 2
        else:
            # False
            block = Trans_CNNBlock

        if pool:
            # True
            # block_list.append(SimplifiedLIP(in_ch))
            block_list.append(nn.MaxPool2d(scale))
            block_list.append(block(in_ch, out_ch, embed_dims))
        else:
            block_list.append(block(in_ch, out_ch, embed_dims, stride=2))

        for i in range(num_block - 1):
            block_list.append(block(out_ch, out_ch, embed_dims, stride=1))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x):
        return self.conv(x)


class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, scale=(2, 2), bottleneck=False):
        super().__init__()
        self.scale = scale

        self.conv_ch = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        if bottleneck:
            block = BasicBlock
        else:
            block = BasicBlock

        block_list = []
        block_list.append(block(2 * out_ch, out_ch))

        for i in range(num_block - 1):
            block_list.append(block(out_ch, out_ch))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x1 = self.conv_ch(x1)

        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)

        return out

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                     groups=groups)


class RoPE(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)

    def forward(self, slen: Tuple[int]):
        index_h = torch.arange(slen[0]).to(self.angle)
        index_w = torch.arange(slen[1]).to(self.angle)
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :]) 
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :]) 
        sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1) 
        sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1) 
        sin = torch.cat([sin_h, sin_w], -1)
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :]) 
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :]) 
        cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1) 
        cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1) 
        cos = torch.cat([cos_h, cos_w], -1) 
        retention_rel_pos = (sin.flatten(0, 1), cos.flatten(0, 1))
        return retention_rel_pos

def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)
def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


class Attention(nn.Module):  ### OSRA
    def __init__(self, dim,
                 num_heads=1,
                 qk_scale=None,
                 attn_drop=0,
                 sr_ratio=1, ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.sr = nn.Identity()
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2))
        self.RoPE = RoPE(dim, num_heads)

    def forward(self, x,cnn_features):
        B, C, H, W = x.shape
        sin, cos = self.RoPE((H, W))
        q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        
        if cnn_features is not None:
            B1,C1,H1,W1 = cnn_features.shape
            if H1!=H:
                cnn_features = self.maxpool(cnn_features)
            kv = self.sr(cnn_features)
            kv = self.local_conv(kv) + kv
            k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)
            v = v.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
            k = k.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1,-2)
        else:
            kv = self.sr(x)
            kv = self.local_conv(kv) + kv
            k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)

            v = v.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
            k = k.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        
        q_rope = theta_shift(q, sin, cos)
        k_rope = theta_shift(k, sin, cos)
        attn = (q_rope @ k_rope.transpose(-1,-2)) * self.scale
        
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-1, -2)
        return x.reshape(B, C, H, W)



class fft_Mlp(nn.Module):  ### MS-FFN
    """
    Mlp implemented by with 1x1 convolutions.

    Input: Tensor with shape [B, C, H, W].
    Output: Tensor with shape [B, C, H, W].
    Args:
        in_features (int): Dimension of input features.
        hidden_features (int): Dimension of hidden features.
        out_features (int): Dimension of output features.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0, ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_features),
            self.act,
        )
        self.dwconv = MultiScaleDWConv(hidden_features)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_features),

        )
        self.drop = nn.Dropout(drop)
        

        self.fft_channel_weight = nn.Parameter(torch.randn((1, hidden_features, 1, 1)))
        # self.fft_channel_bias = nn.Parameter(torch.randn((1, hidden_features, 1, 1)))

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw//factor+1)*factor-hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad
    def unpad(self, x, t_pad):
        hw = x.shape[-1]
        return x[...,t_pad[0]:hw-t_pad[1]]

    def forward(self, x):
        x = self.fc1(x)
        residual = x
        x = self.dwconv(x)
        x, pad_w = self.pad(x,2)
        x = torch.fft.rfft2(x)
        # x = self.fft_channel_weight * x + self.fft_channel_bias
        x = self.fft_channel_weight * x
        # print('no bias')
        x = torch.fft.irfft2(x)
        x = self.unpad(x, pad_w)
        # print('fft')
        x = x + residual
        x = self.norm(self.act(x))

        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class PatchEmbed(nn.Module):
    """Patch Embedding module implemented by a layer of convolution.

    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    Args:
        patch_size (int): Patch size of the patch embedding. Defaults to 16.
        stride (int): Stride of the patch embedding. Defaults to 16.
        padding (int): Padding of the patch embedding. Defaults to 0.
        in_chans (int): Input channels. Defaults to 3.
        embed_dim (int): Output dimension of the patch embedding.
            Defaults to 768.
        norm_layer (module): Normalization module. Defaults to None (not use).
    """

    def __init__(self,
                 patch_size=16,
                 stride=16,
                 padding=0,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans,
                      embed_dim, kernel_size=patch_size,
                      stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(embed_dim),
        )

    def forward(self, x):
        return self.proj(x)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = PGR_Net(in_chan=3, base_chan=32, num_classes=9).to(device)
    from torchinfo import summary  # 使用更现代的库

    summary(
        net,
        input_size=(1,3, 224, 224),  # 输入维度 (C, H, W)
        col_names=["input_size", "output_size", "num_params", "kernel_size"],  # 显示更多信息
        verbose=1  # 完整输出模式
    )
