import logging
logging.basicConfig(format='%(asctime)s-<%(funcName)s>-[line:%(lineno)d]-%(levelname)s: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # define logging print level



import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import collections



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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformerSys(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        print(
            "SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(
                depths,
                depths_decoder, drop_path_rate, num_classes))

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                              self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                             patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                             patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C

        return x, x_downsample

    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C

        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)

        return x

    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)

        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


########################################################################
########################################################################
##
##
##
########################################################################
########################################################################


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        # print(x.shape, position_embeddings.shape, '？？？')
        return x + position_embeddings


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                          stride=stride, padding=kernel // 2, bias=False))
        self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

        # print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)

    def forward(self, x):
        return super().forward(x)

class BRLayer(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()

        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))

    def forward(self, x):
        return super().forward(x)



class HarDBlock_v2(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.insert(0, k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, dwconv=False):
        super().__init__()
        self.links = []
        conv_layers_ = []
        bnrelu_layers_ = []
        self.layer_bias = []
        self.out_channels = 0
        self.out_partition = collections.defaultdict(list)

        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            for j in link:
                self.out_partition[j].append(outch)

        cur_ch = in_channels
        for i in range(n_layers):
            accum_out_ch = sum(self.out_partition[i])
            real_out_ch = self.out_partition[i][0]
            # print( self.links[i],  self.out_partition[i], accum_out_ch)
            conv_layers_.append(nn.Conv2d(cur_ch, accum_out_ch, kernel_size=3, stride=1, padding=1, bias=True))
            bnrelu_layers_.append(BRLayer(real_out_ch))
            cur_ch = real_out_ch
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += real_out_ch
        # print("Blk out =",self.out_channels)

        self.conv_layers = nn.ModuleList(conv_layers_)
        self.bnrelu_layers = nn.ModuleList(bnrelu_layers_)

    def transform(self, blk, trt=False):
        # Transform weight matrix from a pretrained HarDBlock v1
        in_ch = blk.layers[0][0].weight.shape[1]
        for i in range(len(self.conv_layers)):
            link = self.links[i].copy()
            link_ch = [blk.layers[k - 1][0].weight.shape[0] if k > 0 else
                       blk.layers[0][0].weight.shape[1] for k in link]
            part = self.out_partition[i]
            w_src = blk.layers[i][0].weight
            b_src = blk.layers[i][0].bias

            self.conv_layers[i].weight[0:part[0], :, :, :] = w_src[:, 0:in_ch, :, :]
            self.layer_bias.append(b_src)

            if b_src is not None:
                if trt:
                    self.conv_layers[i].bias[1:part[0]] = b_src[1:]
                    self.conv_layers[i].bias[0] = b_src[0]
                    self.conv_layers[i].bias[part[0]:] = 0
                    self.layer_bias[i] = None
                else:
                    # for pytorch, add bias with standalone tensor is more efficient than within conv.bias
                    # this is because the amount of non-zero bias is small,
                    # but if we use conv.bias, the number of bias will be much larger
                    self.conv_layers[i].bias = None
            else:
                self.conv_layers[i].bias = None

            in_ch = part[0]
            link_ch.reverse()
            link.reverse()
            if len(link) > 1:
                for j in range(1, len(link)):
                    ly = link[j]
                    part_id = self.out_partition[ly].index(part[0])
                    chos = sum(self.out_partition[ly][0:part_id])
                    choe = chos + part[0]
                    chis = sum(link_ch[0:j])
                    chie = chis + link_ch[j]
                    self.conv_layers[ly].weight[chos:choe, :, :, :] = w_src[:, chis:chie, :, :]

            # update BatchNorm or remove it if there is no BatchNorm in the v1 block
            self.bnrelu_layers[i] = None
            if isinstance(blk.layers[i][1], nn.BatchNorm2d):
                self.bnrelu_layers[i] = nn.Sequential(
                    blk.layers[i][1],
                    blk.layers[i][2])
            else:
                self.bnrelu_layers[i] = blk.layers[i][1]

    def forward(self, x):
        layers_ = []
        outs_ = []
        xin = x
        for i in range(len(self.conv_layers)):
            link = self.links[i]
            part = self.out_partition[i]

            xout = self.conv_layers[i](xin)
            layers_.append(xout)

            xin = xout[:, 0:part[0], :, :] if len(part) > 1 else xout
            if self.layer_bias[i] is not None:
                xin += self.layer_bias[i].view(1, -1, 1, 1)

            if len(link) > 1:
                for j in range(len(link) - 1):
                    ly = link[j]
                    part_id = self.out_partition[ly].index(part[0])
                    chs = sum(self.out_partition[ly][0:part_id])
                    che = chs + part[0]

                    xin += layers_[ly][:, chs:che, :, :]

            xin = self.bnrelu_layers[i](xin)

            if i % 2 == 0 or i == len(self.conv_layers) - 1:
                outs_.append(xin)

        out = torch.cat(outs_, 1)
        return out



class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False):
        super().__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.grmul = grmul
        self.n_layers = n_layers
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0  # if upsample else in_channels
        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            use_relu = residual_out
            layers_.append(ConvLayer(inch, outch))
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        # print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or \
                    (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # print("upsample",in_channels, out_channels)

    def forward(self, x, skip, concat=True):
        out = F.interpolate(
            x,
            size=(skip.size(2), skip.size(3)),
            mode="bilinear",
            align_corners=False,
        )
        if concat:
            out = torch.cat([out, skip], 1)

        return out

#####################################################
#####  2021-09-25  -
#####  cherhoo
#####  수정내용: 기존 모델 구조는 patch partition --> swin transformer --> merging  -->  결과를 encoder 에 전달
#####################################################
class BasicLayer_new(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
    def up_x4_temp_forView(self, x):
        #H, W = self.patches_resolution
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        #logging.debug("H({}), W({}), B({}), L({}), C({}), x.shape={} ".format(H, W, B, L, C, x.shape))
        assert L == H * W, "input features has wrong size"
        x = x.view(B, H, W, -1)
        logging.debug("in baseLayer ==> x.view(B, H, W, -1)        : {}".format(x.shape))
        # x = x.permute(0, 3, 1, 2)  # B,C,H,W
        # logging.debug("in baseLayer ==> x.permute(0,3,1,2)# B,C,H,W: {}".format(x.shape))
        return x

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        tmp = self.up_x4_temp_forView(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class SwinTransformerSegmentation_unet(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        print("SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(depths, depths_decoder, drop_path_rate, num_classes))
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        logging.debug("self.num_layers = {}".format(self.num_layers))
        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer_new(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                              self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                             patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                             patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):
        logging.debug("input x: {}".format(x.shape))
        x = self.patch_embed(x)
        logging.debug("patch_embed x: {}".format(x.shape))
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        logging.debug("pos_drop x: {}".format(x.shape))
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)
            logging.debug("-------------x_downsample x: {}".format(x.shape))
            tmp_x = self.up_x4_temp_forView(x)
            logging.debug("=============x_downsample x: {} ==> decoder =-=> {}\n".format(x.shape, tmp_x.shape))

        logging.debug("norm-before x: {}".format(x.shape))
        x = self.norm(x)  # B L C
        logging.debug("norm-after x: {}".format(x.shape))
        return x, x_downsample

    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
        x = self.norm_up(x)  # B L C
        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        logging.debug("H({}), W({}), B({}), L({}), C({}), x.shape={} ".format(H, W, B, L, C, x.shape))
        assert L == H * W, "input features has wrong size"
        if self.final_upsample == "expand_first":
            x = self.up(x)
            logging.debug("in up_*4 ==>    self.up(x)              : {}".format(x.shape))
            x = x.view(B, 4 * H, 4 * W, -1)
            logging.debug("in up_*4 ==> x.view(B, 4 * H, 4 * W, -1): {}".format(x.shape))
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            logging.debug("in up_*4 ==> x.permute(0,3,1,2)# B,C,H,W: {}".format(x.shape))
            x = self.output(x)
            logging.debug("in up_*4 ==>    self.output(x)          : {}".format(x.shape))
        return x

    def up_x4_temp_forView(self, x):
        #H, W = self.patches_resolution
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        logging.debug("H({}), W({}), B({}), L({}), C({}), x.shape={} ".format(H, W, B, L, C, x.shape))
        assert L == H * W, "input features has wrong size"
        x = x.view(B, H, W, -1)
        logging.debug("in up_x4_temp*4 ==> x.view(B, H, W, -1)        : {}".format(x.shape))
        x = x.permute(0, 3, 1, 2)  # B,C,H,W
        logging.debug("in up_x4_temp*4 ==> x.permute(0,3,1,2)# B,C,H,W: {}".format(x.shape))
        return x

    def forward(self, x):
        logging.debug("run  forward_features")
        x, x_downsample = self.forward_features(x)
        logging.debug("after encode x: {}".format(x.shape))
        x = self.forward_up_features(x, x_downsample)
        logging.debug("after decode x: {}".format(x.shape))
        x = self.up_x4(x)
        logging.debug("up_x4 : {}".format(x.shape))
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

class SwinTransformerSegmentation(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        print("SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(depths, depths_decoder, drop_path_rate, num_classes))
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        logging.debug("self.num_layers = {}".format(self.num_layers))
        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer_new(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # build decoder layers
        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        #self.expand = nn.Linear(self.embed_dim*4, self.embed_dim*4*2, bias=False)
        self.expand = nn.Linear(768, 768*4, bias=False)
        self.output = nn.Conv2d(in_channels=3 , out_channels=self.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):
        logging.debug("input x: {}".format(x.shape))
        x = self.patch_embed(x)
        logging.debug("patch_embed x: {}".format(x.shape))
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        logging.debug("pos_drop x: {}".format(x.shape))
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)
            logging.debug("-------------x_downsample x: {}".format(x.shape))
            tmp_x = self.up_x4_temp_forView(x)
            #logging.debug("=============x_downsample x: {} ==> decoder =-=> {}\n".format(x.shape, tmp_x.shape))

        logging.debug("norm-before x: {}".format(x.shape))
        x = self.norm(x)  # B L C
        logging.debug("norm-after x: {}".format(x.shape))
        return x, x_downsample

    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
        x = self.norm_up(x)  # B L C
        return x

    def up_x4(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        logging.debug("H({}), W({}), B({}), L({}), C({}), x.shape={} ".format(H, W, B, L, C, x.shape))
        assert L == H * W, "input features has wrong size"
        # if self.final_upsample == "expand_first":
        x = self.expand(x)
        logging.debug("in up_*4 ==>   self.expand(x)             : {}".format(x.shape))
        # x = self.up(x)
        # logging.debug("in up_*4 ==>    self.up(x)              : {}".format(x.shape))
        x = x.view(B, 32 * H, 32 * W, -1)
        logging.debug("in up_*4 ==> x.view(B, 4 * H, 4 * W, -1): {}".format(x.shape))
        x = x.permute(0, 3, 1, 2)  # B,C,H,W
        logging.debug("in up_*4 ==> x.permute(0,3,1,2)# B,C,H,W: {}".format(x.shape))
        # x = self.norm(x)
        # logging.debug("in up_*4 ==> x.permute(0,3,1,2)# B,C,H,W: {}".format(x.shape))
        # x = self.output(x)
        # logging.debug("in up_*4 ==>    self.output(x)          : {}".format(x.shape))
        return x

    def up_x4_temp_forView(self, x):
        #H, W = self.patches_resolution
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        logging.debug("H({}), W({}), B({}), L({}), C({}), x.shape={} ".format(H, W, B, L, C, x.shape))
        assert L == H * W, "input features has wrong size"
        x = x.view(B, H, W, -1)
        logging.debug("in up_x4_temp*4 ==> x.view(B, H, W, -1)        : {}".format(x.shape))
        x = x.permute(0, 3, 1, 2)  # B,C,H,W
        logging.debug("in up_x4_temp*4 ==> x.permute(0,3,1,2)# B,C,H,W: {}".format(x.shape))
        return x

    def forward(self, x):
        logging.debug("run  forward_features")
        x, x_downsample = self.forward_features(x)
        logging.debug("after encode x: {}".format(x.shape))
        # x = self.forward_up_features(x, x_downsample)
        # logging.debug("after decode x: {}".format(x.shape))
        x_tmp = self.up_x4_temp_forView(x)
        logging.debug("x_tmp : {}".format(x_tmp.shape))
        x = self.up_x4(x)
        logging.debug("up_x4 : {}".format(x.shape))

        x = self.output(x)
        logging.debug("self.output(x) : {}".format(x.shape))

        x = F.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=True)
        logging.debug(" x: {}".format(x.shape))

        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops



class SwinTransformerSegmentation_connection(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, line_chans_in=48*16*4, line_chans_out=48*16*4*2,decoder_up_num=32,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        print("SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(depths, depths_decoder, drop_path_rate, num_classes))
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.decoder_up_num = decoder_up_num


        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        logging.debug("self.num_layers = {}".format(self.num_layers))
        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer_new(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # build decoder layers
        self.norm = norm_layer(self.num_features)
        #self.norm_up = norm_layer(self.embed_dim)
        self.expand = nn.Linear(line_chans_in, line_chans_out, bias=False)

        #self.output = nn.Conv2d(in_channels=3 , out_channels=self.num_classes, kernel_size=1, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):
        logging.debug("input x: {}".format(x.shape))
        x = self.patch_embed(x)
        logging.debug("patch_embed x: {}".format(x.shape))
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        logging.debug("pos_drop x: {}".format(x.shape))
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)
            logging.debug("-------------x_downsample x: {}".format(x.shape))
            tmp_x = self.up_x4_temp_forView(x)
            logging.debug("=============   x_downsample x: {} ==> decoder =-=> {}\n".format(x.shape, tmp_x.shape))

        logging.debug("norm-before x: {}".format(x.shape))
        x = self.norm(x)  # B L C
        logging.debug("norm-after x: {}".format(x.shape))
        return x, x_downsample

    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
        x = self.norm_up(x)  # B L C
        return x

    def decoder(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        logging.debug("H({}), W({}), B({}), L({}), C({}), x.shape={} ".format(H, W, B, L, C, x.shape))
        assert L == H * W, "input features has wrong size"
        x = self.expand(x)
        logging.debug("in up_*4 ==>   self.expand(x)             : {}".format(x.shape))
        x = x.view(B, self.decoder_up_num * H, self.decoder_up_num * W, -1)
        logging.debug("in up_*4 ==> x.view(B, 4 * H, 4 * W, -1): {}".format(x.shape))
        x = x.permute(0, 3, 1, 2)  # B,C,H,W
        logging.debug("in up_*4 ==> x.permute(0,3,1,2)# B,C,H,W: {}".format(x.shape))
        return x

    def up_x4_temp_forView(self, x):
        #H, W = self.patches_resolution
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        logging.debug("H({}), W({}), B({}), L({}), C({}), x.shape={} ".format(H, W, B, L, C, x.shape))
        assert L == H * W, "input features has wrong size"
        x = x.view(B, H, W, -1)
        logging.debug("in up_x4_temp*4 ==> x.view(B, H, W, -1)        : {}".format(x.shape))
        x = x.permute(0, 3, 1, 2)  # B,C,H,W
        logging.debug("in up_x4_temp*4 ==> x.permute(0,3,1,2)# B,C,H,W: {}".format(x.shape))
        return x

    def forward(self, x):
        logging.debug("run  forward_features")
        x, x_downsample = self.forward_features(x)
        logging.debug("after encode x: {}".format(x.shape))
        x_tmp = self.up_x4_temp_forView(x)
        logging.debug("************************--->  x_tmp : {}".format(x_tmp.shape))
        x = self.decoder(x)
        logging.debug("decode : {}".format(x.shape))
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops





class swin_hardnet_all_transformer_embed_dim_3d(nn.Module):
    def __init__(self, n_classes=19, in_channels=4, st_img_size=112, st_windows=7):
        super(swin_hardnet_all_transformer_embed_dim_3d, self).__init__()

        self.st_img_size = st_img_size
        self.st_windows = st_windows
        ###############################################################
        self.swin_transformer = SwinTransformerSegmentation_connection(img_size=self.st_img_size,
                                                                       patch_size=4,
                                                                       in_chans=48,
                                                                       num_classes=6,
                                                                       # embed_dim=48*16,
                                                                       embed_dim=96,
                                                                       line_chans_in=96 * 4,
                                                                       line_chans_out=96 * 8 * 4,
                                                                       decoder_up_num = 16,
                                                                       depths=[2, 2, 2],
                                                                       num_heads=[3, 6, 12],
                                                                       window_size=self.st_windows,
                                                                       mlp_ratio=4.,
                                                                       qkv_bias=True,
                                                                       qk_scale=None,
                                                                       drop_rate=0.,
                                                                       drop_path_rate=0.1,
                                                                       ape=False,
                                                                       patch_norm=True,
                                                                       use_checkpoint=False)
        ###############################################################

        first_ch = [16, 24, 32, 48]
        ch_list = [64, 96, 160, 224, 320]
        grmul = 1.7
        gr = [10, 16, 18, 24, 32]
        n_layers = [4, 4, 8, 8, 8]

        blks = len(n_layers)
        self.shortcut_layers = []

        self.base = nn.ModuleList([])
        self.base.append(ConvLayer(in_channels=in_channels, out_channels=first_ch[1], kernel=3, stride=2))  ## 0
        # self.base.append(ConvLayer(first_ch[0], first_ch[1], kernel=3))                           ## 1
        # self.base.append(ConvLayer(first_ch[1], first_ch[2], kernel=3, stride=2))      ## 2
        self.base.append(ConvLayer(first_ch[1], first_ch[3], kernel=3))                           ## 3

        skip_connection_channel_counts = []
        ch = first_ch[3]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.get_out_ch()
            skip_connection_channel_counts.append(ch)
            self.base.append(blk)
            if i < blks - 1:
                self.shortcut_layers.append(len(self.base) - 1)

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]

            if i < blks - 1:
                self.base.append(nn.AvgPool2d(kernel_size=2, stride=2))

        cur_channels_count = ch
        prev_block_channels = ch
        n_blocks = blks - 1
        self.n_blocks = n_blocks

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up = nn.ModuleList([])

        for i in range(n_blocks - 1, -1, -1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.conv1x1_up.append(ConvLayer(cur_channels_count, cur_channels_count // 2, kernel=1))
            cur_channels_count = cur_channels_count // 2

            blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])

            self.denseBlocksUp.append(blk)
            prev_block_channels = blk.get_out_ch()
            cur_channels_count = prev_block_channels

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count, out_channels=n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_skip_up3 = nn.Conv2d(in_channels=12, out_channels=48, kernel_size=1, stride=1, padding=0, bias=True)
    def v2_transform(self, trt=False):
        for i in range(len(self.base)):
            if isinstance(self.base[i], HarDBlock):
                blk = self.base[i]
                self.base[i] = HarDBlock_v2(blk.in_channels, blk.growth_rate, blk.grmul, blk.n_layers)
                self.base[i].transform(blk, trt)

        for i in range(self.n_blocks):
            blk = self.denseBlocksUp[i]
            self.denseBlocksUp[i] = HarDBlock_v2(blk.in_channels, blk.growth_rate, blk.grmul, blk.n_layers)
            self.denseBlocksUp[i].transform(blk, trt)

    def forward(self, x):
        skip_connections = []
        decoder_list = []
        size_in = x.size()
        for i in range(len(self.base)):
            x = self.base[i](x)
            logging.debug("self.base[{}](x): {}".format(i, x.shape))
            if i in self.shortcut_layers:
                logging.debug("add layer to skip::   skip_connections.append(x): {}  {}".format(i, x.shape))
                skip_connections.append(x)
        out = x
        logging.debug("after encoder (out.shape): {}".format(out.shape))
        logging.debug("===========================================")
        logging.debug("self.n_blocks = {}".format(self.n_blocks))
        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            logging.debug("self.transUpBlocks[{}](out:{}, skip:{}, True)".format(i, out.shape, skip.shape))
            if i == 3:
                logging.debug("in decoder skip_connections[{}]: x={}".format(i, skip.shape))
                x = self.swin_transformer(skip)
                logging.debug(" =**=  swin_transformer: x={}".format(x.shape))
                x = self.conv_skip_up3(x)
                skip = x
                logging.debug("x.size[2], x.size[3] = {} {}".format(x.shape[2], x.shape[3]))
                logging.debug("self.conv_layer: out={}-=-=-=-=-=-=-".format(skip.shape))

            logging.debug("brfore self.transUpBlocks[{}](out, skip, True) : {}".format(i, skip.shape))
            out = self.transUpBlocks[i](out, skip, True)
            logging.debug("after self.transUpBlocks[{}](out, skip, True) : {}".format(i, out.shape))
            out = self.conv1x1_up[i](out)
            logging.debug("self.conv1x1_up[{}](out): {}".format(i, out.shape))
            out = self.denseBlocksUp[i](out)
            logging.debug("self.denseBlocksUp[{}](out): {}".format(i, out.shape))
            decoder_list.append(out)
            logging.debug("--------- decoder ----->({}) : {}\n".format(i, out.shape))

        logging.debug("self.finalConv(out) before: {}".format(out.shape))
        out = self.finalConv(out)
        logging.debug("===========================================")
        logging.debug("===========================================")
        logging.debug("self.finalConv(out): {}".format(out.shape))
        out = F.interpolate(
            out,
            size=(size_in[2], size_in[3]),
            # size=(224, 224),
            mode="bilinear",
            align_corners=True)
        logging.debug("output: {}".format(out.shape))
        return out




class swin_hardnet_all_transformer_embed_dim_2connect(nn.Module):
    def __init__(self, n_classes=19, st_img_size=112, st_windows=7):
        super(swin_hardnet_all_transformer_embed_dim_2connect, self).__init__()

        self.st_img_size = st_img_size
        self.st_windows = st_windows
        ###############################################################
        self.swin_transformer = SwinTransformerSegmentation_connection(img_size=self.st_img_size,
                                                                       patch_size=4,
                                                                       in_chans=48,
                                                                       num_classes=6,
                                                                       # embed_dim=48*16,
                                                                       embed_dim=96,
                                                                       line_chans_in=96 * 4,
                                                                       line_chans_out=96 * 8 * 4,
                                                                       decoder_up_num = 16,
                                                                       depths=[2, 2, 2],
                                                                       num_heads=[3, 6, 12],
                                                                       window_size=self.st_windows,
                                                                       mlp_ratio=4.,
                                                                       qkv_bias=True,
                                                                       qk_scale=None,
                                                                       drop_rate=0.,
                                                                       drop_path_rate=0.1,
                                                                       ape=False,
                                                                       patch_norm=True,
                                                                       use_checkpoint=False)
        self.st_skip_2 = SwinTransformerSegmentation_connection(img_size=self.st_img_size//2,
                                                                       patch_size=4,
                                                                       in_chans=78,
                                                                       num_classes=2,
                                                                       # embed_dim=48*16,
                                                                       embed_dim=96,
                                                                       line_chans_in=96 * 2,
                                                                       line_chans_out=96 * 8 * 2,
                                                                       decoder_up_num=8,
                                                                       depths=[2, 2],
                                                                       num_heads=[12, 24],
                                                                       window_size=self.st_windows,
                                                                       mlp_ratio=4.,
                                                                       qkv_bias=True,
                                                                       qk_scale=None,
                                                                       drop_rate=0.,
                                                                       drop_path_rate=0.1,
                                                                       ape=False,
                                                                       patch_norm=True,
                                                                       use_checkpoint=False)
        ###############################################################

        first_ch = [16, 24, 32, 48]
        ch_list = [64, 96, 160, 224, 320]
        grmul = 1.7
        gr = [10, 16, 18, 24, 32]
        n_layers = [4, 4, 8, 8, 8]

        blks = len(n_layers)
        self.shortcut_layers = []

        self.base = nn.ModuleList([])
        self.base.append(ConvLayer(in_channels=1, out_channels=first_ch[1], kernel=3, stride=2))  ## 0
        # self.base.append(ConvLayer(first_ch[0], first_ch[1], kernel=3))                           ## 1
        # self.base.append(ConvLayer(first_ch[1], first_ch[2], kernel=3, stride=2))      ## 2
        self.base.append(ConvLayer(first_ch[1], first_ch[3], kernel=3))                           ## 3

        skip_connection_channel_counts = []
        ch = first_ch[3]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.get_out_ch()
            skip_connection_channel_counts.append(ch)
            self.base.append(blk)
            if i < blks - 1:
                self.shortcut_layers.append(len(self.base) - 1)

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]

            if i < blks - 1:
                self.base.append(nn.AvgPool2d(kernel_size=2, stride=2))

        cur_channels_count = ch
        prev_block_channels = ch
        n_blocks = blks - 1
        self.n_blocks = n_blocks

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up = nn.ModuleList([])

        for i in range(n_blocks - 1, -1, -1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.conv1x1_up.append(ConvLayer(cur_channels_count, cur_channels_count // 2, kernel=1))
            cur_channels_count = cur_channels_count // 2

            blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])

            self.denseBlocksUp.append(blk)
            prev_block_channels = blk.get_out_ch()
            cur_channels_count = prev_block_channels

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count, out_channels=n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_skip_up3 = nn.Conv2d(in_channels=12, out_channels=48, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_skip_up2 = nn.Conv2d(in_channels=24, out_channels=78, kernel_size=1, stride=1, padding=0, bias=True)
    def v2_transform(self, trt=False):
        for i in range(len(self.base)):
            if isinstance(self.base[i], HarDBlock):
                blk = self.base[i]
                self.base[i] = HarDBlock_v2(blk.in_channels, blk.growth_rate, blk.grmul, blk.n_layers)
                self.base[i].transform(blk, trt)

        for i in range(self.n_blocks):
            blk = self.denseBlocksUp[i]
            self.denseBlocksUp[i] = HarDBlock_v2(blk.in_channels, blk.growth_rate, blk.grmul, blk.n_layers)
            self.denseBlocksUp[i].transform(blk, trt)

    def forward(self, x):
        skip_connections = []
        decoder_list = []
        size_in = x.size()
        for i in range(len(self.base)):
            x = self.base[i](x)
            logging.debug("self.base[{}](x): {}".format(i, x.shape))
            if i in self.shortcut_layers:
                logging.debug("add layer to skip::   skip_connections.append(x): {}  {}".format(i, x.shape))
                skip_connections.append(x)
        out = x
        logging.debug("after encoder (out.shape): {}".format(out.shape))
        logging.debug("===========================================")
        logging.debug("self.n_blocks = {}".format(self.n_blocks))
        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            logging.debug("self.transUpBlocks[{}](out:{}, skip:{}, True)".format(i, out.shape, skip.shape))

            if i == 2:
                logging.debug("in decoder skip_connections[{}]: x={}".format(i, skip.shape))
                x = self.st_skip_2(skip)
                logging.debug(" =**=  swin_transformer[skip-2]: x={}".format(x.shape))
                x = self.conv_skip_up2(x)
                skip = x
                logging.debug("x.size[2], x.size[3] = {} {}".format(x.shape[2], x.shape[3]))
                logging.debug("self.conv_layer: out={}-=-=-=-=-=-=-".format(skip.shape))
            if i == 3:
                logging.debug("in decoder skip_connections[{}]: x={}".format(i, skip.shape))
                x = self.swin_transformer(skip)
                logging.debug(" =**=  swin_transformer: x={}".format(x.shape))
                x = self.conv_skip_up3(x)
                skip = x
                logging.debug("x.size[2], x.size[3] = {} {}".format(x.shape[2], x.shape[3]))
                logging.debug("self.conv_layer: out={}-=-=-=-=-=-=-".format(skip.shape))

            logging.debug("brfore self.transUpBlocks[{}](out, skip, True) : {}".format(i, skip.shape))
            out = self.transUpBlocks[i](out, skip, True)
            logging.debug("after self.transUpBlocks[{}](out, skip, True) : {}".format(i, out.shape))
            out = self.conv1x1_up[i](out)
            logging.debug("self.conv1x1_up[{}](out): {}".format(i, out.shape))
            out = self.denseBlocksUp[i](out)
            logging.debug("self.denseBlocksUp[{}](out): {}".format(i, out.shape))
            decoder_list.append(out)
            logging.debug("--------- decoder ----->({}) : {}\n".format(i, out.shape))

        logging.debug("self.finalConv(out) before: {}".format(out.shape))
        out = self.finalConv(out)
        logging.debug("===========================================")
        logging.debug("===========================================")
        logging.debug("self.finalConv(out): {}".format(out.shape))
        out = F.interpolate(
            out,
            size=(size_in[2], size_in[3]),
            # size=(224, 224),
            mode="bilinear",
            align_corners=True)
        logging.debug("output: {}".format(out.shape))
        return out





class swin_hardnet_all_transformer_embed_dim_768(nn.Module):
    def __init__(self, n_classes=19, st_img_size=112, st_windows=7):
        super(swin_hardnet_all_transformer_embed_dim_768, self).__init__()

        self.st_img_size = st_img_size
        self.st_windows = st_windows
        ###############################################################
        self.swin_transformer = SwinTransformerSegmentation_connection(img_size=self.st_img_size,
                                                                       patch_size=4,
                                                                       in_chans=48,
                                                                       num_classes=6,
                                                                       embed_dim=48*16,
                                                                       # embed_dim=96,
                                                                       line_chans_in=768 * 4,
                                                                       line_chans_out=768 * 4 ,
                                                                       decoder_up_num = 16,
                                                                       depths=[2, 2, 2],
                                                                       num_heads=[3, 6, 12],
                                                                       window_size=self.st_windows,
                                                                       mlp_ratio=4.,
                                                                       qkv_bias=True,
                                                                       qk_scale=None,
                                                                       drop_rate=0.,
                                                                       drop_path_rate=0.1,
                                                                       ape=False,
                                                                       patch_norm=True,
                                                                       use_checkpoint=False)
        ###############################################################

        first_ch = [16, 24, 32, 48]
        ch_list = [64, 96, 160, 224, 320]
        grmul = 1.7
        gr = [10, 16, 18, 24, 32]
        n_layers = [4, 4, 8, 8, 8]

        blks = len(n_layers)
        self.shortcut_layers = []

        self.base = nn.ModuleList([])
        self.base.append(ConvLayer(in_channels=1, out_channels=first_ch[1], kernel=3, stride=2))  ## 0
        # self.base.append(ConvLayer(first_ch[0], first_ch[1], kernel=3))                           ## 1
        # self.base.append(ConvLayer(first_ch[1], first_ch[2], kernel=3, stride=2))      ## 2
        self.base.append(ConvLayer(first_ch[1], first_ch[3], kernel=3))                           ## 3

        skip_connection_channel_counts = []
        ch = first_ch[3]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.get_out_ch()
            skip_connection_channel_counts.append(ch)
            self.base.append(blk)
            if i < blks - 1:
                self.shortcut_layers.append(len(self.base) - 1)

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]

            if i < blks - 1:
                self.base.append(nn.AvgPool2d(kernel_size=2, stride=2))

        cur_channels_count = ch
        prev_block_channels = ch
        n_blocks = blks - 1
        self.n_blocks = n_blocks

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up = nn.ModuleList([])

        for i in range(n_blocks - 1, -1, -1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.conv1x1_up.append(ConvLayer(cur_channels_count, cur_channels_count // 2, kernel=1))
            cur_channels_count = cur_channels_count // 2

            blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])

            self.denseBlocksUp.append(blk)
            prev_block_channels = blk.get_out_ch()
            cur_channels_count = prev_block_channels

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count, out_channels=n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_skip_up3 = nn.Conv2d(in_channels=12, out_channels=48, kernel_size=1, stride=1, padding=0, bias=True)
    def v2_transform(self, trt=False):
        for i in range(len(self.base)):
            if isinstance(self.base[i], HarDBlock):
                blk = self.base[i]
                self.base[i] = HarDBlock_v2(blk.in_channels, blk.growth_rate, blk.grmul, blk.n_layers)
                self.base[i].transform(blk, trt)

        for i in range(self.n_blocks):
            blk = self.denseBlocksUp[i]
            self.denseBlocksUp[i] = HarDBlock_v2(blk.in_channels, blk.growth_rate, blk.grmul, blk.n_layers)
            self.denseBlocksUp[i].transform(blk, trt)

    def forward(self, x):
        skip_connections = []
        decoder_list = []
        size_in = x.size()
        for i in range(len(self.base)):
            x = self.base[i](x)
            logging.debug("self.base[{}](x): {}".format(i, x.shape))
            if i in self.shortcut_layers:
                logging.debug("add layer to skip::   skip_connections.append(x): {}  {}".format(i, x.shape))
                skip_connections.append(x)
        out = x
        logging.debug("after encoder (out.shape): {}".format(out.shape))
        logging.debug("===========================================")
        logging.debug("self.n_blocks = {}".format(self.n_blocks))
        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            logging.debug("self.transUpBlocks[{}](out:{}, skip:{}, True)".format(i, out.shape, skip.shape))
            if i == 3:
                logging.debug("in decoder skip_connections[{}]: x={}".format(i, skip.shape))
                x = self.swin_transformer(skip)
                logging.debug(" =**=  swin_transformer: x={}".format(x.shape))
                x = self.conv_skip_up3(x)
                skip = x
                logging.debug("x.size[2], x.size[3] = {} {}".format(x.shape[2], x.shape[3]))
                logging.debug("self.conv_layer: out={}-=-=-=-=-=-=-".format(skip.shape))

            logging.debug("brfore self.transUpBlocks[{}](out, skip, True) : {}".format(i, skip.shape))
            out = self.transUpBlocks[i](out, skip, True)
            logging.debug("after self.transUpBlocks[{}](out, skip, True) : {}".format(i, out.shape))
            out = self.conv1x1_up[i](out)
            logging.debug("self.conv1x1_up[{}](out): {}".format(i, out.shape))
            out = self.denseBlocksUp[i](out)
            logging.debug("self.denseBlocksUp[{}](out): {}".format(i, out.shape))
            decoder_list.append(out)
            logging.debug("--------- decoder ----->({}) : {}\n".format(i, out.shape))

        logging.debug("self.finalConv(out) before: {}".format(out.shape))
        out = self.finalConv(out)
        logging.debug("===========================================")
        logging.debug("===========================================")
        logging.debug("self.finalConv(out): {}".format(out.shape))
        out = F.interpolate(
            out,
            size=(size_in[2], size_in[3]),
            # size=(224, 224),
            mode="bilinear",
            align_corners=True)
        logging.debug("output: {}".format(out.shape))
        return out



class hardnet_new(nn.Module):
    def __init__(self, n_classes=19, st_img_size=112, st_windows=7):
        super(hardnet_new, self).__init__()

        self.st_img_size = st_img_size
        self.st_windows = st_windows


        first_ch = [16, 24, 32, 48]
        ch_list = [64, 96, 160, 224, 320]
        grmul = 1.7
        gr = [10, 16, 18, 24, 32]
        n_layers = [4, 4, 8, 8, 8]

        blks = len(n_layers)
        self.shortcut_layers = []

        self.base = nn.ModuleList([])
        self.base.append(ConvLayer(in_channels=1, out_channels=first_ch[0], kernel=3, stride=2))  ## 0
        self.base.append(ConvLayer(first_ch[0], first_ch[1], kernel=3))                           ## 1
        self.base.append(ConvLayer(first_ch[1], first_ch[2], kernel=3, stride=2))      ## 2
        self.base.append(ConvLayer(first_ch[2], first_ch[3], kernel=3))                           ## 3

        skip_connection_channel_counts = []
        ch = first_ch[3]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.get_out_ch()
            skip_connection_channel_counts.append(ch)
            self.base.append(blk)
            if i < blks - 1:
                self.shortcut_layers.append(len(self.base) - 1)

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]

            if i < blks - 1:
                self.base.append(nn.AvgPool2d(kernel_size=2, stride=2))

        cur_channels_count = ch
        prev_block_channels = ch
        n_blocks = blks - 1
        self.n_blocks = n_blocks

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up = nn.ModuleList([])

        for i in range(n_blocks - 1, -1, -1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.conv1x1_up.append(ConvLayer(cur_channels_count, cur_channels_count // 2, kernel=1))
            cur_channels_count = cur_channels_count // 2

            blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])

            self.denseBlocksUp.append(blk)
            prev_block_channels = blk.get_out_ch()
            cur_channels_count = prev_block_channels

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count, out_channels=n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        #self.conv_skip_up3 = nn.Conv2d(in_channels=6, out_channels=48, kernel_size=1, stride=1, padding=0, bias=True)
    def v2_transform(self, trt=False):
        for i in range(len(self.base)):
            if isinstance(self.base[i], HarDBlock):
                blk = self.base[i]
                self.base[i] = HarDBlock_v2(blk.in_channels, blk.growth_rate, blk.grmul, blk.n_layers)
                self.base[i].transform(blk, trt)

        for i in range(self.n_blocks):
            blk = self.denseBlocksUp[i]
            self.denseBlocksUp[i] = HarDBlock_v2(blk.in_channels, blk.growth_rate, blk.grmul, blk.n_layers)
            self.denseBlocksUp[i].transform(blk, trt)

    def forward(self, x):
        skip_connections = []
        decoder_list = []
        size_in = x.size()
        for i in range(len(self.base)):
            x = self.base[i](x)
            logging.debug("self.base[{}](x): {}".format(i, x.shape))
            if i in self.shortcut_layers:
                logging.debug("add layer to skip::   skip_connections.append(x): {}  {}".format(i, x.shape))
                skip_connections.append(x)
        out = x
        logging.debug("after encoder (out.shape): {}".format(out.shape))
        logging.debug("===========================================")
        logging.debug("self.n_blocks = {}".format(self.n_blocks))
        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            logging.debug("self.transUpBlocks[{}](out:{}, skip:{}, True)".format(i, out.shape, skip.shape))
            logging.debug("brfore self.transUpBlocks[{}](out, skip, True) : {}".format(i, skip.shape))
            out = self.transUpBlocks[i](out, skip, True)
            logging.debug("after self.transUpBlocks[{}](out, skip, True) : {}".format(i, out.shape))
            out = self.conv1x1_up[i](out)
            logging.debug("self.conv1x1_up[{}](out): {}".format(i, out.shape))
            out = self.denseBlocksUp[i](out)
            logging.debug("self.denseBlocksUp[{}](out): {}".format(i, out.shape))
            decoder_list.append(out)
            logging.debug("--------- decoder ----->({}) : {}\n".format(i, out.shape))

        logging.debug("self.finalConv(out) before: {}".format(out.shape))
        out = self.finalConv(out)
        logging.debug("===========================================")
        logging.debug("===========================================")
        logging.debug("self.finalConv(out): {}".format(out.shape))
        out = F.interpolate(
            out,
            size=(size_in[2], size_in[3]),
            # size=(224, 224),
            mode="bilinear",
            align_corners=True)
        logging.debug("output: {}".format(out.shape))
        return out




if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set the GPUs 2 and 3 to use
    # ###############################################################
    # model = SwinTransformerSegmentation_connection(img_size=128,
    #                                       patch_size=4,
    #                                       in_chans=48,
    #                                       num_classes=2,
    #                                       embed_dim=96,
    #                                       # embed_dim=48,
    #                                       depths=[2, 2, 2, 2],
    #                                       num_heads=[3, 6, 12, 24],
    #                                       window_size=8,
    #                                       mlp_ratio=4.,
    #                                       qkv_bias=True,fa
    #                                       qk_scale=None,
    #                                       drop_rate=0.,
    #                                       drop_path_rate=0.1,
    #                                       ape=False,
    #                                       patch_norm=True,
    #                                       use_checkpoint=False).cuda()
    # model = swin_hardnet_all_transformer_embed_dim_2connect(n_classes=2, st_img_size=112, st_windows=7).cuda()
    model = hardnet_new(n_classes=2, st_img_size=112, st_windows=7).cuda()   ##  skip 3 하고 연결 128*128
    model.cuda()
    cuda0 = torch.device('cuda:0')
    # x = torch.rand((8, 48, 128, 128), device=cuda0)
    x = torch.rand((8, 1, 224, 224), device=cuda0)
    y = model(x)
    total_params = sum(p.numel() for p in model.parameters())
    logging.debug('Parameters:{}'.format(total_params))
    print(y.shape)
