import math
import torch
from torch import nn
from transformer import TransfromerBlock

def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class LightWeightBlock(nn.Module):
    def __init__(self, inp):
        super(LightWeightBlock, self).__init__()

        self.stride = 1
        oup = inp
        branch_features = oup // 2

        self.conv1 = nn.Conv2d(inp if (self.stride > 1) else branch_features, branch_features, kernel_size=1, stride=1,
                               padding=0, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1,
                               bias=True, groups=branch_features)
        self.conv3 = nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x2 = self.conv1(x2)
        x2 = self.relu1(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        x2 = self.relu2(x2)
        out = torch.cat((x1, x2), dim=1)

        out = channel_shuffle(out, 2)

        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SSAM(nn.Module):
    def __init__(self, in_channels):
        super(SSAM, self).__init__()

        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2),
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 光谱注意力模块（多尺度卷积：1x1, 3x1, 5x1）
        self.spectral_attention = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),  # r=1
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0)),  # r=3
            nn.Conv2d(in_channels, in_channels, kernel_size=(5, 1), padding=(2, 0))  # r=5
        ])
        self.spectral_sigmoid = nn.Sigmoid()

        # 可选加权求和参数
        self.spatial_alpha = nn.Parameter(torch.tensor(0.5))  # 空间注意力加权
        self.spectral_alpha = nn.Parameter(torch.tensor(0.5))  # 光谱注意力加权

        # 前馈网络（Feed Forward Network, FFN）用于增强特征表示
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1),  # 先扩展通道数
            nn.LeakyReLU(),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),  # 再压缩回原始通道数
            nn.InstanceNorm2d(in_channels)  # 使用 BatchNorm2d 替代 LayerNorm
        )

        # 最终的残差连接
        self.residual_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 残差映射

        # 加权求和后的残差连接
        self.weighted_sum_residual_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 加权求和后的残差映射

    def forward(self, x):
        # 获取空间注意力图
        spatial_attention_map = self.spatial_attention(x)

        # 获取光谱注意力图（分别计算各尺度的特征图，并融合）
        spectral_attention_maps = []
        for conv in self.spectral_attention:
            spectral_attention_maps.append(self.spectral_sigmoid(conv(x)))

        # 融合光谱注意力图（通过加权求和）
        spectral_attention_map = sum(spectral_attention_maps) / len(spectral_attention_maps)

        # 空间和光谱注意力加权融合
        weighted_sum = self.spatial_alpha * spatial_attention_map + self.spectral_alpha * spectral_attention_map

        # 加权求和后的残差连接
        weighted_sum_residual = self.weighted_sum_residual_conv(weighted_sum)

        # 添加前馈网络（FFN）增强特征表示
        enhanced_feats = self.ffn(weighted_sum)

        # 融合前馈网络输出与原始输入特征图和加权求和后的残差
        residual = self.residual_conv(x)  # 对输入进行 1x1 卷积映射
        output = enhanced_feats + residual + weighted_sum_residual  # 残差连接

        return output

class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.se2 = SELayer(channel=64)
        self.block2 = LightWeightBlock(64)
        self.se3 = SELayer(channel=64)
        self.block3 = LightWeightBlock(64)
        self.se4 = SELayer(channel=64)
        self.block4 = ResidualBlock(64)
        self.se5 = SELayer(channel=64)
        self.block5 = ResidualBlock(64)
        self.se6 = SELayer(channel=64)
        self.block6 = ResidualBlock(64)

        # 加入 SSAM 模块
        self.ssam = SSAM(in_channels=64)

        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )

        self.transformerblock1 = TransfromerBlock(dim=64, depth=4, patch_size=(2, 2), mlp_dim=64 * 4)
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 4, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block2 = self.se2(block2)
        block3 = self.block3(block2)
        block3 = self.se3(block3)
        block4 = self.block4(block3)
        block4 = self.se4(block4)
        block5 = self.block5(block4)
        block5 = self.se5(block5)
        block6 = self.block6(block5)
        block6 = self.se6(block6)

        # Apply SSAM here after residual blocks
        block6 = self.ssam(block6)  # 修改为使用block6作为输入

        block7 = self.block7(block6)
        block7 = self.transformerblock1(block7)
        block8 = self.block8(block1 + block7)  # 确保block1和block7的尺寸一致
        return (torch.tanh(block8) + 1) / 2

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),

            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 512, kernel_size=1),

            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=1)
        )
    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # self.bn1 = nn.InstanceNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # self.bn2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        # residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        # residual = self.bn2(residual)

        return x + residual

class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

