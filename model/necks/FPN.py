import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule


class GeneralizedLSSFPN(nn.Module):
    """
    Generalized LSS Feature Pyramid Network.
    
    此模块从骨干网络获取特征，通过自上而下的方式融合不同尺度的特征，输出多尺度特征图。
    与传统FPN不同，它在融合过程中使用了拼接操作而非加法操作。
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN2d"),
        act_cfg=dict(type="ReLU"),
        upsample_cfg=dict(mode="bilinear", align_corners=True),
    ):
        """
        参数:
            in_channels (List[int]): 每个尺度的输入通道数
            out_channels (int): 输出通道数(用于每个尺度)
            num_outs (int): 输出特征图的数量
            start_level (int): 开始使用的骨干网络层级索引
            end_level (int): 结束使用的骨干网络层级索引
        """
        super(GeneralizedLSSFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins - 1
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i]
                + (
                    in_channels[i + 1]
                    if i == self.backbone_end_level - 1
                    else out_channels
                ),
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs):
        """前向传播函数"""
        # 上采样 -> 拼接 -> 1x1卷积 -> 3x3卷积
        assert len(inputs) == len(self.in_channels)

        # 构建横向特征
        laterals = [inputs[i + self.start_level] for i in range(len(inputs))]

        # 构建自顶向下路径
        used_backbone_levels = len(laterals) - 1
        for i in range(used_backbone_levels - 1, -1, -1):
            x = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                **self.upsample_cfg,
            )
            laterals[i] = torch.cat([laterals[i], x], dim=1)
            laterals[i] = self.lateral_convs[i](laterals[i])
            laterals[i] = self.fpn_convs[i](laterals[i])

        # 构建输出
        outs = [laterals[i] for i in range(used_backbone_levels)]
        return tuple(outs)


if __name__=="__main__":
    # 测试GeneralizedLSSFPN
    lss_fpn = GeneralizedLSSFPN(in_channels=[64, 128, 256], out_channels=256, num_outs=3)
    x = [torch.randn(2, 64, 64, 64), torch.randn(2, 128, 32, 32), torch.randn(2, 256, 16, 16)]
    outs = lss_fpn(x)
    for i, out in enumerate(outs):
        print(f"GeneralizedLSSFPN输出 {i} shape: {out.shape}")