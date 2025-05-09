from mmdet.models.backbones.swin import SwinTransformer
import torch
import torch.nn as nn

class SwinBackbone(nn.Module):
    """
    特征提取网络层，基于SwinTransformer
    
    Args:
        pretrain_img_size (int | tuple[int]): 预训练时输入图像的尺寸，默认224
        in_channels (int): 输入通道数，默认3
        embed_dims (int): 特征维度，默认96
        depths (tuple[int]): 各阶段Swin Transformer块的深度，默认(2, 2, 6, 2)
        num_heads (tuple[int]): 各阶段注意力头的数量，默认(3, 6, 12, 24)
        window_size (int): 窗口大小，默认7
        out_indices (tuple[int]): 输出的阶段索引，默认(0, 1, 2, 3)
        use_abs_pos_embed (bool): 是否使用绝对位置编码，默认False
        pretrained (str): 预训练模型路径，默认None
        frozen_stages (int): 冻结的阶段数，默认-1表示不冻结
        **kwargs: 其他SwinTransformer参数
    """
    
    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 window_size=7,
                 out_indices=(1, 2, 3),
                 use_abs_pos_embed=False,
                 frozen_stages=-1,
                 use_checkpoint=False,
                 **kwargs):
        super(SwinBackbone, self).__init__()
        
        self.backbone = SwinTransformer(
            pretrain_img_size=pretrain_img_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            out_indices=out_indices,
            use_abs_pos_embed=use_abs_pos_embed,
            frozen_stages=frozen_stages,
            **kwargs
        )
        
        # 获取各阶段的特征维度，用于后续的特征融合或处理
        self.num_features = self.backbone.num_features
        self.out_indices = out_indices
    
    def forward(self, x):
        """
        前向传播函数
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (B, C, H, W)
            
        Returns:
            tuple[torch.Tensor]: 各阶段的特征图
        """
        return self.backbone(x)


if __name__ == '__main__':
    # 创建 SwinBackbone 实例
    backbone = SwinBackbone(
        embed_dims=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        # with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
        )
    )

    # backbone.to('cuda')
    
    # 输入张量
    x = torch.randn(2, 3, 540, 960)
    
    # 前向传播
    features = backbone(x)
    
    # 打印输出特征图的形状
    for i, feat in enumerate(features):
        print(f"Feature {i} shape: {feat.shape}")