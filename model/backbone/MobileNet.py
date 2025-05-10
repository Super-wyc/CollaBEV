import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large
from torchvision.models.feature_extraction import create_feature_extractor
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class MultiResolutionMobileNetV3(nn.Module):
    def __init__(self, out_indices=(3, 6, 9), pretrained=False):
        """
        Args:
            out_indices (tuple): 指定需要输出的特征层索引。
        """
        super().__init__()
        
        # 加载预训练的 MobileNetV3 Large 模型
        self.mobilenet = mobilenet_v3_large(pretrained=pretrained)
        
        # 提取中间层特征
        self.feature_extractor = create_feature_extractor(
            self.mobilenet.features,
            return_nodes={f'{i}': f'feat{i}' for i in out_indices}
        )
        
        # 记录输出特征图的索引
        self.out_indices = out_indices

    def forward(self, x):
        """
        前向传播，返回多个分辨率的特征图。
        """
        features = self.feature_extractor(x)
        return [features[f'feat{i}'] for i in self.out_indices]
    


if __name__ == '__main__':
    # 创建 backbone
    backbone = MultiResolutionMobileNetV3(out_indices=(3, 6, 12))
    backbone = backbone.to('cuda') if torch.cuda.is_available() else backbone

    # 输入张量
    x = torch.randn(16, 3, 256, 704)
    x = x.to('cuda') if torch.cuda.is_available() else x

    # 前向传播
    features = backbone(x)

    # 打印输出特征图的形状
    for i, feat in enumerate(features):
        print(f"Feature {i} shape: {feat.shape}")