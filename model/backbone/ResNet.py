import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101
from torchvision.models.feature_extraction import create_feature_extractor
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class MultiResolutionResNet(nn.Module):
    def __init__(self, depth=50, out_indices=(1, 2, 3, 4), pretrained=False):
        super().__init__()
        
        resnet_model_dict = {
            18: resnet18,
            34: resnet34,
            50: resnet50,
            101: resnet101
        }
        
        if depth not in resnet_model_dict:
            raise KeyError(f"Invalid depth {depth} for ResNet. Available options: {list(resnet_model_dict.keys())}")
        
        self.resnet = resnet_model_dict[depth](pretrained=pretrained)
        
        self.stage_names = {
            1: 'layer1',  # 输出尺寸为输入的1/4
            2: 'layer2',  # 输出尺寸为输入的1/8
            3: 'layer3',  # 输出尺寸为输入的1/16
            4: 'layer4'   # 输出尺寸为输入的1/32
        }
        
        return_nodes = {self.stage_names[i]: f'stage{i}' for i in out_indices}
        
        # 提取中间层特征
        self.feature_extractor = create_feature_extractor(
            self.resnet,
            return_nodes=return_nodes
        )
        
        # 记录输出特征图的索引
        self.out_indices = out_indices

    def forward(self, x):
        """
        前向传播，返回多个分辨率的特征图。
        """
        features = self.feature_extractor(x)
        return [features[f'stage{i}'] for i in self.out_indices]


if __name__ == '__main__':
    backbone = MultiResolutionResNet(depth=101, out_indices=(1, 2, 3))

    x = torch.randn(16, 3, 224, 224)

    features = backbone(x)

    for i, feat in enumerate(features):
        print(f"Feature {i} shape: {feat.shape}")
