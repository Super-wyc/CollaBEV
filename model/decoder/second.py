from mmdet3d.models.backbones import SECOND
from mmdet3d.models.necks import SECONDFPN

import torch.nn as nn

class SecondDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = SECOND(**cfg['backbone'])
        self.neck = SECONDFPN(**cfg['neck'])

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return x