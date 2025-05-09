from mmdet.models.losses import GaussianFocalLoss
from mmdet.models.losses import L1Loss


class ClssificationLoss(GaussianFocalLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RegressionLoss(L1Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)