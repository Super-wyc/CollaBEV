import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.MobileNet import MultiResolutionMobileNetV3
from .backbone.ResNet import MultiResolutionResNet
from .backbone.SwinT import SwinBackbone
from .necks.FPN import FPN
from .transfrom.depth_lss import DepthLSSTransform
from .head.centerpoint import CenterPointHead
from .decoder.second import SecondDecoder
from .transfrom.voxel import Voxelization
from .fusion.conv_fusion import ConvFuser

from mmdet3d.models.middle_encoders.bev_sparse_encoder import BEVFusionSparseEncoder as SparseEncoder


class CollaBEV(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 图像编码
        if cfg['img_backbone']['type'] == 'MobileNetV3':
            self.img_backbone = MultiResolutionMobileNetV3(eval(cfg['img_backbone']['out_indices']))
        elif cfg['img_backbone']['type'] == 'ResNet':
            self.img_backbone = MultiResolutionResNet()
        elif cfg['img_backbone']['type'] == 'SwinTransformer':
            self.img_backbone = SwinBackbone()
        else:
            raise NotImplementedError(f"Unknown img_backbone type: {cfg['img_backbone']['type']}")
        
        self.img_neck = FPN(in_channels=cfg['img_neck']['in_channels'], out_channels=cfg['img_neck']['out_channels'], num_outs=cfg['img_neck']['num_outs'])

        if 'deepth_lss' in cfg:
        # if False:
            self.deepth_lss_module = DepthLSSTransform(
                in_channels=cfg['deepth_lss']['in_channels'],
                out_channels=cfg['deepth_lss']['out_channels'],
                image_size=tuple(cfg['deepth_lss']['image_size']),
                feature_size=tuple(cfg['deepth_lss']['feature_size']),
                xbound=tuple(cfg['deepth_lss']['xbound']),
                ybound=tuple(cfg['deepth_lss']['ybound']),
                zbound=tuple(cfg['deepth_lss']['zbound']),
                dbound=tuple(cfg['deepth_lss']['dbound']),
                downsample=cfg['deepth_lss'].get('downsample', 2)
            )
        else:
            self.deepth_lss_module = None

        self.voxelization = Voxelization(
            voxel_size=cfg['voxelization']['voxel_size'],
            point_cloud_range=cfg['voxelization']['point_cloud_range'],
            max_num_points=cfg['voxelization']['max_num_points'],
            max_voxels=cfg['voxelization']['max_voxels'],
            deterministic=cfg['voxelization']['deterministic']
        )

        self.voxelize_reduce=cfg['voxelization'].get('voxelize_reduce', True)

        self.pts_encoder = SparseEncoder(
            sparse_shape = cfg['pts_encoder']['sparse_shape'],
            in_channels = cfg['pts_encoder']['in_channels'],
            output_channels = 128,
            order = tuple(cfg['pts_encoder']['order']),
            norm_cfg = cfg['pts_encoder']['norm_cfg'],
            encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
            encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0)),
            block_type=cfg['pts_encoder']['block_type'],
        )

        self.decoder = SecondDecoder(cfg['decoder'])

        if cfg['fusion']['type'] == 'ConvFuser':
            self.fusion = ConvFuser(
                in_channels=cfg['fusion']['in_channels'],
                out_channels=cfg['fusion']['out_channels']
            )

        self.head = CenterPointHead(**cfg['head'])
    
    def point_feature_extractor(self, points):
        feats, coords, sizes = self.voxelize(points)
        batch_size = coords[-1, 0] + 1
        pts_features = self.pts_encoder(feats, coords, batch_size)
        
        return pts_features

    def img_featrue_extractor(self, img):
        B, N, C, H, W = img.shape
        img = img.view(B*N, C, H, W).contiguous()
        img = self.img_backbone(img)

        img = self.img_neck(img)[-1]

        BN, C, H, W = img.size()

        img = img.view(B, int(BN/B), C, H, W)

        return img
    
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.voxelization(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    def deepth_lss(self, img_features, points,
                  lidar2image, cam_intrinsic, camera2lidar,
                  img_aug_matrix, lidar_aug_matrix, metas=None):
        if self.deepth_lss_module is None:
            raise ValueError("deepth_lss 未配置")
            
        bev_features = self.deepth_lss_module(
            img=img_features,
            points=points,
            lidar2image=lidar2image,
            cam_intrinsic=cam_intrinsic,
            camera2lidar=camera2lidar,
            img_aug_matrix=img_aug_matrix,
            lidar_aug_matrix=lidar_aug_matrix,
            metas=metas
        )
        
        return bev_features

    def forward(self, img, points, ann_info,
                cam2lidars, lidar2cams, intrinsics,
                img_aug_matrix, lidar_aug_matrix
                ):
        img_features = self.img_featrue_extractor(img)

        img_features = self.deepth_lss(
            img_features=img_features,
            points=points,
            lidar2image=lidar2cams,
            cam_intrinsic=intrinsics,
            camera2lidar=cam2lidars,
            img_aug_matrix=img_aug_matrix,
            lidar_aug_matrix=lidar_aug_matrix,
            metas=None
        )

        points_features = self.point_feature_extractor(points)

        x = self.fusion(inputs=[img_features, points_features])
        x = self.decoder(x)

        return self.head(x)
    
    def loss(self, pred, target):
        return self.head.loss(pred, target)
        # return sum(value for key, value in self.head.loss(pred, target).items() if 'loss' in key)

# todo: 模型loss、评估等