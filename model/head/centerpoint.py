from mmdet3d.models.dense_heads.centerpoint_head import CenterHead
from mmdet3d.models.utils import (clip_sigmoid, draw_heatmap_gaussian,
                                  gaussian_radius)
from mmdet.models.utils import multi_apply

import torch
import math
from mmdet3d.models.layers import circle_nms, nms_bev
from mmdet3d.structures import xywhr2xyxyr

from losses.losses import ClssificationLoss, RegressionLoss

class CenterPointHead(CenterHead):
    def __init__(self,
                 in_channels=256,
                 tasks=None,
                 bbox_coder=None,
                 common_heads=dict(),
                 loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
                 separate_head=dict(type='SeparateHead', init_bias=-2.19, final_kernel=3),
                 share_conv_channel=64,
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 norm_bbox=True,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            tasks=tasks,
            bbox_coder=bbox_coder,
            common_heads=common_heads,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            separate_head=separate_head,
            share_conv_channel=share_conv_channel,
            num_heatmap_convs=num_heatmap_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias,
            norm_bbox=norm_bbox,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            **kwargs)
        
    def forward(self, feats):
        """前向传播函数"""
        return super().forward(feats)

    def get_targets(self, batch_targets):
        """根据批次中的标注信息生成训练目标

        Args:
            batch_targets (list[dict]): 批次中每个样本的标注列表，
                每个标注为字典形式，包含'type', 'location', 'dimensions', 'rotation'键

        Returns:
            tuple[list[torch.Tensor]]: 包含以下结果的元组
                - list[torch.Tensor]: 热图分数
                - list[torch.Tensor]: 真实框
                - list[torch.Tensor]: 指示有效框位置的索引
                - list[torch.Tensor]: 指示哪些框有效的掩码
        """
        heatmaps, anno_boxes, inds, masks =multi_apply(
                            self.get_targets_single, batch_targets)
        # Transpose heatmaps
        heatmaps = list(map(list, zip(*heatmaps)))
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        # Transpose anno_boxes
        anno_boxes = list(map(list, zip(*anno_boxes)))
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        # Transpose inds
        inds = list(map(list, zip(*inds)))
        inds = [torch.stack(inds_) for inds_ in inds]
        # Transpose inds
        masks = list(map(list, zip(*masks)))
        masks = [torch.stack(masks_) for masks_ in masks]
        return heatmaps, anno_boxes, inds, masks
    
    def get_targets_single(self, anno_info):
        """根据单个样本的标注信息生成训练目标

        Args:
            anno_info ([list[dict]]): 
                批次中每个样本的标注列表，
                每个标注为字典形式，包含'type', 'location', 'dimensions', 'rotation'键

        Returns:
            tuple[list[torch.Tensor]]: 包含以下结果的元组
                - list[torch.Tensor]: 热图分数
                - list[torch.Tensor]: 真实框
                - list[torch.Tensor]: 指示有效框位置的索引
                - list[torch.Tensor]: 指示哪些框有效的掩码
        """
        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])
        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 收集所有任务的所有类型
        all_types = []
        for class_names in self.class_names:
            all_types.extend(class_names)
        
        # 筛选出有效的标注并转为张量形式
        valid_annos = []
        for anno in anno_info:
            if anno['type'] in all_types:
                valid_annos.append(anno)
        
        if len(valid_annos) == 0:
            # 如果没有有效标注，为每个任务创建空结果
            heatmaps, anno_boxes, inds, masks = [], [], [], []
            for idx in range(len(self.task_heads)):
                heatmap = torch.zeros(
                    (len(self.class_names[idx]), feature_map_size[1], feature_map_size[0]),
                    device=device)
                anno_box = torch.zeros((max_objs, 9), dtype=torch.float32, device=device)
                ind = torch.zeros((max_objs), dtype=torch.int64, device=device)
                mask = torch.zeros((max_objs), dtype=torch.uint8, device=device)
                
                heatmaps.append(heatmap)
                anno_boxes.append(anno_box)
                inds.append(ind)
                masks.append(mask)
            
            return heatmaps, anno_boxes, inds, masks
        
        # 为每个任务创建掩码
        task_masks = []
        
        # 获取每个标注的类型和对应的任务索引
        anno_types = [anno['type'] for anno in valid_annos]
        anno_task_ids = []
        anno_class_ids = []
        
        for anno_type in anno_types:
            found = False
            for task_id, class_names in enumerate(self.class_names):
                if anno_type in class_names:
                    anno_task_ids.append(task_id)
                    anno_class_ids.append(class_names.index(anno_type))
                    found = True
                    break
            if not found:
                # 应该不会到这里，因为我们已经过滤了无效类型
                anno_task_ids.append(-1)
                anno_class_ids.append(-1)
        
        # 创建每个标注的边界框
        gt_locations = torch.stack([torch.tensor(anno['location'], device=device) for anno in valid_annos])
        gt_dimensions = torch.stack([torch.tensor(anno['dimensions'], device=device) for anno in valid_annos])
        gt_rotations = torch.stack([torch.tensor(anno['rotation'], device=device) for anno in valid_annos])
        
        # 构建gt_bboxes_3d
        gt_bboxes_3d = torch.cat([gt_locations, gt_dimensions, gt_rotations], dim=1)
        
        # 按任务分配标注
        task_boxes = []
        task_classes = []
        
        for task_id in range(len(self.class_names)):
            task_indices = [i for i, tid in enumerate(anno_task_ids) if tid == task_id]
            
            if task_indices:
                task_box = gt_bboxes_3d[task_indices]
                # 获取该任务内的类别索引，并+1（0为背景）
                task_class = torch.tensor([anno_class_ids[i] + 1 for i in task_indices], 
                                        dtype=torch.long, device=device)
                
                task_boxes.append(task_box)
                task_classes.append(task_class)
            else:
                # 创建空张量
                task_boxes.append(torch.zeros((0, gt_bboxes_3d.shape[1]), device=device))
                task_classes.append(torch.zeros((0,), dtype=torch.long, device=device))
        
        # 使用和原代码相同的方式生成热图和其他目标
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []
        
        for idx, task_head in enumerate(self.task_heads):
            heatmap = torch.zeros(
                (len(self.class_names[idx]), feature_map_size[1], feature_map_size[0]),
                device=device)
                
            anno_box = torch.zeros((max_objs, 9), dtype=torch.float32, device=device)
            ind = torch.zeros((max_objs), dtype=torch.int64, device=device)
            mask = torch.zeros((max_objs), dtype=torch.uint8, device=device)
            
            num_objs = min(task_boxes[idx].shape[0], max_objs)
            
            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1  # -1是因为task_classes中+1了
                
                length = task_boxes[idx][k][3]
                width = task_boxes[idx][k][4]
                length = length / voxel_size[0] / self.train_cfg['out_size_factor']
                width = width / voxel_size[1] / self.train_cfg['out_size_factor']
                
                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (width, length),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))
                    
                    # 注意标注框坐标系统
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][1], task_boxes[idx][k][2]
                    
                    coor_x = (x - pc_range[0]) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (y - pc_range[1]) / voxel_size[1] / self.train_cfg['out_size_factor']
                    
                    center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                    center_int = center.to(torch.int32)
                    
                    # 舍弃不在范围内的对象，避免在创建热图时超出数组区域
                    if not (0 <= center_int[0] < feature_map_size[0] and 
                            0 <= center_int[1] < feature_map_size[1]):
                        continue
                        
                    draw_gaussian(heatmap[cls_id], center_int, radius)
                    
                    new_idx = k
                    x, y = center_int[0], center_int[1]
                    
                    assert(y * feature_map_size[0] + x < feature_map_size[0] * feature_map_size[1])
                    
                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1
                    
                    # 旋转角度 - 三个元素: roll, yaw, pitch
                    roll, yaw, pitch = task_boxes[idx][k][6], task_boxes[idx][k][7], task_boxes[idx][k][8]
                    
                    box_dim = task_boxes[idx][k][3:6]
                    if self.norm_bbox:
                        box_dim = box_dim.log()
                        
                    # 构建anno_box，包含中心点偏移、z坐标、尺寸以及旋转
                    anno_box[new_idx] = torch.cat([
                        center - torch.tensor([x, y], device=device),  # 中心点偏移(2)
                        z.unsqueeze(0),                                # 高度信息(1)
                        box_dim,                                       # 尺寸信息(3)
                        roll.unsqueeze(0),                             # roll(1)
                        yaw.unsqueeze(0),                              # yaw(1)
                        pitch.unsqueeze(0)                             # pitch(1)
                    ])
                    
            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
            
        return heatmaps, anno_boxes, inds, masks

    def loss(self, preds, batch_targets):
        """计算损失函数

        Args:
            preds (list[dict]): 模型预测结果，包含热图、回归框等信息
            batch_targets (list[dict]): 批次中每个样本的标注列表，
                每个标注为字典形式，包含'type', 'location', 'dimensions', 'rotation'键

        Returns:
            dict: 损失字典，包含各个损失项的值
        """
        heatmaps, anno_boxes, inds, masks = self.get_targets(batch_targets)
        
        loss_dict = dict()

        for task_id, preds_dict in enumerate(preds):
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_haetmap = self.loss_cls(
                preds_dict[0]['heatmap'], heatmaps[task_id],
                avg_factor=max(num_pos, 1)
            )
            target_box = anno_boxes[task_id]

            preds_dict[0]['anno_box'] = torch.cat(
                (preds_dict[0]['reg'], preds_dict[0]['height'],
                 preds_dict[0]['dim'], preds_dict[0]['rot']), dim=1)
            
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = super()._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()

            mask *= isnotnan

            code_weights = self.train_cfg.get('code_weights', None)
            bbox_weights = mask * mask.new_tensor(code_weights)
            loss_bbox = self.loss_bbox(
                pred, target_box, bbox_weights, avg_factor=(num + 1e-4))
            loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox
            loss_dict[f'task{task_id}.loss_heatmap'] = loss_haetmap
        
        return loss_dict
    
    def get_bboxes(self, preds):
        """根据网络预测结果生成最终的边界框
        
        Args:
            preds (list[list[dict]]): 网络预测结果，包含热图、回归框等信息
                    
        Returns:
            list[dict]: 每个样本的预测结果，包含'bboxes', 'scores', 'labels'
        """
        rets = []
        
        # 遍历每个任务
        for task_id, preds_dict in enumerate(preds):
            # 获取该任务的类别数
            num_class_with_bg = len(self.class_names[task_id])
            
            # 获取批次大小
            batch_size = preds_dict[0]['heatmap'].shape[0]
            
            # 对热图进行sigmoid激活
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()
            
            # 获取回归值、高度、尺寸等
            batch_reg = preds_dict[0]['reg']
            batch_hei = preds_dict[0]['height']
            
            # 根据是否归一化处理尺寸
            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]['dim'])
            else:
                batch_dim = preds_dict[0]['dim']
            
            # 处理旋转值 - 从roll, yaw, pitch中提取yaw并计算sin和cos
            # 注意：这里假设rot的形状为[B, 3, H, W]，其中第二维度对应[roll, yaw, pitch]
            batch_rots = torch.sin(preds_dict[0]['rot'][:, 1:2])  # sin(yaw)
            batch_rotc = torch.cos(preds_dict[0]['rot'][:, 1:2])  # cos(yaw)
            
            # 速度不在预测结构中，设为None
            batch_vel = None
            
            # 使用bbox_coder.decode解码预测结果
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id)
            
            # 提取边界框、分数和标签
            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]
            
            # 根据NMS类型进行处理
            if self.test_cfg['nms_type'] == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['min_radius'][task_id],
                            post_max_size=self.test_cfg['post_max_size']),
                        dtype=torch.long,
                        device=boxes.device)
                    
                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                # 旋转NMS
                batch_input_metas = [{
                    'box_type_3d': lambda x, y: x  # 这是一个简单的替代，实际应根据需要调整
                } for _ in range(batch_size)]
                
                rets.append(
                    self.get_task_detections(num_class_with_bg,
                                            batch_cls_preds, batch_reg_preds,
                                            batch_cls_labels,
                                            batch_input_metas))
        
        # 合并所有任务的结果
        num_samples = len(rets[0])
        ret_list = []
        
        for i in range(num_samples):
            result = {}
            bboxes = torch.cat([ret[i]['bboxes'] for ret in rets])
            scores = torch.cat([ret[i]['scores'] for ret in rets])
            
            # 处理标签，确保不同任务的标签不重叠
            flag = 0
            for j, num_class in enumerate(self.num_classes):
                rets[j][i]['labels'] += flag
                flag += num_class
            labels = torch.cat([ret[i]['labels'].int() for ret in rets])
            
            # 调整高度位置（CenterPoint通常将高度定义为底部到顶部）
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            
            result['bboxes'] = bboxes
            result['scores'] = scores  
            result['labels'] = labels
            
            ret_list.append(result)
        
        return ret_list