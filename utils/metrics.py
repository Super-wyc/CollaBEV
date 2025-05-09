import numpy as np
import torch
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from collections import defaultdict

def convert_to_corners(box):
    """
    将中心点表示的3D边界框转换为角点表示便于计算IoU
    
    Args:
        box: [x, y, z, dx, dy, dz, roll, yaw, pitch]
            x, y, z: 中心点坐标
            dx, dy, dz: 尺寸
            roll, yaw, pitch: 旋转角度
            
    Returns:
        corners: 4个角点的坐标 [4, 2] 用于BEV IoU计算
    """
    x, y, z, dx, dy, dz, roll, yaw, pitch = box
    
    # 计算旋转矩阵
    cos_yaw = torch.cos(yaw * torch.pi / 180.0)
    sin_yaw = torch.sin(yaw * torch.pi / 180.0)
    
    # 计算未旋转的角点 (BEV 视图)
    x_corners = torch.tensor([dx/2, dx/2, -dx/2, -dx/2], device=box.device)
    y_corners = torch.tensor([dy/2, -dy/2, -dy/2, dy/2], device=box.device)
    
    # 旋转角点
    rotated_x = x_corners * cos_yaw - y_corners * sin_yaw + x
    rotated_y = x_corners * sin_yaw + y_corners * cos_yaw + y
    
    return torch.stack([rotated_x, rotated_y], dim=1)

def compute_bev_iou(box1, box2):
    """
    计算两个3D边界框在BEV视图中的IoU
    
    Args:
        box1: 第一个框 [9] 格式为 [x, y, z, dx, dy, dz, roll, yaw, pitch]
        box2: 第二个框 [9] 格式为 [x, y, z, dx, dy, dz, roll, yaw, pitch]
        
    Returns:
        iou: 交并比
    """
    # 转换为角点表示
    corners1 = convert_to_corners(box1)
    corners2 = convert_to_corners(box2)
    
    # 使用Shapely计算IoU
    try:
        polygon1 = Polygon(corners1.cpu().numpy())
        polygon2 = Polygon(corners2.cpu().numpy())
        
        if not polygon1.is_valid or not polygon2.is_valid:
            return 0.0
        
        inter_area = polygon1.intersection(polygon2).area
        union_area = polygon1.area + polygon2.area - inter_area
        
        return float(inter_area / union_area) if union_area > 0 else 0.0
    except Exception as e:
        print(f"计算IoU时发生错误: {e}")
        return 0.0

def compute_ap(recalls, precisions, interpolation=True):
    """
    根据精度和召回率计算平均精度 (VOC 2010标准)
    
    Args:
        recalls: 召回率数组
        precisions: 精度数组
        interpolation: 是否进行插值，VOC 2010使用插值
        
    Returns:
        average precision
    """
    # 确保数据是numpy数组
    recalls = np.asarray(recalls)
    precisions = np.asarray(precisions)
    
    # 按照召回率对精度进行排序
    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]
    
    # VOC2010: 对精度进行插值
    if interpolation:
        # 从右向左计算最大精度
        for i in range(len(precisions)-2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i+1])
    
    # 计算AP: 召回率变化的每个点处精度的平均值
    indices = np.where(np.diff(np.concatenate(([0], recalls))))[0]  # 找出召回率变化的点
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap

def compute_detection_metrics(predictions, ground_truths, classes, iou_threshold=0.5, plot=False):
    """
    计算目标检测的评估指标 (AP, mAP)
    
    Args:
        predictions: 预测结果列表，每个元素是一个字典 {'bboxes': tensor, 'scores': tensor, 'labels': tensor}
        ground_truths: 真值标注列表，每个元素是一个字典 {'type': str, 'location': tensor, 'dimensions': tensor, 'rotation': tensor}
        classes: 类别列表
        iou_threshold: IoU阈值，默认0.5
        plot: 是否绘制PR曲线
        
    Returns:
        metrics: 包含每个类别AP以及mAP的字典
    """
    if not isinstance(predictions, list):
        predictions = [predictions]
    
    # 初始化结果字典
    metrics = {'AP': {}, 'mAP': 0.0}
    
    all_precisions = []
    all_recalls = []
    
    # 为每个类别计算AP
    for class_id, class_name in enumerate(classes):
        # 收集所有预测和真值
        all_detections = []  # [(image_idx, score, tp_flag)]
        num_ground_truths = 0
        
        # 遍历所有样本
        for image_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            # 筛选当前类别的预测
            class_mask = pred['labels'] == class_id
            class_bboxes = pred['bboxes'][class_mask]
            class_scores = pred['scores'][class_mask]
            
            # 筛选当前类别的真值
            gt_class_bboxes = []
            for gt_box in gt:
                if gt_box['type'] == class_name:
                    # 构建完整bbox格式 [x, y, z, dx, dy, dz, roll, yaw, pitch]
                    bbox = torch.cat([
                        gt_box['location'],
                        gt_box['dimensions'],
                        gt_box['rotation'].unsqueeze(0) if gt_box['rotation'].dim() == 0 else gt_box['rotation']
                    ])
                    gt_class_bboxes.append(bbox)
            
            if len(gt_class_bboxes) > 0:
                gt_class_bboxes = torch.stack(gt_class_bboxes)
            else:
                gt_class_bboxes = torch.zeros((0, 9), device=class_bboxes.device)
            
            num_ground_truths += len(gt_class_bboxes)
            
            # 初始化匹配标记
            gt_matched = torch.zeros(len(gt_class_bboxes), dtype=torch.bool)
            
            # 按置信度排序预测结果
            if len(class_scores) > 0:
                sort_idx = torch.argsort(class_scores, descending=True)
                class_bboxes = class_bboxes[sort_idx]
                class_scores = class_scores[sort_idx]
            
            # 判断每个预测是否是TP
            for det_idx, (bbox, score) in enumerate(zip(class_bboxes, class_scores)):
                if len(gt_class_bboxes) == 0:
                    # 没有真值，所有预测都是FP
                    all_detections.append((image_idx, score.item(), 0))
                    continue
                
                # 计算与所有未匹配真值的IoU
                ious = torch.zeros(len(gt_class_bboxes))
                for gt_idx, gt_bbox in enumerate(gt_class_bboxes):
                    if not gt_matched[gt_idx]:
                        ious[gt_idx] = compute_bev_iou(bbox, gt_bbox)
                
                # 找到最大IoU
                max_iou, max_idx = torch.max(ious, dim=0)
                
                # 判断是否是TP
                if max_iou >= iou_threshold and not gt_matched[max_idx]:
                    gt_matched[max_idx] = True
                    all_detections.append((image_idx, score.item(), 1))  # TP
                else:
                    all_detections.append((image_idx, score.item(), 0))  # FP
        
        # 如果没有真值或预测，AP=0
        if num_ground_truths == 0 or len(all_detections) == 0:
            metrics['AP'][class_name] = 0.0
            all_precisions.append([0.0])
            all_recalls.append([0.0])
            continue
        
        # 按置信度排序所有预测
        all_detections.sort(key=lambda x: x[1], reverse=True)
        
        # 计算累积TP和FP
        tp = np.zeros(len(all_detections))
        fp = np.zeros(len(all_detections))
        
        for i, (_, _, is_tp) in enumerate(all_detections):
            if is_tp:
                tp[i] = 1
            else:
                fp[i] = 1
        
        # 累积求和
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # 计算精度和召回率
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        recalls = tp_cumsum / num_ground_truths
        
        # 计算AP
        ap = compute_ap(recalls, precisions)
        metrics['AP'][class_name] = ap
        
        all_precisions.append(precisions)
        all_recalls.append(recalls)
        
        # 绘制PR曲线
        if plot:
            plt.figure(figsize=(10, 8))
            plt.plot(recalls, precisions, label=f'{class_name} (AP={ap:.4f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - IoU={iou_threshold}')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.grid(True)
            plt.legend()
            plt.savefig(f'pr_curve_{class_name}_{iou_threshold}.png')
            plt.close()
    
    # 计算mAP
    metrics['mAP'] = np.mean(list(metrics['AP'].values()))
    
    # 绘制所有类别的PR曲线
    if plot:
        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(classes):
            if class_name in metrics['AP']:
                plt.plot(all_recalls[i], all_precisions[i], label=f'{class_name} (AP={metrics["AP"][class_name]:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - mAP={metrics["mAP"]:.4f}, IoU={iou_threshold}')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(True)
        plt.legend()
        plt.savefig(f'pr_curve_all_{iou_threshold}.png')
        plt.close()
    
    return metrics

def evaluate_model(model, dataloader, class_names, iou_thresholds=[0.3, 0.5], device='cuda'):
    """
    评估模型在验证集上的性能
    
    Args:
        model: 待评估的模型
        dataloader: 数据加载器
        class_names: 类别名称列表
        iou_thresholds: IoU阈值列表，默认为[0.3, 0.5]
        device: 计算设备
        
    Returns:
        metrics_dict: 包含不同IoU阈值下评估指标的字典
    """
    model.eval()
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 将batch移动到指定设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # 获取模型预测
            pred = model(img=batch['inputs']['images'], 
                          points=batch['inputs']['points'],
                          ann_info=batch['vehicles'],
                          lidar2cams=batch['inputs']['lidar2cams'], 
                          cam2lidars=batch['inputs']['cam2lidars'],
                          intrinsics=batch['inputs']['intrinsics'],
                          img_aug_matrix=batch['inputs']['img_aug_matrix'],
                          lidar_aug_matrix=batch['inputs']['lidar_aug_matrix'])
            
            # 获取检测结果
            detections = model.head.get_bboxes(pred)
            
            # 添加预测和真值
            predictions.extend(detections)
            ground_truths.extend(batch['vehicles'])
    
    # 计算不同IoU阈值下的指标
    metrics_dict = {}
    for iou_th in iou_thresholds:
        metrics = compute_detection_metrics(predictions, ground_truths, class_names, iou_threshold=iou_th)
        metrics_dict[f'IoU@{iou_th}'] = metrics
        
        # 打印结果
        print(f"--- Results at IoU@{iou_th} ---")
        for class_name, ap in metrics['AP'].items():
            print(f"{class_name} AP: {ap:.4f}")
        print(f"mAP: {metrics['mAP']:.4f}")
        print()
    
    return metrics_dict

# 辅助函数：计算验证集性能并添加到训练日志
def validate_and_log(model, val_loader, class_names, epoch, logger=None):
    """
    在训练过程中进行验证并记录结果
    
    Args:
        model: 当前训练的模型
        val_loader: 验证数据加载器
        class_names: 类别名称列表
        epoch: 当前训练周期
        logger: 日志记录器（如TensorBoard）
        
    Returns:
        mAP: 平均mAP值，可用于模型选择
    """
    print(f"\n--- Validating at Epoch {epoch} ---")
    
    # 计算评估指标
    metrics_dict = evaluate_model(model, val_loader, class_names)
    
    # 记录指标
    if logger is not None:
        for iou_th, metrics in metrics_dict.items():
            logger.add_scalar(f'Val/mAP_{iou_th}', metrics['mAP'], epoch)
            for class_name, ap in metrics['AP'].items():
                logger.add_scalar(f'Val/AP_{iou_th}/{class_name}', ap, epoch)
    
    # 返回0.3和0.5阈值下的平均mAP，用于模型选择
    avg_map = (metrics_dict['IoU@0.3']['mAP'] + metrics_dict['IoU@0.5']['mAP']) / 2
    return avg_map