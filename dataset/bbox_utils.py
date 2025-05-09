import torch
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

def box3d_to_corners(box):
    """
    将3D边界框转换为8个角点坐标
    
    Args:
        box: 包含location、dimensions和rotation的字典
        
    Returns:
        8个角点的坐标，形状为(8, 3)的numpy数组
    """
    location = box['location'].numpy() if isinstance(box['location'], torch.Tensor) else box['location']
    dimensions = box['dimensions'].numpy() if isinstance(box['dimensions'], torch.Tensor) else box['dimensions']
    rotation = box['rotation'].numpy() if isinstance(box['rotation'], torch.Tensor) else box['rotation']
    
    # 获取box参数
    l, w, h = dimensions[0], dimensions[1], dimensions[2]  # 长宽高
    
    # 创建8个角点的坐标（相对于中心点）
    x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
    y_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
    z_corners = np.array([0, 0, 0, 0, h, h, h, h]) - h/2  # 设置底部为0，顶部为h
    
    # 创建旋转矩阵
    rot = R.from_euler('xzy', rotation, degrees=True)
    rot_matrix = rot.as_matrix()
    
    # 应用旋转
    corners = np.vstack([x_corners, y_corners, z_corners])
    corners = rot_matrix @ corners
    
    # 移动至中心位置
    corners = corners.T + location
    
    return corners

def compute_box_3d_iou(box1, box2):
    """
    计算两个3D边界框的IoU
    
    Args:
        box1, box2: 两个3D框，包含location、dimensions和rotation信息
        
    Returns:
        两个框的IoU值，范围0~1
    """
    # 将边界框转换为角点
    corners1 = box3d_to_corners(box1)
    corners2 = box3d_to_corners(box2)
    
    # 计算每个边界框的体积
    volume1 = np.prod(box1['dimensions'].numpy())
    volume2 = np.prod(box2['dimensions'].numpy())
    
    # 尝试计算交集
    try:
        # 合并所有点
        all_corners = np.vstack([corners1, corners2])
        
        # 计算3D凸包
        hull = ConvexHull(all_corners)
        
        # 检查凸包体积是否可能表示有效交集
        hull_volume = hull.volume
        union_volume = volume1 + volume2
        
        # 如果凸包体积小于等于两个框体积之和，则可能有交集
        if hull_volume <= union_volume:
            # 估算交集体积
            intersection_volume = (volume1 + volume2 - hull_volume) / 2
            # 计算IoU
            if intersection_volume > 0:
                return intersection_volume / (volume1 + volume2 - intersection_volume)
    
    except Exception as e:
        # 如果计算过程出错，返回0（无交集）
        pass
        
    return 0.0

def boxes_iou_3d(box1, box2, iou_threshold=0.5):
    """
    判断两个3D框是否有足够的重叠，表示可能是同一物体
    
    Args:
        box1, box2: 两个3D框，包含location、dimensions和rotation信息
        iou_threshold: IoU阈值，超过此值认为是同一物体
        
    Returns:
        布尔值，表示两个框是否可能是同一物体
    """
    # 快速判断：先检查中心点距离
    loc1 = box1['location'].numpy() if isinstance(box1['location'], torch.Tensor) else box1['location']
    loc2 = box2['location'].numpy() if isinstance(box2['location'], torch.Tensor) else box2['location']
    
    # 计算距离
    distance = np.linalg.norm(loc1 - loc2)
    
    # 如果距离太远，快速返回False
    dim1 = box1['dimensions'].numpy() if isinstance(box1['dimensions'], torch.Tensor) else box1['dimensions']
    dim2 = box2['dimensions'].numpy() if isinstance(box2['dimensions'], torch.Tensor) else box2['dimensions']
    max_diagonal = np.linalg.norm(np.maximum(dim1, dim2))
    
    if distance > max_diagonal * 1.5:  # 距离大于最大对角线的1.5倍，肯定不重叠
        return False
    
    # 计算3D IoU
    iou = compute_box_3d_iou(box1, box2)
    
    return iou > iou_threshold

def merge_vehicle_annotations_iou(ego_vehicles, other_vehicles_list, iou_threshold=0.1):
    """
    合并来自不同来源的车辆标注，对于同一辆车只保留一个标注框
    
    Args:
        ego_vehicles: 主车的车辆标注列表
        other_vehicles_list: 其他来源的车辆标注列表(列表的列表)
        iou_threshold: IoU阈值，超过此值认为是同一物体
        
    Returns:
        合并后的车辆标注列表
    """
    # 为所有标注添加来源信息
    all_vehicles = []
    for v in ego_vehicles:
        v_copy = v.copy()
        v_copy['source'] = 'ego'
        all_vehicles.append(v_copy)
    
    for i, other_vehicles in enumerate(other_vehicles_list):
        for v in other_vehicles:
            v_copy = v.copy()
            v_copy['source'] = f'other_{i}'
            all_vehicles.append(v_copy)
    
    # 如果没有标注，直接返回空列表
    if not all_vehicles:
        return []
        
    # 记录已经被合并的框的索引
    merged_indices = set()
    
    # 聚类合并重叠框
    clusters = []
    
    # 对每个边界框检查
    for i in range(len(all_vehicles)):
        if i in merged_indices:
            continue
            
        # 创建新聚类
        current_cluster = [i]
        merged_indices.add(i)
        
        # 检查其他所有框是否与当前框重叠
        for j in range(i+1, len(all_vehicles)):
            if j in merged_indices:
                continue
                
            # 使用IoU判断是否为同一物体
            if boxes_iou_3d(all_vehicles[i], all_vehicles[j], iou_threshold):
                current_cluster.append(j)
                merged_indices.add(j)
                
        clusters.append(current_cluster)
    
    # 从每个聚类中选择一个代表性框
    merged_vehicles = []
    for cluster in clusters:
        # 优先选择ego框，否则选择第一个框
        ego_indices = [i for i in cluster if all_vehicles[i]['source'] == 'ego']
        if ego_indices:
            representative = {k: v for k, v in all_vehicles[ego_indices[0]].items() if k != 'source'}
        else:
            representative = {k: v for k, v in all_vehicles[cluster[0]].items() if k != 'source'}
        
        merged_vehicles.append(representative)
    
    return merged_vehicles