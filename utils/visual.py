import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def visualize_point_cloud(point_cloud_tensor, color_map='viridis', is_discrete=False):

    # 检查输入是否为 PyTorch Tensor
    if not isinstance(point_cloud_tensor, torch.Tensor):
        raise ValueError("输入必须是 PyTorch Tensor")

    # 将 Tensor 转换为 NumPy 数组
    point_cloud_numpy = point_cloud_tensor.numpy()
    # 提取坐标 (前3列)
    coordinates = point_cloud_numpy[:, :3]
    # 提取附加属性 (第4列)
    attributes = point_cloud_numpy[:, 3]
    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coordinates)

    # 根据附加属性生成颜色
    if is_discrete:
        # 离散映射：将第4列视为分类标签
        labels = attributes.astype(int)  # 假设第4列是整数类别
        unique_labels = np.unique(labels)
        colors_map = plt.cm.get_cmap('tab10', len(unique_labels))  # 使用 tab10 colormap
        colors = colors_map(labels % len(unique_labels))[:, :3]  # 映射标签到颜色
    else:
        # 连续映射：将第4列归一化并映射为颜色
        attributes_normalized = (attributes - attributes.min()) / (attributes.max() - attributes.min())
        colors_map = plt.cm.get_cmap(color_map)
        colors = colors_map(attributes_normalized)[:, :3]  # 映射属性值到颜色

    # 设置点云颜色
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 可视化点云
    o3d.visualization.draw_geometries([pcd])

def visualize_sample(sample):
    # 合并点云
    all_points = [sample['ego']['point_cloud'].numpy()]
    for other in sample['others']:
        all_points.append(other['point_cloud'].numpy())
    combined_pc = np.concatenate(all_points, axis=0)
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_pc[:, :3])
    colors = np.zeros_like(combined_pc[:, :3])
    colors[:, 0] = 0  # 默认红色
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 处理标注框
    vis_bboxes = []
    ego_pose = sample['ego']['metadata']['lidar_pose'].numpy()
    
    # 处理主车标注
    for vehicle in sample['ego']['metadata']['vehicles']:
        bbox = create_bbox(vehicle, ego_pose, color=[1, 0, 0])
        vis_bboxes.append(bbox)
    
    # 处理其他传感器标注
    for other in sample['others']:
        # print(other['source'])
        for vehicle in other['metadata']['vehicles']:
            bbox = create_bbox(vehicle, ego_pose, color=[0, 1, 0])
            vis_bboxes.append(bbox)
    
    # 创建坐标系
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    
    # 可视化
    o3d.visualization.draw_geometries([pcd, axis] + vis_bboxes)

def visualize_sample_with_merged(sample):
    # 合并点云
    all_points = [sample['ego']['point_cloud'].numpy()]
    for other in sample['others']:
        all_points.append(other['point_cloud'].numpy())
    combined_pc = np.concatenate(all_points, axis=0)
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_pc[:, :3])
    colors = np.zeros_like(combined_pc[:, :3])
    colors[:, 0] = 0  # 默认颜色
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 处理标注框
    vis_bboxes = []
    ego_pose = sample['ego']['metadata']['lidar_pose'].numpy()
    
    # 显示合并后的标注框
    for vehicle in sample['merged_vehicles']:
        bbox = create_bbox(vehicle, ego_pose, color=[0, 0, 1])  # 使用蓝色表示合并后的框
        vis_bboxes.append(bbox)
    
    # 创建坐标系
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    
    # 可视化
    o3d.visualization.draw_geometries([pcd, axis] + vis_bboxes)

def create_bbox(vehicle, source_pose, color):

    # T = np.linalg.inv(x_to_world(source_pose))
    
    # 转换中心点
    center = vehicle['location'].numpy()
    # center_h = np.append(center, 1)
    # new_center = (T @ center_h)[:3]
    
    # 转换方向
    angles = vehicle['rotation'].numpy()
    rotation = R.from_euler('xzy', angles, degrees=True)
    rot_matrix = rotation.as_matrix()
    
    # 创建3D框
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.center = center
    bbox.extent = np.take(vehicle['dimensions'].numpy(), [1,0,2])
    bbox.R = rot_matrix
    
    # 设置颜色
    # color = np.random.rand(3)
    bbox.color = color
    
    return bbox