import os
import yaml
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
from torchvision import transforms

def x_to_world(pose):
    x, y, z, roll, yaw, pitch = pose
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))
    
    matrix = np.identity(4)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    
    return matrix

def x1_to_x2(x1, x2):
    x1_to_world = x_to_world(x1)
    x2_to_world = x_to_world(x2)
    world_to_x2 = np.linalg.inv(x2_to_world)
    return world_to_x2 @ x1_to_world

class EgoCentricDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        super().__init__()
        self.root = os.path.join(root, split)
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((540, 960)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.samples = []
        for scene in os.listdir(self.root):
            scene_path = os.path.join(self.root, scene)
            if not os.path.isdir(scene_path):
                continue
                
            base_sources = defaultdict(dict)
            for source in os.listdir(scene_path):
                source_path = os.path.join(scene_path, source)
                if not os.path.isdir(source_path):
                    continue
                    
                for fname in os.listdir(source_path):
                    if fname.endswith(".yaml"):
                        base = fname.split(".")[0]
                        yaml_path = os.path.join(source_path, fname)
                        with open(yaml_path, 'r') as f:
                            meta = yaml.safe_load(f)
                        infra = meta.get('infra', False)
                        base_sources[base][source] = {
                            'yaml_path': yaml_path,
                            'bin_path': os.path.join(source_path, f"{base}.bin"),
                            'infra': infra,
                            'source': source
                        }
            
            for base, sources in base_sources.items():
                ego_candidates = [src for src, data in sources.items() if not data['infra']]
                for ego_src in ego_candidates:
                    ego_data = sources[ego_src]
                    others = {src: data for src, data in sources.items() if src != ego_src}
                    self.samples.append({
                        'scene': scene,
                        'base': base,
                        'ego': ego_data,
                        'others': others
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        ego_info = sample['ego']
        others_info = sample['others']
        
        # Load ego data
        ego_meta = self._load_yaml(ego_info['yaml_path'])
        ego_lidar_pose = ego_meta['lidar_pose']
        ego_pc = self._load_point_cloud(ego_info['bin_path'])
        ego_images = self._load_images(ego_meta, ego_info['yaml_path'])
        ego_calibs = self._get_calibs(ego_meta)
        ego_vehicles = self._parse_vehicles(ego_meta.get('vehicles', {}))
        
        # Process other sources
        transformed_others = []
        for src, data in others_info.items():
            other_meta = self._load_yaml(data['yaml_path'])
            other_lidar_pose = other_meta['lidar_pose']
            T = x1_to_x2(other_lidar_pose, ego_lidar_pose)
            
            # Transform point cloud
            other_pc = self._load_point_cloud(data['bin_path'])
            transformed_pc = self._transform_pc(other_pc, T)
            
            # Transform vehicles
            transformed_vehicles = []
            for v in self._parse_vehicles(other_meta.get('vehicles', {})):
                loc = v['location'].numpy()
                loc_h = np.append(loc, 1)
                new_loc = T @ loc_h
                transformed_vehicles.append({
                    'type': v['type'],
                    'location': torch.tensor(new_loc[:3], dtype=torch.float32),
                    'dimensions': v['dimensions'],
                    'rotation': v['rotation']
                })
            
            # Load other data
            other_images = self._load_images(other_meta, data['yaml_path'])
            other_calibs = self._get_calibs(other_meta)
            
            transformed_others.append({
                'source': src,
                'point_cloud': transformed_pc,
                'images': other_images,
                'calibs': other_calibs,
                'metadata': {
                    'infra': data['infra'],
                    'ego_speed': other_meta.get('ego_speed', 0.0),
                    'lidar_pose': torch.tensor(other_lidar_pose, dtype=torch.float32),
                    'vehicles': transformed_vehicles
                }
            })
        
        return {
            'ego': {
                'point_cloud': ego_pc,
                'images': ego_images,
                'calibs': ego_calibs,
                'metadata': {
                    'infra': ego_info['infra'],
                    'ego_speed': ego_meta.get('ego_speed', 0.0),
                    'lidar_pose': torch.tensor(ego_lidar_pose, dtype=torch.float32),
                    'vehicles': ego_vehicles,
                    'scene': sample['scene'],
                    'source': ego_info['source']
                }
            },
            'others': transformed_others,
            'scene': sample['scene'],
            'base': sample['base']
        }
    
    def _load_yaml(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_point_cloud(self, path):
        pc = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return torch.tensor(pc, dtype=torch.float32)
    
    def _load_images(self, meta, yaml_dir):
        images = {}
        base = os.path.basename(yaml_dir).split('.')[0]
        for cam in [k for k in meta if k.startswith('cam')]:
            img_path = os.path.join(os.path.dirname(yaml_dir), f"{base}_{cam}.jpeg")
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images[cam] = img
        return images
    
    def _get_calibs(self, meta):
        return {
            cam: {
                'intrinsic': torch.tensor(meta[cam]['intrinsic'], dtype=torch.float32),
                'extrinsic': torch.tensor(meta[cam]['extrinsic'], dtype=torch.float32)
            } for cam in meta if cam.startswith('cam')
        }
    
    def _parse_vehicles(self, vehicles_dict):
        return [{
            'type': v['obj_type'],
            'location': torch.tensor(v['location'], dtype=torch.float32),
            'dimensions': torch.tensor(v['extent'], dtype=torch.float32),
            'rotation': torch.tensor(v['angle'], dtype=torch.float32)
        } for v in vehicles_dict.values()]
    
    def _transform_pc(self, pc, T):
        homogeneous = torch.cat([pc[:, :3], torch.ones(len(pc), 1)], dim=1)
        transformed = (torch.tensor(T, dtype=torch.float32) @ homogeneous.T).T
        return torch.cat([transformed[:, :3], pc[:, 3:]], dim=1)
    
import open3d as o3d
import matplotlib.pyplot as plt
def visualize_point_cloud(point_cloud_tensor, color_map='viridis', is_discrete=False):
    """
    可视化 [N, 4] 格式的点云数据。
    
    参数:
        point_cloud_tensor (torch.Tensor): 点云数据，形状为 [N, 4]，前3列为坐标，第4列为附加属性。
        color_map (str): 颜色映射名称，默认为 'viridis'。可以选择 Matplotlib 的 colormap。
        is_discrete (bool): 是否将第4列视为离散标签，默认为 False（连续映射）。
    
    返回:
        None
    """
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

if __name__=="__main__":
    dataset = EgoCentricDataset(r"E:\projects\CollaBEV\data")
    print(len(dataset))
    sample = dataset[0]
    print(sample['ego']['point_cloud'].shape)
    visualize_point_cloud(sample['ego']['point_cloud'])
    visualize_point_cloud(sample['others'][0]['point_cloud'])

    # print(sample['ego']['images'])
    # print(sample['ego']['calibs'])
    # print(sample['ego']['metadata'])
    # print(sample['others'][0]['point_cloud'].shape)
    # print(sample['others'][0]['images'])
    # print(sample['others'][0]['calibs'])
    # print(sample['others'][0]['metadata'])
    # print(sample['scene'])
    # print(sample['base'])