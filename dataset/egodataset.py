import os
import yaml
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
from torchvision import transforms
from functools import lru_cache
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from .bbox_utils import merge_vehicle_annotations_iou
from utils.transfrom import x1_to_x2, x_to_world
from utils.visual import create_bbox, visualize_point_cloud, visualize_sample, visualize_sample_with_merged


class EgoCentricDataset(Dataset):
    def __init__(self, root, split="train", transform=None, pc_range=[-154.0, -154.0, -5.0, 154.0, 154.0, 3.0]):
        super().__init__()
        self.root = os.path.join(root, split)
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((540, 960)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.pc_range = pc_range
        
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
        print('Dataset initized with {} samples'.format(len(self.samples)))
    
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
        
        # 筛选ego点云
        ego_pc = self._filter_point_cloud_by_range(
            ego_pc, 
            x_range=[self.pc_range[0], self.pc_range[3]],
            y_range=[self.pc_range[1], self.pc_range[4]],
            z_range=[self.pc_range[2], self.pc_range[5]]
        )
        
        ego_images = self._load_images(ego_meta, ego_info['yaml_path'])
        ego_calibs = self._get_calibs(ego_meta)
        ego_vehicles = self._parse_vehicles(ego_meta.get('vehicles', {}))

        for cam in ego_calibs:
            ego_calibs[cam]['cam_to_ego_lidar'] = ego_calibs[cam]['extrinsic']
            ego_calibs[cam]['lidar_to_ego_cam'] = torch.linalg.inv(ego_calibs[cam]['extrinsic'])
            

        world_to_ego = np.linalg.inv(x_to_world(ego_lidar_pose))

        for vehicle in ego_vehicles:
            vehicle['location'] = torch.tensor((world_to_ego @ np.append(vehicle['location'].numpy(), 1))[:3], dtype=torch.float32)
            vehicle['dimensions'] = vehicle['dimensions'] * 2
        
        # 筛选ego车辆标注
        ego_vehicles = self._filter_vehicles_by_range(
            ego_vehicles,
            x_range=[self.pc_range[0], self.pc_range[3]],
            y_range=[self.pc_range[1], self.pc_range[4]],
            z_range=[self.pc_range[2], self.pc_range[5]]
        )
        
        # Process other sources
        transformed_others = []
        for src, data in others_info.items():
            other_meta = self._load_yaml(data['yaml_path'])
            other_lidar_pose = other_meta['lidar_pose']
            T = x1_to_x2(other_lidar_pose, ego_lidar_pose)
            
            # Transform point cloud
            other_pc = self._load_point_cloud(data['bin_path'])
            transformed_pc = self._transform_pc(other_pc, T)
            
            # 筛选变换后的点云
            transformed_pc = self._filter_point_cloud_by_range(
                transformed_pc,
                x_range=[self.pc_range[0], self.pc_range[3]],
                y_range=[self.pc_range[1], self.pc_range[4]],
                z_range=[self.pc_range[2], self.pc_range[5]]
            )
            
            # Transform vehicles
            transformed_vehicles = []
            for v in self._parse_vehicles(other_meta.get('vehicles', {})):
                loc = v['location'].numpy()
                new_loc = (world_to_ego @ (np.append(loc, 1)))[:3]
                transformed_vehicles.append({
                    'type': v['type'],
                    'location': torch.tensor(new_loc, dtype=torch.float32),
                    'dimensions': v['dimensions'] * 2,
                    'rotation': v['rotation']
                })
            
            # 筛选变换后的车辆标注
            transformed_vehicles = self._filter_vehicles_by_range(
                transformed_vehicles,
                x_range=[self.pc_range[0], self.pc_range[3]],
                y_range=[self.pc_range[1], self.pc_range[4]],
                z_range=[self.pc_range[2], self.pc_range[5]]
            )
            
            # Load other data
            other_images = self._load_images(other_meta, data['yaml_path'])
            other_calibs = self._get_calibs(other_meta)

            for cam in other_calibs:
                other_calibs[cam]['cam_to_ego_lidar'] = torch.tensor(T, dtype=torch.float32) @ other_calibs[cam]['extrinsic']
                other_calibs[cam]['lidar_to_ego_cam'] = torch.linalg.inv(other_calibs[cam]['cam_to_ego_lidar'])
            
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
        
        # 合并所有来源的车辆标注
        all_other_vehicles = [other['metadata']['vehicles'] for other in transformed_others]
        merged_vehicles = merge_vehicle_annotations_iou(ego_vehicles, all_other_vehicles)

        return {
            'ego': {
                'point_cloud': ego_pc,
                'images': ego_images,
                'calibs': ego_calibs,
                'metadata': {
                    'infra': ego_info['infra'],
                    'ego_speed': ego_meta.get('ego_speed', 0.0),
                    'lidar_pose': torch.tensor(ego_lidar_pose, dtype=torch.float32),
                    # 'vehicles': ego_vehicles,
                    'scene': sample['scene'],
                    'source': ego_info['source']
                }
            },
            'others': transformed_others,
            'scene': sample['scene'],
            'base': sample['base'],
            'merged_vehicles': merged_vehicles  # 添加合并后的标注
        }
    
    @lru_cache(maxsize=32)
    def _load_yaml(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    @lru_cache(maxsize=32)
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
                'extrinsic': torch.tensor(meta[cam]['extrinsic'], dtype=torch.float32),
                'cords': torch.tensor(meta[cam]['cords'], dtype=torch.float32)
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
    
    def _filter_point_cloud_by_range(self, points, x_range=[-154.0, 154.0], y_range=[-154.0, 154.0], z_range=[-5.0, 3.0]):
        """
        按照给定范围筛选点云
        
        Args:
            points: 点云数据，形状为[N, 4+]
            x_range: x轴范围 [min_x, max_x]
            y_range: y轴范围 [min_y, max_y]
            z_range: z轴范围 [min_z, max_z]
            
        Returns:
            筛选后的点云数据
        """
        mask = (
            (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
            (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
            (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
        )
        return points[mask]

    def _filter_vehicles_by_range(self, vehicles, x_range=[-154.0, 154.0], y_range=[-154.0, 154.0], z_range=[-5.0, 3.0]):
        """
        按照给定范围筛选车辆标注
        
        Args:
            vehicles: 车辆标注列表
            x_range: x轴范围 [min_x, max_x]
            y_range: y轴范围 [min_y, max_y]
            z_range: z轴范围 [min_z, max_z]
            
        Returns:
            筛选后的车辆标注列表
        """
        filtered_vehicles = []
        for v in vehicles:
            loc = v['location'].numpy()
            if (x_range[0] <= loc[0] <= x_range[1] and
                y_range[0] <= loc[1] <= y_range[1] and
                z_range[0] <= loc[2] <= z_range[1]):
                filtered_vehicles.append(v)
        return filtered_vehicles

if __name__=="__main__":
    dataset = EgoCentricDataset(r"data",
                                pc_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 5.0])
    print(len(dataset))
    sample = dataset[180]
    print(sample['ego']['point_cloud'].shape)

    print(sample['ego']['images'].keys())

    # 将images堆叠
    # images = torch.cat([img.unsqueeze(0) for img in sample['ego']['images'].values()])
    # print(images.shape)

    # 非ego数量
    # print(len(sample['others']))

    # visualize_point_cloud(sample['ego']['point_cloud'])
    # visualize_point_cloud(sample['others'][0]['point_cloud'])
    # visualize_sample(sample)
    visualize_sample_with_merged(sample)

    # 绘制所有点云
    # point_cloud_full = torch.cat([sample['ego']['point_cloud']] + [s['point_cloud'] for s in sample['others']])
    # visualize_point_cloud(point_cloud_full)
