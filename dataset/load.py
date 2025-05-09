import torch

def collate_fn(batch):
    results = {
        'scene': [],
        'timestamp': [],
        'inputs': {
            'images':[],
            'points':[],
            # 'calibs':[],
            'cam2lidars':[],
            'lidar2cams':[],
            'img_aug_matrix':[],
            'lidar_aug_matrix':[],
            'intrinsics':[],
        },
        'vehicles': [],
    }

    for item in batch:
        images = []
        points = []
        cam2lidars = []
        lidar2cams = []
        intrinsics = []
        img_augs = []
        results['scene'].append(item['scene'])
        results['timestamp'].append(item['base'])

        for cam in item['ego']['images'].keys():
            images.append(item['ego']['images'][cam])
            # results['inputs']['calibs'].append(item['ego']['calibs'][cam])
            cam2lidars.append(item['ego']['calibs'][cam]['cam_to_ego_lidar'])
            lidar2cams.append(item['ego']['calibs'][cam]['lidar_to_ego_cam'])
            intrinsics.append(item['ego']['calibs'][cam]['intrinsic'])
            img_augs.append(torch.eye(4))  # 假设没有增强，使用单位矩阵


        # results['inputs']['points'].append(item['ego']['point_cloud'])
        points.append(item['ego']['point_cloud'])

        for source in item['others']:
            for cam in source['images'].keys():
                images.append(source['images'][cam])
                # results['inputs']['calibs'].append(source['calibs'][cam])
                cam2lidars.append(source['calibs'][cam]['cam_to_ego_lidar'])
                lidar2cams.append(source['calibs'][cam]['lidar_to_ego_cam'])
                intrinsics.append(source['calibs'][cam]['intrinsic'])
                img_augs.append(torch.eye(4))  # 假设没有增强，使用单位矩阵
            # results['inputs']['points'].append(source['point_cloud'])
            points.append(source['point_cloud'])
            

        results['inputs']['images'].append(torch.stack(images, dim=0))
        results['inputs']['points'].append(torch.cat(points, dim=0))
        results['inputs']['cam2lidars'].append(torch.stack(cam2lidars, dim=0))
        results['inputs']['lidar2cams'].append(torch.stack(lidar2cams, dim=0))
        results['inputs']['intrinsics'].append(torch.stack(intrinsics, dim=0))
        results['inputs']['img_aug_matrix'].append(torch.stack(img_augs, dim=0))
        results['inputs']['lidar_aug_matrix'].append(torch.eye(4))
        results['vehicles'].append(item['merged_vehicles'])

    # 将图像数据按照[batch, numbers, channel, height, width]的形式堆叠
    results['inputs']['images'] = torch.stack(results['inputs']['images'], dim=0)
    results['inputs']['cam2lidars'] = torch.stack(results['inputs']['cam2lidars'], dim=0)
    results['inputs']['lidar2cams'] = torch.stack(results['inputs']['lidar2cams'], dim=0)
    results['inputs']['intrinsics'] = torch.stack(results['inputs']['intrinsics'], dim=0)
    results['inputs']['img_aug_matrix'] = torch.stack(results['inputs']['img_aug_matrix'], dim=0)
    results['inputs']['lidar_aug_matrix'] = torch.stack(results['inputs']['lidar_aug_matrix'], dim=0)
    
    return results