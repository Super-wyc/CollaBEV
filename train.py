import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import yaml
import os
import numpy as np
import time
import tqdm

from dataset.egodataset import EgoCentricDataset
from dataset.load import collate_fn
from model.CollaBEV import CollaBEV
from utils.tools import to_device


def main(cfg):
    # 创建保存模型的目录
    save_dir = os.path.join("checkpoints", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    
    train_dataset = EgoCentricDataset(root=r"data", split="train")

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'],
                                  shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    # 训练模型
    model = CollaBEV(cfg=cfg)
    model.to("cuda")
    
    # 初始化优化器
    optimizer = Adam(model.parameters(), 
                    lr=cfg['train'].get('lr', 1e-3),
                    weight_decay=cfg['train'].get('weight_decay', 1e-5))
    
    # 初始化学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=cfg['train']['epochs'], 
        eta_min=cfg['train'].get('min_lr', 1e-6)
    )
    
    # 用于记录最佳模型
    best_loss = float('inf')
    
    for epoch in range(cfg['train']['epochs']):
        # 切换到训练模式
        model.train()
        
        epoch_loss = []
        pbar = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        pbar.set_description(f"Epoch {epoch+1}/{cfg['train']['epochs']}")

        for i, batch in pbar:
            # 将batch移动到cuda设备
            batch = to_device(batch, "cuda")
            
            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            result = model(img=batch['inputs']['images'], points=batch['inputs']['points'],
                           ann_info=batch['vehicles'],
                           lidar2cams=batch['inputs']['lidar2cams'], cam2lidars=batch['inputs']['cam2lidars'],
                           intrinsics=batch['inputs']['intrinsics'],
                           img_aug_matrix=batch['inputs']['img_aug_matrix'],
                           lidar_aug_matrix=batch['inputs']['lidar_aug_matrix'])
            
            # bbox = model.head.get_bboxes(result)

            # 计算损失
            loss = model.loss(result, batch['vehicles'])
            # print(loss)
            loss = sum(value for key, value in loss.items() if 'loss' in key)
            
            # 反向传播
            loss.backward()
            
            # 参数更新
            optimizer.step()
            
            # 更新进度条信息
            epoch_loss.append(loss.item())
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'avg_loss': f"{np.mean(epoch_loss):.4f}",
                'lr': f"{current_lr:.6f}"
            })
        
        # 更新学习率
        scheduler.step()
        
        # 计算当前epoch的平均损失
        avg_loss = np.mean(epoch_loss)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': cfg
            }, os.path.join(save_dir, 'best_model.pth'))
        
        # 定期保存模型
        if (epoch + 1) % cfg['train'].get('save_interval', 5) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': cfg
            }, os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))
            
    print(f"训练完成! 最佳损失: {best_loss:.4f}")
    print(f"模型已保存至: {save_dir}")

if __name__ == "__main__":
    with open("config/MobileNet.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    # print(cfg)
    # print(type(cfg['voxelizer']['num_point_features']))
    main(cfg=cfg)

# import torch
# from torch.utils.data import DataLoader
# from torch.optim import Adam
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.utils.tensorboard import SummaryWriter

# import yaml
# import os
# import numpy as np
# import time
# import tqdm

# from dataset.egodataset import EgoCentricDataset
# from dataset.load import collate_fn
# from model.CollaBEV import CollaBEV
# from utils.tools import to_device
# from utils.metrics import validate_and_log  # 导入验证函数

# def main(cfg):
#     # 创建保存模型的目录
#     save_dir = os.path.join("checkpoints", time.strftime("%Y%m%d_%H%M%S"))
#     os.makedirs(save_dir, exist_ok=True)
    
#     # 创建TensorBoard日志
#     log_dir = os.path.join("logs", time.strftime("%Y%m%d_%H%M%S"))
#     os.makedirs(log_dir, exist_ok=True)
#     logger = SummaryWriter(log_dir=log_dir)
    
#     # 创建数据集和数据加载器
#     train_dataset = EgoCentricDataset(root=r"data", split="train")
#     val_dataset = EgoCentricDataset(root=r"data", split="valid")  # 添加验证集
    
#     train_dataloader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'],
#                                   shuffle=True, collate_fn=collate_fn)
#     val_dataloader = DataLoader(val_dataset, batch_size=cfg['train']['batch_size'],
#                                 shuffle=False, collate_fn=collate_fn)
    
#     # 获取类别名称
#     class_names = cfg['head']['class_names'][0]  # 假设这里存储了类别名称
    
#     # 训练模型
#     model = CollaBEV(cfg=cfg)
#     model.to("cuda")
    
#     # 初始化优化器
#     optimizer = Adam(model.parameters(), 
#                     lr=cfg['train'].get('lr', 1e-3),
#                     weight_decay=cfg['train'].get('weight_decay', 1e-5))
    
#     # 初始化学习率调度器
#     scheduler = CosineAnnealingLR(
#         optimizer, 
#         T_max=cfg['train']['epochs'], 
#         eta_min=cfg['train'].get('min_lr', 1e-6)
#     )
    
#     # 用于记录最佳模型
#     best_loss = float('inf')
#     best_map = 0.0
    
#     for epoch in range(cfg['train']['epochs']):
#         # 切换到训练模式
#         model.train()
        
#         epoch_loss = []
#         pbar = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
#         pbar.set_description(f"Epoch {epoch+1}/{cfg['train']['epochs']}")

#         for i, batch in pbar:
#             # 将batch移动到cuda设备
#             batch = to_device(batch, "cuda")
            
#             # 梯度清零
#             optimizer.zero_grad()

#             # 前向传播
#             result = model(img=batch['inputs']['images'], points=batch['inputs']['points'],
#                            ann_info=batch['vehicles'],
#                            lidar2cams=batch['inputs']['lidar2cams'], cam2lidars=batch['inputs']['cam2lidars'],
#                            intrinsics=batch['inputs']['intrinsics'],
#                            img_aug_matrix=batch['inputs']['img_aug_matrix'],
#                            lidar_aug_matrix=batch['inputs']['lidar_aug_matrix'])

#             # 计算损失
#             loss = model.loss(result, batch['vehicles'])   
            
#             # 反向传播
#             loss.backward()
            
#             # 参数更新
#             optimizer.step()
            
#             # 更新进度条信息
#             epoch_loss.append(loss.item())
#             current_lr = optimizer.param_groups[0]['lr']
#             pbar.set_postfix({
#                 'loss': f"{loss.item():.4f}", 
#                 'avg_loss': f"{np.mean(epoch_loss):.4f}",
#                 'lr': f"{current_lr:.6f}"
#             })
            
#             # 记录训练日志
#             global_step = epoch * len(train_dataloader) + i
#             logger.add_scalar('Train/Loss', loss.item(), global_step)
#             logger.add_scalar('Train/LR', current_lr, global_step)
        
#         # 更新学习率
#         scheduler.step()
        
#         # 计算当前epoch的平均损失
#         avg_loss = np.mean(epoch_loss)
        
#         # 在验证集上评估模型（一定间隔进行验证）
#         if (epoch + 1) % cfg['train'].get('val_interval', 1) == 0:
#             avg_map = validate_and_log(model, val_dataloader, class_names, epoch + 1, logger)
            
#             # 保存最佳模型（基于mAP）
#             if avg_map > best_map:
#                 best_map = avg_map
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'loss': avg_loss,
#                     'mAP': best_map,
#                     'config': cfg
#                 }, os.path.join(save_dir, 'best_model_by_map.pth'))
#                 print(f"保存最佳模型 (mAP: {best_map:.4f})")
        
#         # 保存最佳模型（基于损失）
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': best_loss,
#                 'config': cfg
#             }, os.path.join(save_dir, 'best_model_by_loss.pth'))
        
#         # 定期保存模型
#         if (epoch + 1) % cfg['train'].get('save_interval', 5) == 0:
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': avg_loss,
#                 'config': cfg
#             }, os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))
            
#     print(f"训练完成! 最佳损失: {best_loss:.4f}, 最佳mAP: {best_map:.4f}")
#     print(f"模型已保存至: {save_dir}")
#     logger.close()

# if __name__ == "__main__":
#     with open("config/MobileNet.yaml", "r") as f:
#         cfg = yaml.load(f, Loader=yaml.FullLoader)
    
#     main(cfg=cfg)