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
    save_dir = os.path.join("checkpoints", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    
    train_dataset = EgoCentricDataset(root=r"data", split="train")

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'],
                                  shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    model = CollaBEV(cfg=cfg)
    model.to("cuda")
    
    optimizer = Adam(model.parameters(), 
                    lr=cfg['train'].get('lr', 1e-3),
                    weight_decay=cfg['train'].get('weight_decay', 1e-5))
    
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=cfg['train']['epochs'],
        eta_min=cfg['train'].get('min_lr', 1e-6)
    )
    
    best_loss = float('inf')
    
    for epoch in range(cfg['train']['epochs']):
        model.train()
        
        epoch_loss = []
        pbar = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        pbar.set_description(f"Epoch {epoch+1}/{cfg['train']['epochs']}")

        for i, batch in pbar:
            batch = to_device(batch, "cuda")
            
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
            
            loss.backward()
            
            optimizer.step()

            epoch_loss.append(loss.item())
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'avg_loss': f"{np.mean(epoch_loss):.4f}",
                'lr': f"{current_lr:.6f}"
            })
        
        scheduler.step()
        
        avg_loss = np.mean(epoch_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': cfg
            }, os.path.join(save_dir, 'best_model.pth'))
        
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
    
    main(cfg=cfg)