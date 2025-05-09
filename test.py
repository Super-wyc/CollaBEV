import os
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset.egodataset import EgoCentricDataset
from dataset.load import collate_fn
from model.CollaBEV import CollaBEV
from utils.tools import to_device
from utils.metrics import evaluate_model

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试 CollaBEV 模型')
    parser.add_argument('--config', type=str, default='config/SwinT.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/20250508_223432/best_model.pth', 
                        help='模型检查点路径')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--output_dir', type=str, default='results', help='结果保存目录')
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    if args.batch_size:
        cfg['test'] = cfg.get('test', {})
        cfg['test']['batch_size'] = args.batch_size
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    test_dataset = EgoCentricDataset(root="data", split="test")
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=cfg['test'].get('batch_size', 1),
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=4
    )
    
    # 初始化模型
    model = CollaBEV(cfg=cfg)
    
    # 加载预训练权重
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"模型已加载，使用设备: {device}")
    print(f"测试数据集大小: {len(test_dataset)} 样本")
    
    class_names = cfg['class_names']
    
    # 运行评估
    with torch.no_grad():
        metrics_dict = evaluate_model(model, test_dataloader, class_names, 
                                     iou_thresholds=[0.3, 0.5, 0.7], device=device)
    
    print("\n----- 评估结果 -----")
    for iou_th, metrics in metrics_dict.items():
        print(f"\n### {iou_th} ###")
        print(f"平均精度均值 (mAP): {metrics['mAP']:.4f}")
        print("每类精度 (AP):")
        for class_name, ap in metrics['AP'].items():
            print(f"  {class_name}: {ap:.4f}")

    result_file = os.path.join(args.output_dir, "test_results.txt")
    with open(result_file, 'w') as f:
        f.write("----- 评估结果 -----\n")
        for iou_th, metrics in metrics_dict.items():
            f.write(f"\n### {iou_th} ###\n")
            f.write(f"平均精度均值 (mAP): {metrics['mAP']:.4f}\n")
            f.write("每类精度 (AP):\n")
            for class_name, ap in metrics['AP'].items():
                f.write(f"  {class_name}: {ap:.4f}\n")
    
    print(f"\n结果已保存至 {result_file}")


if __name__ == "__main__":
    main()