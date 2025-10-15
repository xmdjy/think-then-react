#!/usr/bin/env python3
"""
修复 PyTorch Lightning 检查点文件，添加缺失的版本信息
"""

import torch
import shutil
import os
from pathlib import Path
import lightning.pytorch as pl

def fix_checkpoint(checkpoint_path):
    """修复检查点文件，添加缺失的 pytorch-lightning_version 键和 state_dict 结构"""
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"错误：检查点文件不存在: {checkpoint_path}")
        return False
    
    # 创建备份
    backup_path = checkpoint_path.with_suffix('.ckpt.backup')
    print(f"创建备份文件: {backup_path}")
    shutil.copy2(checkpoint_path, backup_path)
    
    try:
        # 加载检查点
        print(f"加载检查点文件: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 检查是否已经有正确的结构
        if 'state_dict' in checkpoint and 'pytorch-lightning_version' in checkpoint:
            print("检查点文件已经有正确的结构，无需修复")
            return True
        
        # 分离模型权重和元数据
        model_weights = {}
        metadata = {}
        
        for key, value in checkpoint.items():
            if key in ['pytorch-lightning_version', 'epoch', 'global_step', 'lr_schedulers', 'optimizer_states', 'hyper_parameters']:
                metadata[key] = value
            else:
                # 这些是模型权重
                model_weights[key] = value
        
        # 创建新的检查点结构
        new_checkpoint = {
            'state_dict': model_weights,
            'epoch': metadata.get('epoch', 0),
            'global_step': metadata.get('global_step', 0),
            'pytorch-lightning_version': metadata.get('pytorch-lightning_version', pl.__version__),
        }
        
        # 如果有其他元数据，也包含进去
        for key in ['lr_schedulers', 'optimizer_states', 'hyper_parameters']:
            if key in metadata:
                new_checkpoint[key] = metadata[key]
        
        print(f"添加 pytorch-lightning_version: {new_checkpoint['pytorch-lightning_version']}")
        print(f"重新组织为 state_dict 结构，包含 {len(model_weights)} 个权重")
        print(f"epoch: {new_checkpoint['epoch']}, global_step: {new_checkpoint['global_step']}")
        
        # 保存修复后的检查点
        print(f"保存修复后的检查点: {checkpoint_path}")
        torch.save(new_checkpoint, checkpoint_path)
        
        print("检查点修复成功！")
        return True
        
    except Exception as e:
        print(f"修复检查点时出错: {e}")
        # 恢复备份
        if backup_path.exists():
            print("恢复备份文件...")
            shutil.copy2(backup_path, checkpoint_path)
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='修复 PyTorch Lightning 检查点文件')
    parser.add_argument('checkpoint_path', help='检查点文件路径')
    
    args = parser.parse_args()
    
    success = fix_checkpoint(args.checkpoint_path)
    if success:
        print("\n✅ 检查点修复完成！现在可以正常加载了。")
    else:
        print("\n❌ 检查点修复失败。")

if __name__ == '__main__':
    main()