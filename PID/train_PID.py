import torch
import argparse
import numpy as np
import os
import sys
import random

# 添加 PID 目录以导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 添加 DCNv2 目录以导入 dataloaderx
# 获取当前文件目录的父目录的 DCNv2 (即 ~/SID/DCNv2)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dcnv2_path = os.path.join(project_root, 'DCNv2')
sys.path.append(dcnv2_path)

print(f"Adding path: {dcnv2_path}")

from SCLDataset import SCLDataset
from PID import DBSCANSIDGenerator

try:
    from dataloaderx import TaobaoDataset
except ImportError:
    print("Error: Could not import TaobaoDataset from DCNv2/dataloaderx.py")
    sys.exit(1)

def run_dbscan_sid_generation(args):
    # 配置设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 加载 SCLDataset
    print(f"Loading SCLDataset from {args.data_dir}...")
    try:
        scl_dataset = SCLDataset(args.data_dir)
    except FileNotFoundError:
        print(f"Error: SCLDataset files not found in {args.data_dir}")
        return

    # 2. 加载 TaobaoDataset (用于频次统计)
    print(f"Loading TaobaoDataset for frequency counting...")
    
    # 假设 args.data_dir 是 feature_map 目录
    # root 应该是它的上一级目录 /var/tmp/lhy_datasets/Taobao-MM/mini_dataset/
    dataset_root = os.path.dirname(args.data_dir.rstrip('/'))
    print(f"Dataset Root: {dataset_root}")
    
    # 初始化 TaobaoDataset
    # 注意: limit_files=None 以读取所有文件保证统计准确
    taobao_dataset = TaobaoDataset(
        root=dataset_root, 
        mode='train', 
        max_len=1,        
        dataset='taobao',
        use_sid=False
    )
    
    # 3. 初始化 Generator
    print("Initializing DBSCAN SID Generator...")
    generator = DBSCANSIDGenerator(scl_dataset, taobao_dataset, device=device)
    
    # 4. 运行
    print(f"Starting Generation with: eps={args.eps}, min_samp={args.min_samples}, top%={args.top_percent}")
          
    generator.run(
        save_path=args.output_dir,
        eps=args.eps,
        min_samples=args.min_samples,
        k=args.k,
        diff_threshold=args.diff_threshold,
        percent=args.top_percent
    )
    
    print("\n[Finished] Semantic IDs and Similarities generated and saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 默认路径修改为你的路径
    parser.add_argument('--data_dir', type=str, default='/var/tmp/lhy_datasets/Taobao-MM/mini_dataset/feature_map')
    parser.add_argument('--output_dir', type=str, default='/var/tmp/lhy_datasets/Taobao-MM/mini_dataset/feature_map')
    
    parser.add_argument('--eps', type=float, default=0.6)
    parser.add_argument('--min_samples', type=int, default=3)
    parser.add_argument('--top_percent', type=float, default=0.1)
    
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--diff_threshold', type=float, default=0.1)
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2026)

    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    run_dbscan_sid_generation(args)