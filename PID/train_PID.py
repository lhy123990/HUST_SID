import torch
import argparse
import numpy as np
import os
import sys
import random

# 添加 PID 目录以导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 添加 DCNv2 目录以导入 dataloaderx
# 获取当前文件目录的父目录 (即 ~/SID/) 
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import the correct class name
from PID import ClusteringSIDGenerator     # <--- FIXED IMPORT
from SCLDataset import SCLDataset

try:
    from dataloaderx import TaobaoDataset
except ImportError:
    print("Error: Could not import TaobaoDataset from DCNv2/dataloaderx.py")
    sys.exit(1)

def run_clustering_sid_generation(args):
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
    taobao_dataset = TaobaoDataset(
        root=dataset_root, 
        mode='train', 
        max_len=args.max_len,   
        dataset='taobao',
        use_sid=False
    )
    
    # 3. 初始化 Generator 
    print(f"Initializing SID Generator ({args.method})...")
    # <--- FIXED CLASS NAME
    generator = ClusteringSIDGenerator(scl_dataset, taobao_dataset, device=device) 
    
    # 4. 运行
    print(f"Starting Generation with: method={args.method}, n_clusters={args.n_clusters}, top%={args.top_percent}")
          
    generator.run(
        save_path=args.output_dir,
        method=args.method,        
        n_clusters=args.n_clusters, 
        eps=args.eps,
        min_samples=args.min_samples,
        k=args.k,
        diff_threshold=args.diff_threshold,
        percent=args.top_percent
    )
    
    print("\n[Finished] Semantic IDs and Similarities generated and saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Update defaults as needed
    parser.add_argument('--data_dir', type=str, default='/data/cbn01/mid_dataset/feature_map')
    parser.add_argument('--output_dir', type=str, default='/data/cbn01/mid_dataset/feature_map')

    parser.add_argument('--max_len', type=int, default=200) 
    
    # DBSCAN Params
    parser.add_argument('--eps', type=float, default=0.3) 
    parser.add_argument('--min_samples', type=int, default=4)
    
    # Common Params
    parser.add_argument('--top_percent', type=float, default=0.2)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--diff_threshold', type=float, default=0.3)
    
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2026)
    
    # Default to KMeans
    parser.add_argument('--method', type=str, default='dbscan', choices=['dbscan', 'kmeans'])
    parser.add_argument('--n_clusters', type=int, default=10000)

    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_clustering_sid_generation(args)