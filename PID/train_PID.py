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
#dcnv2_path = os.path.join(project_root, 'DCNv2')
sys.path.append(project_root)  # 添加整个项目根目录，以便导入 DCNv2 中的模块

#print(f"Adding path: {dcnv2_path}")

# Import the correct class name
from PID import ClusteringSIDGenerator     # <--- FIXED IMPORT
from SCLDataset import SCLDataset

try:
    from dataloaderx import TaobaoDataset
except ImportError:
    print("Error: Could not import TaobaoDataset from DCNv2/dataloaderx.py")
    sys.exit(1)

def run_clustering_sid_generation(args):
    # 配置设�??
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 加载 SCLDataset
    print(f"Loading SCLDataset from {args.data_dir}...")
    try:
        scl_dataset = SCLDataset(args.data_dir)
    except FileNotFoundError:
        print(f"Error: SCLDataset files not found in {args.data_dir}")
        return

    # 2. 加载 TaobaoDataset (用于频�?�统�?)
    print(f"Loading TaobaoDataset for frequency counting...")
    
    # 假�?? args.data_dir �? feature_map �?�?
    # root 应�?�是它的上一级目�? /var/tmp/lhy_datasets/Taobao-MM/mini_dataset/
    dataset_root = os.path.dirname(args.data_dir.rstrip('/'))
    print(f"Dataset Root: {dataset_root}")
    
    # 初�?�化 TaobaoDataset
    taobao_dataset = TaobaoDataset(
        root=dataset_root, 
        mode='train', 
        max_len=args.max_len,   
        dataset='taobao',
        use_sid=False
    )
    
    # 3. 初�?�化 Generator 
    print(f"Initializing SID Generator ({args.method})...")
    # <--- FIXED CLASS NAME
    generator = ClusteringSIDGenerator(scl_dataset, taobao_dataset, device=device) 
    
    # 4. 运�??
    print(f"Starting Generation with: method={args.method}, basis_size={args.basis_size}, top%={args.top_percent}")
          
    generator.run(
        save_path=args.output_dir,
        method=args.method,        
        basis_size=args.basis_size,
        eps=args.eps,
        min_samples=args.min_samples,
        k=args.k,
        diff_threshold=args.diff_threshold,
        percent=args.top_percent,
        sampling_mode=args.sampling_mode,
        random_percent=args.random_percent,
        force_basis=args.force_basis,
        merge_basis=args.merge_basis,
    )
    
    print("\n[Finished] Semantic IDs and Similarities generated and saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Update defaults as needed
    parser.add_argument('--data_dir', type=str, default='/data/cbn01/mid_dataset/feature_map')
    parser.add_argument('--output_dir', type=str, default='/data/cbn01/mid_dataset/feature_map')

    parser.add_argument('--max_len', type=int, default=200) 
    
    # DPP Params
    parser.add_argument('--basis_size', type=int, default=400)

    # DBSCAN Params
    parser.add_argument('--eps', type=float, default=0.3)
    parser.add_argument('--min_samples', type=int, default=3)
    
    # Common Params
    parser.add_argument('--top_percent', type=float, default=0.05)
    parser.add_argument('--sampling_mode', type=str, default='random', choices=['top', 'random'],
                        help='Item pre-selection mode: top frequency or random control')
    parser.add_argument('--random_percent', type=float, default=0.05,
                        help='Used when sampling_mode=random. If omitted, reuse --top_percent')
    parser.add_argument('--k', type=int, default=15)
    parser.add_argument('--diff_threshold', type=float, default=0.2)
    
    parser.add_argument('--gpu', type=int, default=1) 
    parser.add_argument('--seed', type=int, default=2026)
    
    parser.add_argument('--method', type=str, default='dpp', choices=['dpp', 'dbscan', 'random', 'hybrid'])
    parser.add_argument('--force_basis', dest='force_basis', action='store_true', default=True,
                        help='Force basis items to only map to themselves')
    parser.add_argument('--no_force_basis', dest='force_basis', action='store_false',
                        help='Disable basis self-only mapping')
    parser.add_argument('--merge_basis', action='store_true', help='Run DPP + DBSCAN then merge basis IDs for a unified top-k mapping')

    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_clustering_sid_generation(args)
    #python train_PID.py --method dbscan --merge_basis --basis_size 300 --eps 0.3 --min_samples 4 --force_basis