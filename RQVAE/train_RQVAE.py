# 假�?�上面的 SCLDataset 已经定义
# 假�?? RQVAE 类在 rqvae.py �???
from SCLDataset import SCLDataset
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import os
import argparse
from rqvae import RQVAE
import random

def calculate_metrics(model, dataloader, device, codebook_sizes):
    """
    Calculate:
    1. Codebook Usage / Coverage (Total unique indices used / Total codebook slots)
    2. Collision Rate (Items with same SID / Total Items)
    """
    model.eval()
    
    # Store all SIDs to check collision
    all_sids = []
    
    # Counters for usage
    # usage_sets[i] stores unique indices used in i-th codebook
    usage_sets = [set() for _ in range(len(codebook_sizes))]
    
    with torch.no_grad():
        for item_id, emb in dataloader:
            emb = emb.to(device)
            sid_list = model.get_indices(emb)
            
            sid_np = sid_list.cpu().numpy()
            all_sids.append(sid_np)
            
            # Count usage per codebook
            for i in range(len(codebook_sizes)):
                usage_sets[i].update(sid_np[:, i])
                
    # Concatenate all SIDs
    all_sids = np.concatenate(all_sids, axis=0) # [N, Num_Codebooks]
    num_items = all_sids.shape[0]
    
    # 1. Coverage (Utilization)
    # Total slots = Sum of all codebook sizes (e.g., 128 + 128 = 256)
    total_slots = sum(codebook_sizes)
    
    # Used slots = Sum of unique indices used in each codebook
    total_used = sum([len(s) for s in usage_sets])
    
    coverage = total_used / total_slots if total_slots > 0 else 0.0
    
    # Individual layer coverage
    layer_coverages = [len(s) / size for s, size in zip(usage_sets, codebook_sizes)]
    
    # 2. Collision Rate
    # Unique SIDs (rows)
    unique_sids = np.unique(all_sids, axis=0)
    num_unique_sids = unique_sids.shape[0]
    
    # Collision Rate = (Total Items - Unique SIDs) / Total Items
    # If all items have unique SID => Rate = 0.0
    # If all items map to same SID => Rate = (N-1)/N ~ 1.0
    collision_rate = (num_items - num_unique_sids) / num_items
    
    return coverage, layer_coverages, collision_rate

def train_rqvae(args):
    # 配置
    feature_map_path = args.data_dir
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # 1. 数据
    dataset = SCLDataset(feature_map_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # Validation/Stats loader without shuffle
    stats_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 获取 input_dim (应�?�为 128)
    input_dim = dataset.embeddings.shape[1] 
    print(f"Input Dimension: {input_dim}")
    
    codebook_sizes = args.codebook_structure
    num_codebooks = len(codebook_sizes)
    
    print(f"Codebook Structure: {codebook_sizes}")
    
    # 2. 模型 (Updated to match new RQVAE __init__)
    # 映射关系:
    # input_dim       -> in_dim
    # hidden_channels -> layers (需要去掉最后一?, 因为新的 RQVAE 会自动加 e_dim)
    # latent_dim      -> e_dim
    # num_codebooks & codebook_size -> num_emb_list
    
    # args.mlp_dim �??? [128],  Encoder结构变成: [Input] -> [128] -> [Latent]
    
    model = RQVAE(
        in_dim=input_dim,
        num_emb_list=codebook_sizes,  # 对应 [64, 128, 128]
        e_dim=args.latent_dim,        # 对应 128
        layers=args.mlp_dim,          # <--- �??改这里：去掉�??�??�??
        dropout_prob=0.0,
        bn=True,                   # 推荐开�??? BN 以稳定�??�???
        loss_type="mse",
        quant_loss_weight=1.0,
        beta=0.2,
        kmeans_init=False,             # 开?? K-Means �??化以加速收??
        kmeans_iters=0
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 3. �???�???
    print("Start Training RQ-VAE...")
    model.train()
    for epoch in range(args.epoch):
        total_loss = 0
        steps = 0
        model.train()
        
        for item_id, emb in dataloader:
            emb = emb.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            out, rq_loss, indices = model(emb)
            
            # Compute Loss
            total_loss_val, recon_loss = model.compute_loss(out, rq_loss, emb)
            
            total_loss_val.backward()
            optimizer.step()
            total_loss += total_loss_val.item()
            steps += 1
            
            if steps % 100 == 0:
                print(f"Epoch {epoch+1} Step {steps} Loss: {total_loss_val.item():.4f} (Recon: {recon_loss.item():.4f}, RQ: {rq_loss.item():.4f})", end='\r')
            
        avg_loss = total_loss / len(dataloader)
        
        # Calculate Metrics
        coverage, layer_covs, collision = calculate_metrics(model, stats_loader, device, codebook_sizes)
        
        layer_str = " | ".join([f"L{i}:{c*100:.1f}%" for i, c in enumerate(layer_covs)])
        print(f"\nEp {epoch+1} | Loss:{avg_loss:.4f} | Cov:{coverage*100:.2f}% [{layer_str}] | Coll:{collision*100:.2f}%")
        
    # 保存模型
    torch.save(model.state_dict(), "rqvae_model.pth")
    print("Model saved.")
    
    # 4. 生成并保�??? Semantic IDs
    save_semantic_ids(model, dataset, args.output_dir, device, args.batch_size)

def save_semantic_ids(model, dataset, output_dir, device, batch_size):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_sids = []
    all_iids = []
    
    print("Generating Semantic IDs...")
    with torch.no_grad():
        for item_id, emb in dataloader:
            emb = emb.to(device)
            
            # Encode -> Quantize -> Extract IDs
            sid_list = model.get_indices(emb)
            
            all_sids.append(sid_list.cpu().numpy()) # [Batch, Num_Codebooks]
            all_iids.append(item_id.numpy())        # [Batch]
            
    # Concat
    all_sids = np.concatenate(all_sids, axis=0) # [Total_Items, Num_Codebooks]
    all_iids = np.concatenate(all_iids, axis=0) # [Total_Items]
    
    print(f"Total Items Processed: {len(all_iids)}")
    print(f"SID Shape: {all_sids.shape}")
    
    keys_path = os.path.join(output_dir, "semantic_id_keys.npy")
    values_path = os.path.join(output_dir, "semantic_id_values.npy")
    
    np.save(keys_path, all_iids)
    np.save(values_path, all_sids)
    
    print(f"Saved SID keys to {keys_path}")
    print(f"Saved SID values to {values_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/cbn01/mid_dataset/feature_map')
    parser.add_argument('--output_dir', type=str, default='/data/cbn01/mid_dataset/feature_map')
    parser.add_argument('--lr', type=float, default=0.0001, dest='learning_rate')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epoch', type=int, default=15) 
    
    # Updated args for structure
    parser.add_argument('--codebook_structure', type=int, nargs='+', default=[200,200,200])
    parser.add_argument('--mlp_dim',  type=int, nargs='+', default=[128,64], help="SIDTierMLPDimension")
    parser.add_argument('--latent_dim', type=int, default=16, help="Latent Dimension")

    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args() 
    seed = 2026
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False   
    os.environ['PYTHONHASHSEED'] = str(seed)
    train_rqvae(args)
