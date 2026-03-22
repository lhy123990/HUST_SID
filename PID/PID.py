"""
[Modified] DBSCAN Based Semantic ID Generator
----------------------------------------------
本文件重写了原 RQ-VAE 代码，改为基于 DBSCAN 聚类生成 Semantic ID.

核心逻辑:
1. 统计训练集中 Item 出现频次，选取 Top 10% (记为 Top Items).
2. 对 Top Items 的 Embedding 进行 DBSCAN 聚类，得到聚类中心 (实际上这里 Top Items 本身也被视为某种基准).
3. 对于 Top Items: 直接设置 SID 为 [Item_ID, -1, ..., -1]. (自身即语义)
4. 对于其他 Items: 计算与 Top Items (聚类中心) 的相似度，选取 Top K.
   截断: sim[i] - sim[i+1] > threshold 时，后续设为 -1.
"""

import torch
import numpy as np
import os
from sklearn.cluster import DBSCAN
import torch.nn.functional as F
from tqdm import tqdm

class DBSCANSIDGenerator:
    """
    SID 生成器 (基于DBSCAN聚类):
    1. 统计 Top 10% Items.
    2. DBSCAN 聚类得到中心.
    3. Normal Items -> 映射到 centroids (with truncation).
    4. Top Items -> 保持自身 [ID, -1, ...].
    """
    def __init__(self, scl_dataset, taobao_dataset=None, device='cuda'):
        """
        Args:
            scl_dataset: 提供 Item Embeddings (SCLDataset 实例)
            taobao_dataset: 提供训练集频次统计 (TaobaoDataset 实例)
            device: 计算设备
        """
        self.scl_dataset = scl_dataset
        self.taobao_dataset = taobao_dataset
        self.device = device
        
        # 加载所有 Item 的 ID 和 Embeddings
        # 兼容处理: item_ids 可能是 Tensor 或 numpy
        if hasattr(scl_dataset, 'item_ids') and isinstance(scl_dataset.item_ids, torch.Tensor):
             self.item_ids = scl_dataset.item_ids.numpy()
        else:
             self.item_ids = getattr(scl_dataset, 'item_ids', [])
             
        if hasattr(scl_dataset, 'embeddings') and isinstance(scl_dataset.embeddings, torch.Tensor):
             self.embeddings = scl_dataset.embeddings.to(device)
        else:
             self.embeddings = torch.tensor(scl_dataset.embeddings).to(device)
        
        # 建立 Raw ID -> Embedding Index 的映射，用于快速查找 (Index in scl_dataset)
        self.id_to_idx = {uid: i for i, uid in enumerate(self.item_ids)}
        
        # 建立 Raw ID Set 为了快速判断是否是 Top Item
        self.dataset_idx_to_raw = {}
        
        self._add_type_conversions_to_map()
        
        print(f"[DBSCANSID] Loaded {len(self.item_ids)} items from SCLDataset.")
        
        self.top_item_raw_ids_set = set() # 存储 Top Items 的原始 ID

    def _add_type_conversions_to_map(self):
        new_entries = {}
        for k, v in self.id_to_idx.items():
            try:
                new_entries[str(k)] = v
                new_entries[int(k)] = v
            except:
                pass
        self.id_to_idx.update(new_entries)

    def get_top_k_percent_items(self, percent=0.1, item_col='205'):
        """
        统计训练集中出现频次最高的 Top K% Items
        """
        if self.taobao_dataset is None:
            raise ValueError("需要提供 TaobaoDataset 以统计频次.")
            
        print(f"[DBSCANSID] Counting frequencies (Col: {item_col})...")
        
        counts = {}
        if hasattr(self.taobao_dataset, 'data_tensor'):
            try:
                # 找到 item_col 对应的特征索引
                feat_names = list(map(str, self.taobao_dataset.feature_names))
                if str(item_col) in feat_names:
                    col_idx = feat_names.index(str(item_col))
                else:
                    raise ValueError(f"Feature {item_col} not found")

                item_col_data = self.taobao_dataset.data_tensor[:, col_idx].numpy()
                unique, counts_arr = np.unique(item_col_data, return_counts=True)
                counts = dict(zip(unique, counts_arr))
                
                self.vocab_map = self.taobao_dataset.maps.get(str(item_col))
                if self.vocab_map is None:
                    print(f"[Warning] No map found for {item_col}. Assuming indices are Raw IDs.")
                    
            except Exception as e:
                print(f"[Error] Failed to count: {e}")
                return None
        else:
             print("[Error] No 'data_tensor'.")
             return None

        # 排序并截取 Top K%
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        top_n = int(len(unique) * percent)
        
        top_items_list = sorted_counts[:top_n] 
        print(f"[DBSCANSID] Top {percent*100}% = {top_n} items.")
        
        # 获取 Top Items 的 Embeddings
        top_embeddings = []
        
        found_count = 0
        self.top_item_raw_ids_set = set() # Reset
        
        for vocab_idx, _ in tqdm(top_items_list, desc="Retrieving Top Embeddings"):
            # 1. Vocab Index -> Raw ID
            if self.vocab_map is not None:
                if vocab_idx < len(self.vocab_map):
                    raw_id = self.vocab_map[vocab_idx]
                else:
                    continue
            else:
                raw_id = vocab_idx
            
            # 2. Raw ID -> Embedding
            if raw_id in self.id_to_idx:
                idx = self.id_to_idx[raw_id]
                top_embeddings.append(self.embeddings[idx])
                
                # 记录 Raw ID (注意类型统一)
                self.top_item_raw_ids_set.add(raw_id)
                try: self.top_item_raw_ids_set.add(int(raw_id))
                except: pass
                try: self.top_item_raw_ids_set.add(str(raw_id))
                except: pass
                
                found_count += 1
        
        if not top_embeddings:
            print("[Error] No embeddings found for top items.")
            return None

        print(f"[DBSCANSID] Found {found_count} top items embeddings.")
        top_emb_tensor = torch.stack(top_embeddings) # [M, D]
        return top_emb_tensor

    def run(self, save_path, eps=0.5, min_samples=5, k=10, diff_threshold=0.1, percent=0.1):
        """
        执行完整流程
        """
        # ================= Step 1: 获取 Top Items =================
        print("\n=== Step 1: Getting Top % Items ===")
        top_emb_tensor = self.get_top_k_percent_items(percent=percent)
        if top_emb_tensor is None: return

        # ================= Step 2: DBSCAN & Centroids =================
        print(f"\n=== Step 2: DBSCAN (eps={eps}, min={min_samples}) ===")
        X = top_emb_tensor.cpu().numpy()
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=-1).fit(X)
        labels = db.labels_
        
        unique_labels = set(labels)
        if -1 in unique_labels: unique_labels.remove(-1) 
        
        num_clusters = len(unique_labels)
        print(f"[DBSCANSID] Found {num_clusters} clusters.")
        
        if num_clusters == 0:
            print("[Error] No clusters found.")
            return

        # 计算聚类中心
        centroids = []
        # 这里原来的 cluster_ids 是聚类标签，现在我们需要让中心代表 "Item ID"
        # 你的需求里没有明确说 SID 必须是 Cluster ID 还是 Item ID。
        # "取k个最相似的中心item id", 通常意味着我们用聚类中心代表一类语义。
        # 由于 SID 通常是离散整数，我们假设每个 cluster 分配一个新的 ID (0 ~ C-1)
        # 或者使用 Cluster 中最靠近中心的那个 Item 的 ID？
        # 为了简单且符合语义 ID 定义，我们用 Cluster Label (0~N) 作为 ID.
        # 但这里的 ID 必须能被 Embedding Layer 接受。
        # 如果你的下游模型 Embedding 是针对 Raw Item ID 训练的，那这里需要给 Cluster 一个 "Virtual Item ID"。
        # 通常做法：建立一个新的 Codebook，大小为 Num_Clusters。Cluster Label 即为 Codebook Index。
        
        # 修正：需求是 "取k个最相似的中心item id"，这可能意味着用 Top Item 本身作为中心。
        # "DBSCAN 对其进行聚类并保存" -> 保存聚类结果？
        # "每个item于聚类中心计算相似度" -> 这里还是用 Cluster Centroid.
        # "中心 item id" -> 可能是指距离 centroid 最近的那个 item 的 raw id?
        # 为了通用性，这里我们输出 Cluster ID (0...C-1)。
        # 如果需要 Raw ID，需要在下游映射。
        # *假定*: 输出的是 Cluster ID。
        
        cluster_ids = [] 
        
        print("\n=== Step 3: Computing Centroids ===")
        for label in tqdm(unique_labels, desc="Centroids"):
            mask = (labels == label)
            mask_t = torch.from_numpy(mask).to(self.device)
            cluster_points = top_emb_tensor[mask_t]
            centroid = cluster_points.mean(dim=0)
            centroid = F.normalize(centroid, p=2, dim=0)
            
            centroids.append(centroid)
            cluster_ids.append(label) # 即 Cluster ID
            
        centroid_matrix = torch.stack(centroids) # [C, D]

        # ================= Step 4: 映射所有 Item =================
        print("\n=== Step 4: Mapping Items (Special handling for Top Items) ===")
        
        all_sid_rows = [] 
        all_sim_rows = [] 
        
        # 提前准备 Raw IDs 用于核对
        # self.item_ids 是所有 Items 的 Raw ID 列表
        
        batch_size = 1024
        num_items = self.embeddings.size(0)
        
        for i in tqdm(range(0, num_items, batch_size), desc="Mapping"):
            end = min(i + batch_size, num_items)
            
            # 当前 batch 的 embedding
            batch_emb = self.embeddings[i : end] # [B, D]
            # 当前 batch 的 raw ids
            batch_raw_ids = self.item_ids[i : end] 
            
            # 1. 计算所有相似度 (不管是不是 Top)
            sim_scores = torch.matmul(batch_emb, centroid_matrix.t())
            
            # 2. Get Top K
            curr_k = min(k, len(centroids))
            top_vals, top_inds = torch.topk(sim_scores, k=curr_k, dim=1)
            
            top_vals_np = top_vals.cpu().numpy()
            top_inds_np = top_inds.cpu().numpy()
            
            # 3. 逐个处理
            for b_idx in range(len(batch_raw_ids)):
                raw_id = batch_raw_ids[b_idx]
                
                # Check if Top Item
                is_top = False
                if raw_id in self.top_item_raw_ids_set: 
                    is_top = True
                
                # Container
                curr_sids = []
                curr_sims = []
                
                if is_top:
                    # Top Item Rule: [Raw_ID, -1, -1, ...]
                    # 注意：这里的 SID 空间变成了混合空间 (Raw IDs U Cluster IDs)
                    # 如果后续 Embedding Table 是分开的，需要小心。
                    # 通常 Top Item 的 SID 应该是一个特殊的保留 ID 或者它自己的 Cluster ID。
                    # 按照 "它的语义ID为 [item_p, -1 ...]"，这里填入 raw_id。
                    # 为了类型一致 (int32)，确保 raw_id 是数字。
                    try:
                        rid = int(raw_id)
                    except:
                        # 如果 raw_id 是字符串hash或其他，这里可能需要 mapping
                        # 假设都是 int
                        rid = 0 
                        print(f"Warning: Non-int Raw ID {raw_id} for Top Item")

                    curr_sids.append(rid)
                    curr_sims.append(1.0) # Self similarity is 1
                    
                    # 后面全是 -1
                    while len(curr_sids) < k:
                        curr_sids.append(-1)
                        curr_sims.append(0.0) # 或者 -1.0
                        
                else:
                    # Normal Item Rule: Nearest Centroids with dynamic truncation
                    vals = top_vals_np[b_idx]
                    inds = top_inds_np[b_idx]
                    
                    if len(vals) > 0:
                        # First always in
                        curr_sids.append( int(cluster_ids[inds[0]]) )
                        curr_sims.append( float(vals[0]) )
                    
                    # Truncation loop
                    is_truncated = False
                    for j in range(1, len(vals)):
                        diff = vals[j-1] - vals[j]
                        
                        if diff > diff_threshold:
                            is_truncated = True
                            # Truncate: stop adding
                            break
                        
                        curr_sids.append( int(cluster_ids[inds[j]]) )
                        curr_sims.append( float(vals[j]) )
                    
                    # Fill rest with -1
                    while len(curr_sids) < k:
                        curr_sids.append(-1)
                        curr_sims.append(0.0) # 对应 mask 位置，sim 可以是 0
                
                all_sid_rows.append(curr_sids)
                all_sim_rows.append(curr_sims)

        # ================= Step 5: 保存 =================
        os.makedirs(save_path, exist_ok=True)
        
        res_keys = self.item_ids
        res_vals = np.array(all_sid_rows, dtype=np.int32)
        res_sims = np.array(all_sim_rows, dtype=np.float32)

        print(f"\n=== Step 5: Saving results to {save_path} ===")
        print(f"Keys shape: {res_keys.shape}")
        print(f"SIDs shape: {res_vals.shape}")
        print(f"Sims shape: {res_sims.shape}")
        
        np.save(os.path.join(save_path, "dbscan_sid_keys.npy"), res_keys)
        np.save(os.path.join(save_path, "dbscan_sid_values.npy"), res_vals)
        np.save(os.path.join(save_path, "dbscan_sid_sims.npy"), res_sims)
        
        print("[DBSCANSID] Done.")

if __name__ == "__main__":
    pass
