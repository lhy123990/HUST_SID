"""
[Modified] Clustering Based Semantic ID Generator (DBSCAN / K-Means)
----------------------------------------------
核心逻辑:
1. 统计训练集中 Item 出现频次，选取 Top Items.
2. 对 Top Items 的 Embedding 进行聚类 (DBSCAN 或 K-Means).
3. 找到每个 Cluster 的 Medoid (距离中心最近的真实 Item).
4. 映射所有 Item 到这些 Medoids.
"""

import torch
import numpy as np
import os
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
import torch.nn.functional as F
from tqdm import tqdm

class ClusteringSIDGenerator:
    """
    SID 生成器 (支持 DBSCAN 和 KMeans):
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
        if hasattr(scl_dataset, 'item_ids') and isinstance(scl_dataset.item_ids, torch.Tensor):
             self.item_ids = scl_dataset.item_ids.numpy()
        else:
             self.item_ids = getattr(scl_dataset, 'item_ids', [])
             
        if hasattr(scl_dataset, 'embeddings') and isinstance(scl_dataset.embeddings, torch.Tensor):
             self.embeddings = scl_dataset.embeddings.to(device)
        else:
             self.embeddings = torch.tensor(scl_dataset.embeddings).to(device)
        
        print(f"[SIDGenerator] Building ID -> Index map for {len(self.item_ids)} items...")
        self.id_to_idx = {uid: i for i, uid in enumerate(self.item_ids)}
        
        print(f"[SIDGenerator] Loaded {len(self.item_ids)} items from SCLDataset.")
        self.top_item_raw_ids_set = set() 

    def get_top_k_percent_items(self, percent=0.1, item_col='205'):
        """
        统计训练集中出现频次最高的 Top K% Items
        Returns: (Tensor[M, D], List[RawID])
        """
        if self.taobao_dataset is None:
            raise ValueError("需要提供 TaobaoDataset 以统计频次.")
            
        print(f"[SIDGenerator] Counting frequencies (Col: {item_col})...")
        
        counts = {}
        if hasattr(self.taobao_dataset, 'data_tensor'):
            try:
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
                return None, None
        else:
             print("[Error] No 'data_tensor'.")
             return None, None

        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        top_n = int(len(unique) * percent)
        
        top_items_list = sorted_counts[:top_n] 
        print(f"[SIDGenerator] Top {percent*100}% = {top_n} items.")
        
        top_embeddings = []
        top_raw_ids_list = [] 
        
        found_count = 0
        self.top_item_raw_ids_set = set() 
        
        for vocab_idx, _ in tqdm(top_items_list, desc="Retrieving Top Embeddings"):
            if self.vocab_map is not None:
                if vocab_idx < len(self.vocab_map):
                    raw_id = self.vocab_map[vocab_idx]
                else:
                    continue
            else:
                raw_id = vocab_idx
            
            idx = self.id_to_idx.get(raw_id)
            if idx is None:
                try: idx = self.id_to_idx.get(int(raw_id))
                except: pass
            if idx is None:
                try: idx = self.id_to_idx.get(str(raw_id))
                except: pass
                
            if idx is not None:
                top_embeddings.append(self.embeddings[idx])
                top_raw_ids_list.append(raw_id) 
                
                self.top_item_raw_ids_set.add(raw_id)
                try: self.top_item_raw_ids_set.add(int(raw_id))
                except: pass
                try: self.top_item_raw_ids_set.add(str(raw_id))
                except: pass
                found_count += 1
        
        if not top_embeddings:
            print("[Error] No embeddings found for top items.")
            return None, None

        print(f"[SIDGenerator] Found {found_count} top items embeddings.")
        top_emb_tensor = torch.stack(top_embeddings) 
        return top_emb_tensor, top_raw_ids_list

    def run(self, save_path, method='dbscan', n_clusters=1000, eps=0.5, min_samples=5, k=10, diff_threshold=0.1, percent=0.1):
        """
        执行完整流程
        Args:
            method: 'dbscan' or 'kmeans'
            n_clusters: KMeans 的聚类数 (DBSCAN 忽略此参数)
        """
        # ================= Step 1: 获取 Top Items =================
        print("\n=== Step 1: Getting Top % Items ===")
        top_emb_tensor, top_raw_ids_list = self.get_top_k_percent_items(percent=percent)
        if top_emb_tensor is None: return

        X = top_emb_tensor.cpu().numpy()
        labels = None
        
        # ================= Step 2: Clustering =================
        if method == 'kmeans':
            print(f"\n=== Step 2: MiniBatchKMeans (n_clusters={n_clusters}) ===")
            # [Fix] 使用 MiniBatchKMeans 解决大 K 值卡死问题
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters, 
                batch_size=4096, 
                random_state=2026, 
                n_init='auto'
            ).fit(X)
            labels = kmeans.labels_
            
        elif method == 'dbscan':
            print(f"\n=== Step 2: DBSCAN (eps={eps}, min={min_samples}) ===")
            db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=-1).fit(X)
            labels = db.labels_
        else:
            raise ValueError(f"Unknown method: {method}")

        unique_labels = set(labels)
        if -1 in unique_labels: unique_labels.remove(-1) 
        
        num_clusters = len(unique_labels)
        print(f"[SIDGenerator] Found {num_clusters} clusters (Method: {method}).")
        
        if num_clusters == 0:
            print("[Error] No clusters found.")
            return

        # ================= Step 3: Computing Medoids =================
        centroids = []
        cluster_medoid_ids = [] 
        
        print("\n=== Step 3: Computing Medoids (Real Item IDs) ===")
        top_raw_ids_arr = np.array(top_raw_ids_list)
        
        for label in tqdm(unique_labels, desc="Centroids"):
            mask = (labels == label)
            mask_t = torch.from_numpy(mask).to(self.device)
            cluster_points = top_emb_tensor[mask_t] 
            
            # --- Centroid ---
            centroid = cluster_points.mean(dim=0)
            centroid = F.normalize(centroid, p=2, dim=0) 
            centroids.append(centroid)
            
            # --- Medoid (Nearest Real Item) ---
            sims = torch.matmul(cluster_points, centroid) 
            best_local_idx = torch.argmax(sims).item()
            
            global_indices = np.where(mask)[0]
            best_global_idx = global_indices[best_local_idx]
            
            real_item_id = top_raw_ids_arr[best_global_idx]
            cluster_medoid_ids.append(real_item_id)
            
        centroid_matrix = torch.stack(centroids) 

        # ================= Step 4: Mapping =================
        print(f"\n=== Step 4: Mapping Items to Top {k} Medoids (Threshold: {diff_threshold}) ===")
        
        all_sid_rows = [] 
        all_sim_rows = [] 
        
        batch_size = 1024
        num_items = self.embeddings.size(0)
        
        for i in tqdm(range(0, num_items, batch_size), desc="Mapping"):
            end = min(i + batch_size, num_items)
            
            batch_emb = self.embeddings[i : end] 
            batch_raw_ids = self.item_ids[i : end] 
            
            # Sim
            sim_scores = torch.matmul(batch_emb, centroid_matrix.t())
            
            # Top K
            curr_k = min(k, len(centroids))
            top_vals, top_inds = torch.topk(sim_scores, k=curr_k, dim=1)
            
            top_vals_np = top_vals.cpu().numpy()
            top_inds_np = top_inds.cpu().numpy()
            
            for b_idx in range(len(batch_raw_ids)):
                raw_id = batch_raw_ids[b_idx]
                
                is_top = False
                if raw_id in self.top_item_raw_ids_set: is_top = True # Simplified check
                
                curr_sids = []
                curr_sims = []
                
                if is_top:
                    try: rid = int(raw_id)
                    except: rid = 0 
                    curr_sids.append(rid)
                    curr_sims.append(1.0) 
                    while len(curr_sids) < k:
                        curr_sids.append(0); curr_sims.append(0.0)
                else:
                    vals = top_vals_np[b_idx]
                    inds = top_inds_np[b_idx]
                    
                    for j in range(len(vals)):
                        sim_val = float(vals[j])
                        if sim_val < diff_threshold: break
                        curr_sids.append( int(cluster_medoid_ids[inds[j]]) )
                        curr_sims.append( sim_val )
                    
                    while len(curr_sids) < k:
                        curr_sids.append(0); curr_sims.append(0.0)
                
                all_sid_rows.append(curr_sids)
                all_sim_rows.append(curr_sims)

        # ================= Step 5: Saving =================
        os.makedirs(save_path, exist_ok=True)
        res_keys = self.item_ids
        # [Int64 Fix]
        res_vals = np.array(all_sid_rows, dtype=np.int64) 
        res_sims = np.array(all_sim_rows, dtype=np.float32)

        print(f"\n=== Step 5: Saving results to {save_path} ===")
        valid_lens = np.sum(res_vals != 0, axis=1) 
        avg_len = np.mean(valid_lens)
        print(f"[SIDGenerator] Average effective length: {avg_len:.4f}")
        
        np.save(os.path.join(save_path, "dbscan_sid_keys.npy"), res_keys)
        np.save(os.path.join(save_path, "dbscan_sid_values.npy"), res_vals)
        np.save(os.path.join(save_path, "dbscan_sid_sims.npy"), res_sims)
        print("[SIDGenerator] Done.")

if __name__ == "__main__":
    pass
