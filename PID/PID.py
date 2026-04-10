"""
[Modified] Clustering Based Semantic ID Generator (DPP)
----------------------------------------------
核心逻辑:
1. 统�?��??练集�? Item 出现频�?�，选取 Top Items.
2. �? Top Items �? Embedding 进�? DPP 固定大小采样，形�? Global Basis Pool.
3. 对任�? Item 计算其与 Basis Pool 的相似度，取 Top-k.
"""
 
import torch
import numpy as np
import os
from sklearn.cluster import DBSCAN
import torch.nn.functional as F
from tqdm import tqdm

class ClusteringSIDGenerator:
    """
    SID 生成�? (Global Basis -> Local Top-k):
    """
    def __init__(self, scl_dataset, taobao_dataset=None, device='cuda'):
        """
        Args:
            scl_dataset: 提供 Item Embeddings (SCLDataset 实例)
            taobao_dataset: 提供�?练集频�?�统�? (TaobaoDataset 实例)
            device: 计算设�??
        """
        self.scl_dataset = scl_dataset
        self.taobao_dataset = taobao_dataset
        self.device = device
        
        # 加载所�? Item �? ID �? Embeddings
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

    def get_top_k_percent_items(self, percent=0.1, item_col='205', sampling_mode='top', random_percent=None):
        """
        从训练集中选取一部分 Items 作为候选池。
        sampling_mode='top': 频次最高的 Top K%
        sampling_mode='random': 从出现过的 Items 中随机采样 K%
        Returns: (Tensor[M, D], List[RawID])
        """
        if self.taobao_dataset is None:
            raise ValueError("需要提�? TaobaoDataset 以统计�?��??.")

        if sampling_mode not in {'top', 'random'}:
            raise ValueError(f"Unsupported sampling_mode={sampling_mode}, expected 'top' or 'random'.")

        effective_percent = percent if random_percent is None else random_percent
        if effective_percent <= 0 or effective_percent > 1:
            raise ValueError(f"percent must be in (0, 1], got {effective_percent}")

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
        base_n = len(unique)
        sample_n = max(1, int(base_n * effective_percent))

        if sampling_mode == 'top':
            selected_items_list = sorted_counts[:sample_n]
            print(f"[SIDGenerator] Top {effective_percent*100:.2f}% = {sample_n} items.")
        else:
            # Random control: same proportion, sampled from observed items.
            all_items = [kv[0] for kv in sorted_counts]
            picked = np.random.choice(all_items, size=sample_n, replace=False)
            selected_items_list = [(x, 0) for x in picked.tolist()]
            print(f"[SIDGenerator] Random {effective_percent*100:.2f}% = {sample_n} items.")
        
        top_embeddings = []
        top_raw_ids_list = [] 
        
        found_count = 0
        
        retrieve_desc = "Retrieving Top Embeddings" if sampling_mode == 'top' else "Retrieving Random Embeddings"
        for vocab_idx, _ in tqdm(selected_items_list, desc=retrieve_desc):
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
                
                found_count += 1
        
        if not top_embeddings:
            print("[Error] No embeddings found for top items.")
            return None, None

        print(f"[SIDGenerator] Found {found_count} sampled item embeddings.")
        top_emb_tensor = torch.stack(top_embeddings) 
        return top_emb_tensor, top_raw_ids_list

    def select_dpp_basis(self, top_emb_tensor, top_raw_ids_list, basis_size=1000):
        """
        Greedy DPP-MAP on normalized embeddings.
        Returns: (basis_matrix[Mb, D], basis_raw_ids[List])
        """
        if top_emb_tensor is None or top_emb_tensor.numel() == 0:
            return None, None

        X = F.normalize(top_emb_tensor, p=2, dim=1)
        n_items = X.size(0)
        target_m = min(max(1, basis_size), n_items)

        print(f"\n=== Step 2: DPP Basis Sampling (target M={target_m}) ===")

        # Greedy volume maximization in feature space (L = X X^T).
        residual = X.clone()
        residual_norm2 = torch.sum(residual * residual, dim=1)
        selected = []
        selected_mask = torch.zeros(n_items, dtype=torch.bool, device=X.device)

        for _ in tqdm(range(target_m), desc="DPP Selecting"):
            scores = residual_norm2.clone()
            scores[selected_mask] = -1.0

            best_idx = int(torch.argmax(scores).item())
            #if scores[best_idx] <= 1e-12:
             #   break

            selected.append(best_idx)
            selected_mask[best_idx] = True

            v = residual[best_idx]
            v_norm = torch.norm(v) + 1e-12
            u = v / v_norm
            proj_coeff = torch.matmul(residual, u).unsqueeze(1)
            residual = residual - proj_coeff * u.unsqueeze(0)
            residual_norm2 = torch.sum(residual * residual, dim=1)

        if len(selected) == 0:
            print("[Error] DPP failed to select basis items.")
            return None, None

        basis_matrix = X[selected]
        basis_raw_ids = [top_raw_ids_list[i] for i in selected]
        print(f"[SIDGenerator] DPP selected {len(basis_raw_ids)} basis items.")
        return basis_matrix, basis_raw_ids

    def select_dbscan_basis(self, top_emb_tensor, top_raw_ids_list, eps=0.3, min_samples=3):
        """
        DBSCAN over top-item embeddings, then use medoids as global basis.
        Returns: (basis_matrix[Mb, D], basis_raw_ids[List])
        """
        if top_emb_tensor is None or top_emb_tensor.numel() == 0:
            return None, None

        X = F.normalize(top_emb_tensor, p=2, dim=1)
        X_np = X.detach().cpu().numpy()
        print(f"\n=== Step 2: DBSCAN Basis Building (eps={eps}, min_samples={min_samples}) ===")

        db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=-1).fit(X_np)
        labels = db.labels_

        unique_labels = sorted(set(labels.tolist()))
        if -1 in unique_labels:
            unique_labels.remove(-1)
        if len(unique_labels) == 0:
            print("[Error] DBSCAN found no valid clusters.")
            return None, None

        basis_vectors = []
        basis_raw_ids = []
        top_raw_ids_arr = np.array(top_raw_ids_list)

        for label in tqdm(unique_labels, desc="DBSCAN Medoids"):
            mask = (labels == label)
            if not np.any(mask):
                continue

            cluster_points = X[torch.from_numpy(mask).to(X.device)]
            centroid = F.normalize(cluster_points.mean(dim=0), p=2, dim=0)
            sims = torch.matmul(cluster_points, centroid)
            best_local_idx = int(torch.argmax(sims).item())

            global_indices = np.where(mask)[0]
            best_global_idx = int(global_indices[best_local_idx])
            basis_vectors.append(X[best_global_idx])
            basis_raw_ids.append(top_raw_ids_arr[best_global_idx])

        if len(basis_vectors) == 0:
            print("[Error] DBSCAN did not produce usable medoids.")
            return None, None

        basis_matrix = torch.stack(basis_vectors)
        print(f"[SIDGenerator] DBSCAN selected {len(basis_raw_ids)} basis items.")
        return basis_matrix, basis_raw_ids

    def select_random_basis(self, top_emb_tensor, top_raw_ids_list, basis_size=1000):
        """
        Randomly select a fixed number of items from the top items globally.
        Returns: (basis_matrix[Mb, D], basis_raw_ids[List])
        """
        if top_emb_tensor is None or top_emb_tensor.numel() == 0:
            return None, None

        X = F.normalize(top_emb_tensor, p=2, dim=1)
        n_items = X.size(0)
        target_m = min(max(1, basis_size), n_items)

        print(f"\n=== Step 2: Random Basis Sampling (target M={target_m}) ===")
        
        # Randomly sample indices without replacement
        selected = torch.randperm(n_items)[:target_m].tolist()

        basis_matrix = X[selected]
        basis_raw_ids = [top_raw_ids_list[i] for i in selected]
        print(f"[SIDGenerator] Random selected {len(basis_raw_ids)} basis items.")
        return basis_matrix, basis_raw_ids

    def _to_int_raw_id(self, raw_id):
        try:
            return int(raw_id)
        except Exception:
            return None

    def merge_basis_ids(self, dpp_basis_raw_ids, dbscan_basis_raw_ids):
        """
        Merge two basis ID lists with de-duplication while keeping stable order.
        """
        merged = []
        seen = set()

        for rid in list(dpp_basis_raw_ids) + list(dbscan_basis_raw_ids):
            rid_int = self._to_int_raw_id(rid)
            if rid_int is None or rid_int in seen:
                continue
            seen.add(rid_int)
            merged.append(rid_int)
        return merged

    def basis_matrix_from_raw_ids(self, basis_raw_ids):
        """
        Build normalized basis embedding matrix from raw IDs.
        """
        basis_vecs = []
        valid_ids = []

        for rid in basis_raw_ids:
            idx = self.id_to_idx.get(rid)
            if idx is None:
                idx = self.id_to_idx.get(str(rid))
            if idx is None:
                idx = self.id_to_idx.get(int(rid)) if self._to_int_raw_id(rid) is not None else None
            if idx is None:
                continue

            basis_vecs.append(self.embeddings[idx])
            valid_ids.append(int(rid))

        if len(basis_vecs) == 0:
            return None, None

        basis_matrix = F.normalize(torch.stack(basis_vecs), p=2, dim=1)
        return basis_matrix, valid_ids

    def run(
        self,
        save_path,
        method='dpp',
        basis_size=1000,
        eps=0.3,
        min_samples=3,
        k=10,
        diff_threshold=0.1,
        percent=0.1,
        sampling_mode='top',
        random_percent=None,
        force_basis=False,
        merge_basis=False,
        write_legacy_dbscan_alias=False,
    ):
        """
        执�?�完整流�?
        Args:
            method: 'dpp', 'dbscan', 'random' or 'hybrid'
            basis_size: DPP Global Basis Pool size M
            merge_basis: if True, run DPP+DBSCAN and merge their basis IDs.
            write_legacy_dbscan_alias: if True, also writes dbscan_sid_* as alias.
        """
        # ================= Step 1: 获取 Top Items =================
        print("\n=== Step 1: Getting Top % Items ===")
        top_emb_tensor, top_raw_ids_list = self.get_top_k_percent_items(
            percent=percent,
            sampling_mode=sampling_mode,
            random_percent=random_percent,
        )
        if top_emb_tensor is None: return

        effective_method = 'hybrid' if merge_basis else method

        if effective_method == 'dpp':
            basis_matrix, basis_raw_ids = self.select_dpp_basis(
                top_emb_tensor,
                top_raw_ids_list,
                basis_size=basis_size,
            )
        elif effective_method == 'random':
            basis_matrix, basis_raw_ids = self.select_random_basis(
                top_emb_tensor,
                top_raw_ids_list,
                basis_size=basis_size,
            )
        elif effective_method == 'dbscan':
            basis_matrix, basis_raw_ids = self.select_dbscan_basis(
                top_emb_tensor,
                top_raw_ids_list,
                eps=eps,
                min_samples=min_samples,
            )
        elif effective_method == 'hybrid':
            dpp_basis_matrix, dpp_basis_raw_ids = self.select_dpp_basis(
                top_emb_tensor,
                top_raw_ids_list,
                basis_size=basis_size,
            )
            dbscan_basis_matrix, dbscan_basis_raw_ids = self.select_dbscan_basis(
                top_emb_tensor,
                top_raw_ids_list,
                eps=eps,
                min_samples=min_samples,
            )

            if dpp_basis_matrix is None or dbscan_basis_matrix is None:
                print("[Error] Hybrid mode failed: one of DPP/DBSCAN basis pools is empty.")
                return

            merged_basis_raw_ids = self.merge_basis_ids(dpp_basis_raw_ids, dbscan_basis_raw_ids)
            basis_matrix, basis_raw_ids = self.basis_matrix_from_raw_ids(merged_basis_raw_ids)
            if basis_matrix is None:
                print("[Error] Hybrid mode failed: no valid merged basis IDs found in embedding table.")
                return

            print(
                f"[SIDGenerator] Hybrid merged basis size: {len(basis_raw_ids)} "
                f"(DPP={len(dpp_basis_raw_ids)}, DBSCAN={len(dbscan_basis_raw_ids)})"
            )
        else:
            raise ValueError(f"Unknown method: {method}. Expected 'dpp', 'random', 'dbscan' or 'hybrid'.")

        if basis_matrix is None:
            return

        # ================= Step 3: Mapping =================
        print(f"\n=== Step 3: Mapping Items to Top {k} Basis Items (Threshold: {diff_threshold}) ===")
        
        all_sid_rows = [] 
        all_sim_rows = [] 
        
        batch_size = 1024
        num_items = self.embeddings.size(0)
        
        basis_set = set(int(rid) for raw in basis_raw_ids for rid in ([raw] if not isinstance(raw, list) else raw)) if force_basis else set()

        for i in tqdm(range(0, num_items, batch_size), desc="Mapping"):
            end = min(i + batch_size, num_items)
            
            batch_emb = self.embeddings[i : end] 
            batch_raw_ids = self.item_ids[i : end] 
            batch_emb = F.normalize(batch_emb, p=2, dim=1)
            
            # Sim
            sim_scores = torch.matmul(batch_emb, basis_matrix.t())
            
            # Top K
            curr_k = min(k, basis_matrix.size(0))
            # 1. 计算绝对值

            top_vals, top_inds = torch.topk(sim_scores, k=curr_k, dim=1)
            
            top_vals_np = top_vals.cpu().numpy()
            top_inds_np = top_inds.cpu().numpy()
            
            for b_idx in range(len(batch_raw_ids)):
                raw_id_val = int(batch_raw_ids[b_idx])
                if force_basis and raw_id_val in basis_set:
                    curr_sids = [raw_id_val] + [0] * (k - 1)
                    curr_sims = [1.0] + [0.0] * (k - 1)
                    all_sid_rows.append(curr_sids[:k])
                    all_sim_rows.append(curr_sims[:k])
                    continue

                curr_sids = []
                curr_sims = []

                vals = top_vals_np[b_idx]
                inds = top_inds_np[b_idx]

                for j in range(len(vals)):
                    sim_val = float(vals[j])
                    if sim_val< diff_threshold:
                        break
                    curr_sids.append(int(basis_raw_ids[inds[j]]))
                    curr_sims.append(sim_val)

                while len(curr_sids) < k:
                    curr_sids.append(0)
                    curr_sims.append(0.0)
                
                all_sid_rows.append(curr_sids)
                all_sim_rows.append(curr_sims)

        # ================= Step 4: Saving =================
        os.makedirs(save_path, exist_ok=True)
        res_keys = self.item_ids
        # [Int64 Fix]
        res_vals = np.array(all_sid_rows, dtype=np.int64) 
        res_sims = np.array(all_sim_rows, dtype=np.float32)

        print(f"\n=== Step 4: Saving results to {save_path} ===")
        valid_lens = np.sum(res_vals != 0, axis=1) 
        empty_count = np.sum(valid_lens == 0)
        empty_ratio = empty_count / len(valid_lens) if len(valid_lens) > 0 else 0
        avg_len = np.mean(valid_lens)
        
        print(f"[SIDGenerator] Empty PID items (no basis above diff_threshold): {empty_count} / {len(valid_lens)} ({empty_ratio*100:.2f}%)")
        print(f"[SIDGenerator] Average effective length: {avg_len:.4f}")
        
        if effective_method == 'dpp':
            np.save(os.path.join(save_path, "dpp_sid_keys.npy"), res_keys)
            np.save(os.path.join(save_path, "dpp_sid_values.npy"), res_vals)
            np.save(os.path.join(save_path, "dpp_sid_sims.npy"), res_sims)
            np.save(os.path.join(save_path, "dpp_basis_raw_ids.npy"), np.array(basis_raw_ids, dtype=np.int64))
        elif effective_method == 'random':
            np.save(os.path.join(save_path, "random_sid_keys.npy"), res_keys)
            np.save(os.path.join(save_path, "random_sid_values.npy"), res_vals)
            np.save(os.path.join(save_path, "random_sid_sims.npy"), res_sims)
            np.save(os.path.join(save_path, "random_basis_raw_ids.npy"), np.array(basis_raw_ids, dtype=np.int64))
        elif effective_method == 'dbscan':
            np.save(os.path.join(save_path, "dbscan_sid_keys.npy"), res_keys)
            np.save(os.path.join(save_path, "dbscan_sid_values.npy"), res_vals)
            np.save(os.path.join(save_path, "dbscan_sid_sims.npy"), res_sims)
            np.save(os.path.join(save_path, "dbscan_basis_raw_ids.npy"), np.array(basis_raw_ids, dtype=np.int64))
        elif effective_method == 'hybrid':
            np.save(os.path.join(save_path, "hybrid_sid_keys.npy"), res_keys)
            np.save(os.path.join(save_path, "hybrid_sid_values.npy"), res_vals)
            np.save(os.path.join(save_path, "hybrid_sid_sims.npy"), res_sims)
            np.save(os.path.join(save_path, "hybrid_basis_raw_ids.npy"), np.array(basis_raw_ids, dtype=np.int64))

        # Optional alias for older downstream code that only reads dbscan_sid_*.
        if write_legacy_dbscan_alias and effective_method == 'dpp':
            np.save(os.path.join(save_path, "dbscan_sid_keys.npy"), res_keys)
            np.save(os.path.join(save_path, "dbscan_sid_values.npy"), res_vals)
            np.save(os.path.join(save_path, "dbscan_sid_sims.npy"), res_sims)
        print("[SIDGenerator] Done.")

if __name__ == "__main__":
    pass
