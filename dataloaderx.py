# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import glob

class TaobaoDataset(Dataset):
    def __init__(self, root, dataset='', mode='train', max_len=20, limit_files=None, use_sid=False, use_pid=False):
        """
        root: Dataset root path
        mode: 'train' or 'test'
        max_len: Sequence max length
        """
        self.root = root
        self.mode = mode
        self.max_len = max_len
        self.use_sid = use_sid
        self.use_pid = use_pid
        
        self.data_dir = os.path.join(self.root, mode)
        self.map_dir = os.path.join(self.root, 'feature_map')
        
        # 1. Setup Files
        self.files = sorted(glob.glob(os.path.join(self.data_dir, f'{mode}-shard-*.parquet')))
        if not self.files:
            raise FileNotFoundError(f"No parquet files found in {self.data_dir}")
        
        if limit_files is not None:
            self.files = self.files[:limit_files]
            
        print(f"[{mode}] Loading {len(self.files)} files...")
        
        # 2. Read Data
        dfs = []
        for f in self.files:
            dfs.append(pd.read_parquet(f))
        
        self.df = pd.concat(dfs, ignore_index=True)
        self.data_num = len(self.df)
        print(f"[{mode}] Total samples: {self.data_num}")

        # 3. Load Maps
        self.maps = {}
        # Initialize lookup tables
        self.sid_lookup_table = None
        self.pid_lookup_table = None
        self.pid_sim_table = None

        map_sharing = { '205': '150_2_180', '206': '151_2_180' }

        for col in self.df.columns:
            if col == 'label_0': continue

            map_path = os.path.join(self.map_dir, f"{col}_sorted_map.npy")
            
            if os.path.exists(map_path):
                print(f"Loading map for {col}...")
                self.maps[col] = np.load(map_path)
            elif col in map_sharing:
                source_col = map_sharing[col]
                source_map_path = os.path.join(self.map_dir, f"{source_col}_sorted_map.npy")
                if os.path.exists(source_map_path):
                    print(f"Loading map for {col} (Shared from {source_col})...")
                    if source_col not in self.maps:
                        self.maps[source_col] = np.load(source_map_path)
                    self.maps[col] = self.maps[source_col]

        # === 3.1 SID Loading (Restored) ===
        if self.use_sid:
            print("Loading SID Data...")
            sid_keys_path = os.path.join(self.map_dir, "semantic_id_keys.npy")
            sid_vals_path = os.path.join(self.map_dir, "semantic_id_values.npy")

            if '205' in self.maps and os.path.exists(sid_keys_path) and os.path.exists(sid_vals_path):
                sid_keys = np.load(sid_keys_path)
                sid_vals = np.load(sid_vals_path)
                
                item_map = self.maps['205']
                max_idx = len(item_map) + 1
                self.num_sid_cols = sid_vals.shape[1]
                
                # Init table
                self.sid_lookup_table = np.zeros((max_idx, self.num_sid_cols), dtype=np.int64)
                
                # Map keys
                key_indices = self._map_values(sid_keys, item_map)
                valid_mask = key_indices > 0
                mapped_keys = key_indices[valid_mask]
                
                # Fill table
                self.sid_lookup_table[mapped_keys] = sid_vals[valid_mask]
                print(f"  SID Table built: {self.sid_lookup_table.shape}")
            else:
                print("  [Warning] Missing SID files. Disabling SID.")
                self.use_sid = False
        
        # === 3.2 PID Loading ===
        if self.use_pid:
            print("Loading PID Data...")
            pid_keys_path = os.path.join(self.map_dir, "dbscan_sid_keys.npy")
            pid_vals_path = os.path.join(self.map_dir, "dbscan_sid_values.npy")
            pid_sims_path = os.path.join(self.map_dir, "dbscan_sid_sims.npy")

            if '205' in self.maps and os.path.exists(pid_keys_path) and os.path.exists(pid_vals_path):
                pid_keys = np.load(pid_keys_path)
                pid_vals = np.load(pid_vals_path)
                pid_sims = np.load(pid_sims_path)
                
                item_map = self.maps['205']
                max_idx = len(item_map) + 1
                k = pid_vals.shape[1]
                
                self.pid_lookup_table = np.zeros((max_idx, k), dtype=np.int64)
                self.pid_sim_table = np.zeros((max_idx, k), dtype=np.float32)
                
                key_indices = self._map_values(pid_keys, item_map)
                valid_mask = key_indices > 0
                mapped_keys = key_indices[valid_mask]
                
                flat_vals = pid_vals[valid_mask].reshape(-1)
                mapped_vals = self._map_values(flat_vals, item_map)
                mapped_vals = mapped_vals.reshape(-1, k)
                
                self.pid_lookup_table[mapped_keys] = mapped_vals
                self.pid_sim_table[mapped_keys] = pid_sims[valid_mask]
                print(f"  PID Table built: {self.pid_lookup_table.shape}, K={k}")
            else:
                print("  [Warning] Missing PID files. Disabling PID.")
                self.use_pid = False

        # === Attribute Lookups ===
        self.attr_lookups = {}
        target_attr_cols = ['206', '213', '214'] 
        
        if '205' in self.maps:
            item_map = self.maps['205']
            max_item_idx = len(item_map) + 1
            for col in target_attr_cols:
                if col in self.df.columns and col in self.maps:
                    print(f"Building lookup table for {col}...")
                    temp_df = self.df[['205', col]].drop_duplicates('205')
                    item_indices = self._map_values(temp_df['205'].values, self.maps['205'])
                    attr_indices = self._map_values(temp_df[col].values, self.maps[col])
                    lookup = np.zeros(max_item_idx, dtype=np.int64)
                    valid_mask = item_indices < max_item_idx
                    lookup[item_indices[valid_mask]] = attr_indices[valid_mask]
                    self.attr_lookups[col] = torch.from_numpy(lookup)
        
        # 4. Process Labels
        print("Processing labels...")
        try:
            label_col = self.df['label_0']
            if len(label_col) > 0 and isinstance(label_col.iloc[0], (list, np.ndarray)):
                 self.labels = np.array([x[1] for x in label_col], dtype=np.float32)
            else:
                 self.labels = label_col.astype(np.float32).values
        except:
            self.labels = np.zeros(self.data_num, dtype=np.float32)
            
        # 5. Process Features & Inject SID
        self.dnn_features = {}
        
        scalar_cols = []
        ignore_cols = ['label_0'] 
        known_seq_cols = ['150_2_180', '151_2_180'] 

        print("Identifying scalar features...")
        for col in self.df.columns:
            if col in ignore_cols or col in known_seq_cols: continue
            if col not in self.maps: continue

            sample_val = self.df[col].iloc[0]
            if isinstance(sample_val, (list, np.ndarray)): continue
            scalar_cols.append(col)

        print(f"Mapping scalar features: {scalar_cols}")
        for col in scalar_cols:
            raw_values = self.df[col].values
            if raw_values.dtype == 'object':
                try: raw_values = raw_values.astype(np.int64)
                except: continue
            
            mapped_values = self._map_values(raw_values, self.maps[col])
            self.dnn_features[col] = torch.from_numpy(mapped_values).long()
        
        # === Inject SID features ===
        # SID features are treated as normal categorical inputs (unlike PID)
        if self.use_sid and self.sid_lookup_table is not None:
             if '205' in self.dnn_features:
                 print("Injecting SID features into DNN input...")
                 item_indices = self.dnn_features['205'].numpy()
                 max_sid_idx = len(self.sid_lookup_table)
                 safe_indices = item_indices.copy()
                 safe_indices[safe_indices >= max_sid_idx] = 0
                 
                 sids = self.sid_lookup_table[safe_indices] # [N, 3]
                 
                 for i in range(sids.shape[1]):
                     feat_name = f'sid_{i}'
                     self.dnn_features[feat_name] = torch.from_numpy(sids[:, i]).long()
                     # Register map for dimension calculation
                     max_val = np.max(sids[:, i])
                     self.maps[feat_name] = np.arange(max_val + 1)

        # === PID Injection Removed ===
        # PID is handled internally by the DCNv2 model via helper lookup tables.
        # We do NOT add 'pid_0', 'pid_1'... to dnn_features here.

        self.feature_names = sorted(list(self.dnn_features.keys()))
        print(f"Final DNN Features: {self.feature_names}")
        
        if len(self.feature_names) > 0:
            self.data_tensor = torch.stack([self.dnn_features[col] for col in self.feature_names], dim=1)
        else:
            self.data_tensor = torch.zeros((self.data_num, 1)).long()

        # 6. Process Sequences
        target_seq_col = '150_2_180'
        print(f"Processing sequence feature {target_seq_col}...")
        self.seq_tensor = np.zeros((self.data_num, self.max_len), dtype=np.int64)
        self.mask_tensor = np.zeros((self.data_num, self.max_len), dtype=np.float32)

        if target_seq_col in self.df.columns and target_seq_col in self.maps:
            seq_series = self.df[target_seq_col].apply(lambda x: x[:self.max_len] if len(x) > self.max_len else x)
            lengths = seq_series.apply(len).values
            flat_seq = np.concatenate(seq_series.values)
            mapped_flat = self._map_values(flat_seq, self.maps[target_seq_col])
            
            cursor = 0
            for i, length in enumerate(lengths):
                if length > 0:
                    self.seq_tensor[i, :length] = mapped_flat[cursor : cursor + length]
                    self.mask_tensor[i, :length] = 1.0
                    cursor += length
            
        self.seq_tensor = torch.from_numpy(self.seq_tensor).long()
        self.mask_tensor = torch.from_numpy(self.mask_tensor).float()
        self.field_num = len(self.feature_names)

    def _map_values(self, values, map_array):
        if not isinstance(values, np.ndarray): values = np.array(values)
        idx = np.searchsorted(map_array, values)
        idx[idx == len(map_array)] = 0
        matched = map_array[idx] == values
        return np.where(matched, idx + 1, 0).astype(np.int64)

    def __len__(self): return self.data_num
    def __getitem__(self, idx):
        return self.data_tensor[idx], self.seq_tensor[idx], self.mask_tensor[idx], self.labels[idx]]
