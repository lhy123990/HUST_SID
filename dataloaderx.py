# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import glob

class TaobaoDataset(Dataset):
    def __init__(self, root, dataset='', mode='train', max_len=20, limit_files=None, use_sid=False):
        """
        root: 数据集根�?�?
        mode: 'train' or 'test'
        max_len: 序列最大长�?
        use_sid: �?否加载并使用 Semantic ID 特征
        """
        self.root = root
        self.mode = mode
        self.max_len = max_len
        self.use_sid = use_sid
        self.data_dir = os.path.join(self.root, mode)
        self.map_dir = os.path.join(self.root, 'feature_map')
        
        # 1. �?�? Parquet 文件
        self.files = sorted(glob.glob(os.path.join(self.data_dir, f'{mode}-shard-*.parquet')))
        if not self.files:
            raise FileNotFoundError(f"No parquet files found in {self.data_dir}")
        
        if limit_files is not None:
            self.files = self.files[:limit_files]
            
        print(f"[{mode}] Loading {len(self.files)} files...")
        
        # 2. 读取数据 (合并所有分�?)
        dfs = []
        for f in self.files:
            dfs.append(pd.read_parquet(f))
        
        self.df = pd.concat(dfs, ignore_index=True)
        self.data_num = len(self.df)
        print(f"[{mode}] Total samples: {self.data_num}")

        # 3. 加载映射�? (Feature Maps)
        self.maps = {}
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

        # === 3.1 加载 Semantic IDs ===
        self.sid_lookup_table = None
        if self.use_sid:
            print("Loading Semantic IDs...")
            sid_keys_path = os.path.join(self.map_dir, "semantic_id_keys.npy")
            sid_vals_path = os.path.join(self.map_dir, "semantic_id_values.npy")
            
            # 前置条件：必须有 205 (Item ID) �? Map
            if '205' in self.maps and os.path.exists(sid_keys_path) and os.path.exists(sid_vals_path):
                sid_keys = np.load(sid_keys_path) # [Unique Items]
                sid_vals = np.load(sid_vals_path) # [Unique Items, Num_Codebooks]
                
                # 构建 ItemIndex -> SIDs 的查找表
                item_map = self.maps['205']
                max_idx = len(item_map) + 1
                num_books = sid_vals.shape[1]
                
                self.sid_lookup_table = np.zeros((max_idx, num_books), dtype=np.int64)
                
                # �j�配
                # 找到 sid_keys �? item_map �?的位�?
                indices = np.searchsorted(item_map, sid_keys)
                # 校验匹配
                valid_mask = (indices < len(item_map)) & (item_map[indices] == sid_keys)
                
                mapped_indices = indices[valid_mask] + 1
                self.sid_lookup_table[mapped_indices] = sid_vals[valid_mask]
                print(f"  SID Table built: {self.sid_lookup_table.shape}")
            else:
                print("  [Warning] Missing Map(205) or SID files. Disabling SID.")
                self.use_sid = False

        # === 新增: 构建 Item 属性查找表 (206, 213, 214) ===
        self.attr_lookups = {}
        # 用户指定的属性列
        target_attr_cols = ['206', '213', '214'] 
        
        if '205' in self.maps:
            item_map = self.maps['205']
            max_item_idx = len(item_map) + 1
            
            for col in target_attr_cols:
                # 确保该属性列存在于数据和映射中
                if col in self.df.columns and col in self.maps:
                    print(f"Building lookup table for {col}...")
                    
                    # 提取 ItemID 和 属性列 的对应关系 (去重)
                    # 假设每个 Item ID 对应唯一的属性值
                    temp_df = self.df[['205', col]].drop_duplicates('205')
                    
                    # 将原始值映射为 ID
                    item_indices = self._map_values(temp_df['205'].values, self.maps['205'])
                    attr_indices = self._map_values(temp_df[col].values, self.maps[col])
                    
                    # 构建查找表 [Item_ID] -> Attr_ID
                    lookup = np.zeros(max_item_idx, dtype=np.int64)
                    # 注意处理越界或不匹配的情况
                    valid_mask = item_indices < max_item_idx
                    lookup[item_indices[valid_mask]] = attr_indices[valid_mask]
                    
                    self.attr_lookups[col] = torch.from_numpy(lookup)
        
        # 4. 预�理：Label
        print("Processing labels...")
        try:
            label_col = self.df['label_0']
            if len(label_col) > 0 and isinstance(label_col.iloc[0], (list, np.ndarray)):
                 self.labels = np.array([x[1] for x in label_col], dtype=np.float32)
            else:
                 self.labels = label_col.astype(np.float32).values
        except Exception as e:
            self.labels = np.zeros(self.data_num, dtype=np.float32)
            
        # 5. 预�?�理：标量特�? + SID注入
        self.dnn_features = {}
        processed_cols = set()
        
        scalar_cols = []
        ignore_cols = ['label_0'] 
        known_seq_cols = ['150_2_180', '151_2_180'] 

        print("Identifying scalar features...")
        for col in self.df.columns:
            if col in ignore_cols or col in known_seq_cols: continue
            if col not in self.maps: continue

            sample_val = self.df[col].iloc[0]
            if isinstance(sample_val, (list, np.ndarray)):
                continue
            
            scalar_cols.append(col)

        print(f"Mapping scalar features: {scalar_cols}")
        for col in scalar_cols:
            raw_values = self.df[col].values
            if raw_values.dtype == 'object':
                try: raw_values = raw_values.astype(np.int64)
                except: continue
            
            mapped_values = self._map_values(raw_values, self.maps[col])
            self.dnn_features[col] = torch.from_numpy(mapped_values).long()
        
        # === Inject SID columns ===
        if self.use_sid and self.sid_lookup_table is not None:
             # 获取 Item ID 列的映射�? tensor
             if '205' in self.dnn_features:
                 print("Injecting SID features into DNN input...")
                 item_indices = self.dnn_features['205'].numpy()
                 sids = self.sid_lookup_table[item_indices] # [N, 3]
                 
                 for i in range(sids.shape[1]):
                     feat_name = f'sid_{i}'
                     self.dnn_features[feat_name] = torch.from_numpy(sids[:, i]).long()
                     
                     # 构�? dummy map 以便 train.py 计算 vocab size
                     # 假�?? SID 值域�?紧凑的，最大值即�? max(sid)
                     max_val = np.max(sids[:, i])
                     # 为了安全，这里用 unique values
                     # 但为�? Config 通用性，直接造一�? fake map �?�? len 够就�?
                     self.maps[feat_name] = np.arange(max_val + 1)
        
        self.feature_names = sorted(list(self.dnn_features.keys()))
        print(f"Final DNN Features: {self.feature_names}")
        
        if len(self.feature_names) > 0:
            self.data_tensor = torch.stack([self.dnn_features[col] for col in self.feature_names], dim=1)
        else:
            self.data_tensor = torch.zeros((self.data_num, 1)).long()

        # 6. 预�?�理：序列特�?
        target_seq_col = '150_2_180'
        
        print(f"Processing sequence feature {target_seq_col}...")
        if target_seq_col in self.df.columns and target_seq_col in self.maps:
            seq_series = self.df[target_seq_col].apply(lambda x: x[:self.max_len] if len(x) > self.max_len else x)
            lengths = seq_series.apply(len).values
            flat_seq = np.concatenate(seq_series.values)
            mapped_flat = self._map_values(flat_seq, self.maps[target_seq_col])
            
            self.seq_tensor = np.zeros((self.data_num, self.max_len), dtype=np.int64)
            self.mask_tensor = np.zeros((self.data_num, self.max_len), dtype=np.float32)
            
            cursor = 0
            for i, length in enumerate(lengths):
                if length > 0:
                    self.seq_tensor[i, :length] = mapped_flat[cursor : cursor + length]
                    self.mask_tensor[i, :length] = 1.0
                    cursor += length
        else:
            self.seq_tensor = np.zeros((self.data_num, self.max_len), dtype=np.int64)
            self.mask_tensor = np.zeros((self.data_num, self.max_len), dtype=np.float32)
            
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
        return self.data_tensor[idx], self.seq_tensor[idx], self.mask_tensor[idx], self.labels[idx]