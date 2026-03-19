import torch
from torch.utils.data import Dataset
import numpy as np
import os
import torch.nn.functional as F

class SCLDataset(Dataset):
    def __init__(self, feature_map_dir):
        """
        Modified Dataset to load SCL Embeddings from Feature Map files directly.
        Input: feature_map_dir (path to the folder containing .npy files)
        """
        self.feature_map_dir = feature_map_dir
        
        # Load keys (Item IDs) and values (Embeddings)
        keys_path = os.path.join(feature_map_dir, "scl_emb_int8_p90_keys.npy")
        values_path = os.path.join(feature_map_dir, "scl_emb_int8_p90_values.npy")
        
        if not os.path.exists(keys_path) or not os.path.exists(values_path):
            raise FileNotFoundError(f"Missing SCL embedding files in {feature_map_dir}")

        print(f"Loading SCL keys from {keys_path}...")
        self.item_ids = np.load(keys_path)
        
        print(f"Loading SCL values from {values_path}...")
        self.embeddings_raw = np.load(values_path)
        
        print(f"Loaded {len(self.item_ids)} items.")
        print(f"Embedding Shape: {self.embeddings_raw.shape}") # Should be [N, 128]
        
        # Convert int8 -> float32 and Normalize
        # int8 range [-128, 127] -> float
        embeddings_float = torch.from_numpy(self.embeddings_raw).float()
        
        # Important: RQ-VAE with Cosine Distance usually expects normalized inputs
        # Even with L2 distance, normalization helps training stability for embeddings
        self.embeddings = F.normalize(embeddings_float, p=2, dim=1)
        
        # Ensure item_ids are LongTensor
        self.item_ids = torch.from_numpy(self.item_ids).long()

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        # Return (Item_ID, Embedding)
        return self.item_ids[index], self.embeddings[index]