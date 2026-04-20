import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


@dataclass
class FeatureSpec:
    user_dense_feature_names: list
    user_sparse_feature_names: list
    user_sparse_vocab_sizes: list
    item_dense_feature_names: list
    item_sparse_feature_names: list
    item_sparse_vocab_sizes: list


class KuaiRecCachedDataset(Dataset):
    """KuaiRec dataset with offline cache and DIN-ready fields.

    Returns per sample:
    - user_id
    - target_video_id
    - hist_video_ids
    - hist_mask
    - label

    User/item feature lookup tables are cached and exposed as dataset attributes.
    Model can gather all user/item features by id from these tables.
    """

    def __init__(
        self,
        root="/data/cbn01/KuaiRec",
        mode="train",
        max_len=50,
        like_threshold=3.0,
        min_hist_len=1,
        test_ratio=0.2,
        use_sid=True,
        sid_dir="",
        sid_keys_filename="semantic_id_keys.npy",
        sid_values_filename="semantic_id_values.npy",
        rebuild_cache=False,
    ):
        self.root = root
        self.mode = mode
        self.max_len = max_len
        self.like_threshold = like_threshold
        self.min_hist_len = min_hist_len
        self.test_ratio = float(test_ratio)
        self.use_sid = bool(use_sid)
        self.sid_dir = sid_dir
        self.sid_keys_filename = sid_keys_filename
        self.sid_values_filename = sid_values_filename

        self.data_dir = root if root.endswith("/data") else os.path.join(root, "data")
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Missing data dir: {self.data_dir}")

        self.cache_version = "kuairec_v2_all_features"
        self.cache_tag = (
            f"{self.cache_version}_len{max_len}_thr{like_threshold}_hist{min_hist_len}"
            f"_testr{self.test_ratio}"
        )
        self.cache_dir = os.path.join(root, "cached_data", self.cache_tag)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.paths = {
            "user_ids": os.path.join(self.cache_dir, "user_ids.npy"),
            "target_video_ids": os.path.join(self.cache_dir, "target_video_ids.npy"),
            "hist_video_ids": os.path.join(self.cache_dir, "hist_video_ids.npy"),
            "hist_masks": os.path.join(self.cache_dir, "hist_masks.npy"),
            "labels": os.path.join(self.cache_dir, "labels.npy"),
            "train_indices": os.path.join(self.cache_dir, "train_indices.npy"),
            "test_indices": os.path.join(self.cache_dir, "test_indices.npy"),
            "user_sparse_lookup": os.path.join(self.cache_dir, "user_sparse_lookup.npy"),
            "user_dense_lookup": os.path.join(self.cache_dir, "user_dense_lookup.npy"),
            "item_sparse_lookup": os.path.join(self.cache_dir, "item_sparse_lookup.npy"),
            "item_dense_lookup": os.path.join(self.cache_dir, "item_dense_lookup.npy"),
            "meta": os.path.join(self.cache_dir, "meta.npz"),
        }

        if rebuild_cache or not self._cache_exists():
            self._build_cache()

        self._load_cache()
        self._load_sid_lookup()

    def _cache_exists(self):
        return all(os.path.exists(p) for p in self.paths.values())

    def _build_cache(self):
        print(f"[KuaiRecCachedDataset] Building cache at: {self.cache_dir}")

        matrix_path = os.path.join(self.data_dir, "big_matrix.csv")
        user_path = os.path.join(self.data_dir, "user_features.csv")
        item_path = os.path.join(self.data_dir, "item_daily_features.csv")

        if not os.path.exists(matrix_path):
            raise FileNotFoundError(f"Missing file: {matrix_path}")
        if not os.path.exists(user_path):
            raise FileNotFoundError(f"Missing file: {user_path}")
        if not os.path.exists(item_path):
            raise FileNotFoundError(f"Missing file: {item_path}")

        print("[Step 1/4] Loading big_matrix.csv ...")
        df = pd.read_csv(matrix_path, usecols=["user_id", "video_id", "watch_ratio"])
        df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").fillna(0).astype(np.int64)
        df["video_id"] = pd.to_numeric(df["video_id"], errors="coerce").fillna(0).astype(np.int64)
        df["watch_ratio"] = pd.to_numeric(df["watch_ratio"], errors="coerce").fillna(0.0).astype(np.float32)

        # Keep original order as pseudo timestamp.
        df["_row_order"] = np.arange(len(df), dtype=np.int64)
        df = df.sort_values(["user_id", "_row_order"]).reset_index(drop=True)

        max_user_id = int(df["user_id"].max()) if len(df) > 0 else 0
        max_video_id = int(df["video_id"].max()) if len(df) > 0 else 0

        user_dense_feature_names = [
            "is_lowactive_period",
            "is_live_streamer",
            "is_video_author",
            "follow_user_num",
            "fans_user_num",
            "friend_user_num",
            "register_days",
        ]
        user_str_cate_cols = [
            "user_active_degree",
            "follow_user_num_range",
            "fans_user_num_range",
            "friend_user_num_range",
            "register_days_range",
        ]
        user_int_cate_cols = [f"onehot_feat{i}" for i in range(18)]
        user_sparse_feature_names = user_str_cate_cols + user_int_cate_cols

        item_dense_feature_names = [
            "video_duration",
            "video_width",
            "video_height",
            "show_cnt",
            "show_user_num",
            "play_cnt",
            "play_user_num",
            "play_duration",
            "complete_play_cnt",
            "complete_play_user_num",
            "valid_play_cnt",
            "valid_play_user_num",
            "long_time_play_cnt",
            "long_time_play_user_num",
            "short_time_play_cnt",
            "short_time_play_user_num",
            "play_progress",
            "comment_stay_duration",
            "like_cnt",
            "like_user_num",
            "click_like_cnt",
            "double_click_cnt",
            "cancel_like_cnt",
            "cancel_like_user_num",
            "comment_cnt",
            "comment_user_num",
            "direct_comment_cnt",
            "reply_comment_cnt",
            "delete_comment_cnt",
            "delete_comment_user_num",
            "comment_like_cnt",
            "comment_like_user_num",
            "follow_cnt",
            "follow_user_num",
            "cancel_follow_cnt",
            "cancel_follow_user_num",
            "share_cnt",
            "share_user_num",
            "download_cnt",
            "download_user_num",
            "report_cnt",
            "report_user_num",
            "reduce_similar_cnt",
            "reduce_similar_user_num",
            "collect_cnt",
            "collect_user_num",
            "cancel_collect_cnt",
            "cancel_collect_user_num",
        ]
        item_str_cate_cols = [
            "video_type",
            "upload_dt",
            "upload_type",
            "visible_status",
            "video_tag_name",
        ]
        item_int_cate_cols = ["date", "author_id", "music_id", "video_tag_id"]
        item_sparse_feature_names = item_str_cate_cols + item_int_cate_cols

        print("[Step 2/4] Encoding user feature lookup ...")
        user_df = pd.read_csv(user_path)
        user_df["user_id"] = pd.to_numeric(user_df["user_id"], errors="coerce").fillna(0).astype(np.int64)
        user_df = user_df.drop_duplicates("user_id", keep="last")
        if len(user_df) > 0:
            max_user_id = max(max_user_id, int(user_df["user_id"].max()))

        user_sparse_lookup = np.zeros((max_user_id + 1, len(user_sparse_feature_names)), dtype=np.int64)
        user_dense_lookup = np.zeros((max_user_id + 1, len(user_dense_feature_names)), dtype=np.float32)

        user_str_maps = {}
        user_int_maps = {}
        user_sparse_vocab_sizes = []

        for col in user_str_cate_cols:
            if col not in user_df.columns:
                user_df[col] = "UNKNOWN"
            user_df[col] = user_df[col].fillna("UNKNOWN").astype(str)
            uniq = sorted(user_df[col].unique().tolist())
            user_str_maps[col] = {v: i + 1 for i, v in enumerate(uniq)}
            user_sparse_vocab_sizes.append(len(uniq) + 1)

        for col in user_int_cate_cols:
            if col not in user_df.columns:
                user_df[col] = -1
            user_df[col] = pd.to_numeric(user_df[col], errors="coerce").fillna(-1).astype(np.int64)
            uniq = sorted(user_df[col].unique().tolist())
            user_int_maps[col] = {v: i + 1 for i, v in enumerate(uniq)}
            user_sparse_vocab_sizes.append(len(uniq) + 1)

        for col in user_dense_feature_names:
            if col not in user_df.columns:
                user_df[col] = 0.0
            user_df[col] = pd.to_numeric(user_df[col], errors="coerce").fillna(0.0).astype(np.float32)

        for row in tqdm(user_df.itertuples(index=False), total=len(user_df), desc="User lookup"):
            uid = int(getattr(row, "user_id"))
            if uid < 0 or uid >= user_sparse_lookup.shape[0]:
                continue

            sparse_vals = []
            for col in user_str_cate_cols:
                raw = str(getattr(row, col))
                sparse_vals.append(user_str_maps[col].get(raw, 0))
            for col in user_int_cate_cols:
                raw = int(getattr(row, col))
                sparse_vals.append(user_int_maps[col].get(raw, 0))

            dense_vals = [float(getattr(row, col)) for col in user_dense_feature_names]
            user_sparse_lookup[uid] = np.asarray(sparse_vals, dtype=np.int64)
            user_dense_lookup[uid] = np.asarray(dense_vals, dtype=np.float32)

        print("[Step 3/4] Encoding item feature lookup ...")
        item_df = pd.read_csv(item_path)
        item_df["video_id"] = pd.to_numeric(item_df["video_id"], errors="coerce").fillna(0).astype(np.int64)
        if "date" in item_df.columns:
            item_df["date"] = pd.to_numeric(item_df["date"], errors="coerce").fillna(-1).astype(np.int64)
            item_df = item_df.sort_values(["video_id", "date"]).drop_duplicates("video_id", keep="last")
        else:
            item_df = item_df.drop_duplicates("video_id", keep="last")

        if len(item_df) > 0:
            max_video_id = max(max_video_id, int(item_df["video_id"].max()))

        item_sparse_lookup = np.zeros((max_video_id + 1, len(item_sparse_feature_names)), dtype=np.int64)
        item_dense_lookup = np.zeros((max_video_id + 1, len(item_dense_feature_names)), dtype=np.float32)

        item_str_maps = {}
        item_int_maps = {}
        item_sparse_vocab_sizes = []

        for col in item_str_cate_cols:
            if col not in item_df.columns:
                item_df[col] = "UNKNOWN"
            item_df[col] = item_df[col].fillna("UNKNOWN").astype(str)
            uniq = sorted(item_df[col].unique().tolist())
            item_str_maps[col] = {v: i + 1 for i, v in enumerate(uniq)}
            item_sparse_vocab_sizes.append(len(uniq) + 1)

        for col in item_int_cate_cols:
            if col not in item_df.columns:
                item_df[col] = -1
            item_df[col] = pd.to_numeric(item_df[col], errors="coerce").fillna(-1).astype(np.int64)
            uniq = sorted(item_df[col].unique().tolist())
            item_int_maps[col] = {v: i + 1 for i, v in enumerate(uniq)}
            item_sparse_vocab_sizes.append(len(uniq) + 1)

        for col in item_dense_feature_names:
            if col not in item_df.columns:
                item_df[col] = 0.0
            item_df[col] = pd.to_numeric(item_df[col], errors="coerce").fillna(0.0).astype(np.float32)

        for row in tqdm(item_df.itertuples(index=False), total=len(item_df), desc="Item lookup"):
            vid = int(getattr(row, "video_id"))
            if vid < 0 or vid >= item_sparse_lookup.shape[0]:
                continue

            sparse_vals = []
            for col in item_str_cate_cols:
                raw = str(getattr(row, col))
                sparse_vals.append(item_str_maps[col].get(raw, 0))
            for col in item_int_cate_cols:
                raw = int(getattr(row, col))
                sparse_vals.append(item_int_maps[col].get(raw, 0))

            dense_vals = [float(getattr(row, col)) for col in item_dense_feature_names]
            item_sparse_lookup[vid] = np.asarray(sparse_vals, dtype=np.int64)
            item_dense_lookup[vid] = np.asarray(dense_vals, dtype=np.float32)

        print("[Step 4/4] Building sequence samples and split ...")
        user_ids = []
        target_video_ids = []
        hist_video_ids = []
        hist_masks = []
        labels = []
        train_indices = []
        test_indices = []

        grouped = df.groupby("user_id", sort=False)
        total_users = int(df["user_id"].nunique())

        sample_idx = 0
        for uid, g in tqdm(grouped, total=total_users, desc="Build samples"):
            items = g["video_id"].to_numpy(dtype=np.int64)
            ratios = g["watch_ratio"].to_numpy(dtype=np.float32)

            user_sample_indices = []
            for t in range(len(items)):
                if t < self.min_hist_len:
                    continue

                hist = items[max(0, t - self.max_len):t]
                hlen = len(hist)
                if hlen == 0:
                    continue

                padded_hist = np.zeros(self.max_len, dtype=np.int64)
                hist_mask = np.zeros(self.max_len, dtype=np.float32)
                padded_hist[-hlen:] = hist
                hist_mask[-hlen:] = 1.0

                label = 1.0 if float(ratios[t]) > float(self.like_threshold) else 0.0

                user_ids.append(int(uid))
                target_video_ids.append(int(items[t]))
                hist_video_ids.append(padded_hist)
                hist_masks.append(hist_mask)
                labels.append(label)

                user_sample_indices.append(sample_idx)
                sample_idx += 1

            n_u = len(user_sample_indices)
            if n_u == 1:
                train_indices.extend(user_sample_indices)
                test_indices.extend(user_sample_indices)
            elif n_u > 1:
                n_test = max(1, int(round(n_u * self.test_ratio)))
                n_test = min(n_test, n_u - 1)
                split = n_u - n_test
                train_indices.extend(user_sample_indices[:split])
                test_indices.extend(user_sample_indices[split:])

        user_ids = np.asarray(user_ids, dtype=np.int64)
        target_video_ids = np.asarray(target_video_ids, dtype=np.int64)
        hist_video_ids = np.asarray(hist_video_ids, dtype=np.int64)
        hist_masks = np.asarray(hist_masks, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        train_indices = np.asarray(train_indices, dtype=np.int64)
        test_indices = np.asarray(test_indices, dtype=np.int64)

        np.save(self.paths["user_ids"], user_ids)
        np.save(self.paths["target_video_ids"], target_video_ids)
        np.save(self.paths["hist_video_ids"], hist_video_ids)
        np.save(self.paths["hist_masks"], hist_masks)
        np.save(self.paths["labels"], labels)
        np.save(self.paths["train_indices"], train_indices)
        np.save(self.paths["test_indices"], test_indices)

        np.save(self.paths["user_sparse_lookup"], user_sparse_lookup)
        np.save(self.paths["user_dense_lookup"], user_dense_lookup)
        np.save(self.paths["item_sparse_lookup"], item_sparse_lookup)
        np.save(self.paths["item_dense_lookup"], item_dense_lookup)

        np.savez(
            self.paths["meta"],
            user_dense_feature_names=np.array(user_dense_feature_names, dtype=object),
            user_sparse_feature_names=np.array(user_sparse_feature_names, dtype=object),
            user_sparse_vocab_sizes=np.array(user_sparse_vocab_sizes, dtype=np.int64),
            item_dense_feature_names=np.array(item_dense_feature_names, dtype=object),
            item_sparse_feature_names=np.array(item_sparse_feature_names, dtype=object),
            item_sparse_vocab_sizes=np.array(item_sparse_vocab_sizes, dtype=np.int64),
        )

        pos_ratio = float(labels.mean()) if len(labels) > 0 else 0.0
        print(f"[KuaiRecCachedDataset] samples={len(labels)}, pos_ratio={pos_ratio:.4f}")
        print(f"[KuaiRecCachedDataset] train={len(train_indices)}, test={len(test_indices)}")

    def _load_cache(self):
        self.user_ids = np.load(self.paths["user_ids"], mmap_mode="r")
        self.target_video_ids = np.load(self.paths["target_video_ids"], mmap_mode="r")
        self.hist_video_ids = np.load(self.paths["hist_video_ids"], mmap_mode="r")
        self.hist_masks = np.load(self.paths["hist_masks"], mmap_mode="r")
        self.labels = np.load(self.paths["labels"], mmap_mode="r")
        self.train_indices = np.load(self.paths["train_indices"])
        self.test_indices = np.load(self.paths["test_indices"])

        self.user_sparse_lookup = np.load(self.paths["user_sparse_lookup"], mmap_mode="r")
        self.user_dense_lookup = np.load(self.paths["user_dense_lookup"], mmap_mode="r")
        self.item_sparse_lookup = np.load(self.paths["item_sparse_lookup"], mmap_mode="r")
        self.item_dense_lookup = np.load(self.paths["item_dense_lookup"], mmap_mode="r")

        meta = np.load(self.paths["meta"], allow_pickle=True)
        self.feature_spec = FeatureSpec(
            user_dense_feature_names=meta["user_dense_feature_names"].tolist(),
            user_sparse_feature_names=meta["user_sparse_feature_names"].tolist(),
            user_sparse_vocab_sizes=meta["user_sparse_vocab_sizes"].astype(np.int64).tolist(),
            item_dense_feature_names=meta["item_dense_feature_names"].tolist(),
            item_sparse_feature_names=meta["item_sparse_feature_names"].tolist(),
            item_sparse_vocab_sizes=meta["item_sparse_vocab_sizes"].astype(np.int64).tolist(),
        )

        if self.mode == "train":
            self.indices = self.train_indices
        elif self.mode == "test":
            self.indices = self.test_indices
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        print(f"[KuaiRecCachedDataset] mode={self.mode}, samples={len(self.indices)}")

    def _load_sid_lookup(self):
        self.sid_lookup_table = None
        self.num_sid_cols = 0

        if not self.use_sid:
            print("[KuaiRecCachedDataset] SID disabled by config.")
            return

        if self.sid_dir:
            sid_base = self.sid_dir
        else:
            sid_base = os.path.join(self.data_dir, "bge_m3_caption_embeddings")

        sid_keys_path = os.path.join(sid_base, self.sid_keys_filename)
        sid_vals_path = os.path.join(sid_base, self.sid_values_filename)

        if not (os.path.exists(sid_keys_path) and os.path.exists(sid_vals_path)):
            print(f"[KuaiRecCachedDataset] SID files not found in {sid_base}. Continue without SID.")
            return

        sid_keys = np.load(sid_keys_path)
        sid_vals = np.load(sid_vals_path)

        if sid_vals.ndim != 2:
            print("[KuaiRecCachedDataset] SID values should be 2D. Continue without SID.")
            return

        max_video_idx = int(self.item_sparse_lookup.shape[0])
        self.num_sid_cols = int(sid_vals.shape[1])
        self.sid_lookup_table = np.zeros((max_video_idx, self.num_sid_cols), dtype=np.int64)

        sid_keys = pd.to_numeric(pd.Series(sid_keys), errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        valid = (sid_keys >= 0) & (sid_keys < max_video_idx)
        self.sid_lookup_table[sid_keys[valid]] = sid_vals[valid].astype(np.int64)

        print(
            f"[KuaiRecCachedDataset] Loaded SID lookup: {self.sid_lookup_table.shape}, "
            f"valid_rows={(np.sum(np.any(self.sid_lookup_table != 0, axis=1)))}"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ridx = int(self.indices[idx])
        return {
            "user_id": torch.tensor(self.user_ids[ridx], dtype=torch.long),
            "target_video_id": torch.tensor(self.target_video_ids[ridx], dtype=torch.long),
            "hist_video_ids": torch.tensor(self.hist_video_ids[ridx], dtype=torch.long),
            "hist_mask": torch.tensor(self.hist_masks[ridx], dtype=torch.float32),
            "label": torch.tensor(self.labels[ridx], dtype=torch.float32),
        }


if __name__ == "__main__":
    train_dataset = KuaiRecCachedDataset(root="/data/cbn01/KuaiRec", mode="train", max_len=200)
    sample = train_dataset[0]
    print("sample keys:", sample.keys())
