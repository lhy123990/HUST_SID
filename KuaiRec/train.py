import argparse
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader

from DCNv2 import DCNV2
from dataloaderK import KuaiRecCachedDataset


SID_DIR = "/data/cbn01/KuaiRec/data/bge_m3_caption_embeddings"
SID_KEYS_FILENAME = "semantic_id_keys.npy"
SID_VALUES_FILENAME = "semantic_id_values.npy"


class Config:
    def __init__(self, args):
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.embedding_size = args.embedding_size
        self.cross_depth = args.depth
        self.mlp_hidden_units = args.mlp
        self.dropout = args.dropout
        self.epoch = args.epoch
        self.l2_reg = args.l2_reg
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def evaluate(model, loader, device):
    model.eval()
    preds = []
    labels = []
    user_ids = []

    with torch.no_grad():
        for batch in loader:
            user_ids.extend(batch["user_id"].numpy().tolist())
            labels.extend(batch["label"].numpy().tolist())

            batch = move_batch_to_device(batch, device)
            pred = model(batch)
            preds.extend(pred.detach().cpu().numpy().flatten().tolist())

    auc = roc_auc_score(labels, preds)
    ll = log_loss(labels, preds)

    user_data = defaultdict(lambda: {"labels": [], "preds": []})
    for uid, y, p in zip(user_ids, labels, preds):
        user_data[int(uid)]["labels"].append(float(y))
        user_data[int(uid)]["preds"].append(float(p))

    total_auc = 0.0
    valid_samples = 0
    for data in user_data.values():
        y_arr = np.array(data["labels"], dtype=np.float32)
        p_arr = np.array(data["preds"], dtype=np.float32)
        if len(np.unique(y_arr)) == 2:
            cnt = len(y_arr)
            total_auc += cnt * roc_auc_score(y_arr, p_arr)
            valid_samples += cnt

    gauc = (total_auc / valid_samples) if valid_samples > 0 else float("nan")
    return auc, gauc, ll


def train(args):
    config = Config(args)

    root_path = args.data_root
    print(f"Loading KuaiRec data from {root_path} ...")

    train_dataset = KuaiRecCachedDataset(
        root=root_path,
        mode="train",
        max_len=args.max_len,
        like_threshold=args.like_threshold,
        min_hist_len=args.min_hist_len,
        test_ratio=args.test_ratio,
        use_sid=args.use_sid,
        sid_dir=SID_DIR,
        sid_keys_filename=SID_KEYS_FILENAME,
        sid_values_filename=SID_VALUES_FILENAME,
        rebuild_cache=args.rebuild_cache,
    )
    test_dataset = KuaiRecCachedDataset(
        root=root_path,
        mode="test",
        max_len=args.max_len,
        like_threshold=args.like_threshold,
        min_hist_len=args.min_hist_len,
        test_ratio=args.test_ratio,
        use_sid=args.use_sid,
        sid_dir=SID_DIR,
        sid_keys_filename=SID_KEYS_FILENAME,
        sid_values_filename=SID_VALUES_FILENAME,
        rebuild_cache=False,
    )

    # Pass lookup tables and feature specs to model config.
    config.user_sparse_lookup = np.asarray(train_dataset.user_sparse_lookup)
    config.user_dense_lookup = np.asarray(train_dataset.user_dense_lookup)
    config.item_sparse_lookup = np.asarray(train_dataset.item_sparse_lookup)
    config.item_dense_lookup = np.asarray(train_dataset.item_dense_lookup)

    config.user_sparse_vocab_sizes = train_dataset.feature_spec.user_sparse_vocab_sizes
    config.item_sparse_vocab_sizes = train_dataset.feature_spec.item_sparse_vocab_sizes
    config.user_dense_dim = len(train_dataset.feature_spec.user_dense_feature_names)
    config.item_dense_dim = len(train_dataset.feature_spec.item_dense_feature_names)
    config.sid_lookup = train_dataset.sid_lookup_table

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"User sparse dim: {len(config.user_sparse_vocab_sizes)}, user dense dim: {config.user_dense_dim}")
    print(f"Item sparse dim: {len(config.item_sparse_vocab_sizes)}, item dense dim: {config.item_dense_dim}")
    print(f"SID cols: {0 if config.sid_lookup is None else config.sid_lookup.shape[1]}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = DCNV2(config).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2_reg)
    criterion = nn.BCELoss()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, args.save_name)

    best_auc = 0.0
    print("Start Training ...")

    for epoch in range(config.epoch):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, config.device)
            label = batch["label"].float().unsqueeze(1)

            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 100 == 0:
                print(f"Epoch {epoch + 1} | Step {step} | Loss: {loss.item():.4f}", end="\r")

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"\nEpoch {epoch + 1} Done. Avg Loss: {avg_loss:.4f}")

        auc, gauc, ll = evaluate(model, test_loader, config.device)
        if np.isnan(gauc):
            print(f"test AUC: {auc:.4f}, GAUC: NA (no users with both classes in eval), LogLoss: {ll:.4f}")
        else:
            print(f"test AUC: {auc:.4f}, GAUC: {gauc:.4f}, LogLoss: {ll:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/data/cbn01/KuaiRec")
    parser.add_argument("--save_dir", type=str, default="/data/cbn01/checkpoints")
    parser.add_argument("--save_name", type=str, default="best_kuairec_model.pth")

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=5120)
    parser.add_argument("--embedding_size", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--mlp", type=int, nargs="+", default=[512, 512, 512])
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--l2_reg", type=float, default=1e-5)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--min_hist_len", type=int, default=1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--like_threshold", type=float, default=3.0)
    parser.add_argument("--use_sid", action="store_true", default=True)
    parser.add_argument("--no_use_sid", action="store_false", dest="use_sid")
    parser.add_argument("--rebuild_cache", action="store_true", default=False)

    args = parser.parse_args()

    seed = 2026
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    train(args)
