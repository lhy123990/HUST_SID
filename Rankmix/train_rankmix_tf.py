#!/usr/bin/env python3
# coding: utf-8

import argparse
import random
import numpy as np
import tensorflow.compat.v1 as tf
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader

from dataloaderx import TaobaoDataset

tf.disable_v2_behavior()

from RankMix import RankMixer, RankMixerBackboneConfig, RankMixerTokenizerConfig


def build_graph(args, dense_feature_names, dense_vocab_sizes, seq_vocab_size):
    dense_ids = tf.placeholder(
        tf.int32,
        shape=[None, len(dense_feature_names)],
        name="dense_ids",
    )
    seq_ids = tf.placeholder(tf.int32, shape=[None, None], name="seq_ids")
    seq_mask = tf.placeholder(tf.float32, shape=[None, None], name="seq_mask")
    labels = tf.placeholder(tf.float32, shape=[None, 1], name="labels")

    dense_token_embs = []
    emb_tables = {}
    for i, feat_name in enumerate(dense_feature_names):
        vocab_size = int(max(2, dense_vocab_sizes[i]))
        table = tf.get_variable(
            name="emb_%s" % feat_name,
            shape=[vocab_size, args.embedding_dim],
            initializer=tf.glorot_uniform_initializer(),
        )
        emb_tables[feat_name] = table
        ids_i = tf.clip_by_value(dense_ids[:, i], 0, vocab_size - 1)
        dense_token_embs.append(tf.nn.embedding_lookup(table, ids_i))

    # Build din_output from sequence 205 ids and target 205 embedding.
    if "205" not in emb_tables:
        raise ValueError("Feature '205' is required for DIN alignment.")

    seq_vocab_size = int(max(2, seq_vocab_size))
    seq_safe_ids = tf.clip_by_value(seq_ids, 0, seq_vocab_size - 1)
    seq_emb = tf.nn.embedding_lookup(emb_tables["205"], seq_safe_ids)  # [B, L, E]

    target_idx_205 = dense_feature_names.index("205")
    target_emb = dense_token_embs[target_idx_205]  # [B, E]
    target_expand = tf.tile(tf.expand_dims(target_emb, axis=1), [1, tf.shape(seq_emb)[1], 1])

    din_concat = tf.concat([seq_emb, target_expand, seq_emb * target_expand], axis=-1)
    din_h = tf.keras.layers.Dense(args.embedding_dim, activation=tf.nn.relu, name="din_fc1")(din_concat)
    din_logit = tf.keras.layers.Dense(1, activation=None, name="din_fc2")(din_h)
    din_logit = tf.squeeze(din_logit, axis=-1)
    din_logit = tf.where(seq_mask > 0.5, din_logit, tf.ones_like(din_logit) * (-1e9))
    din_weight = tf.nn.softmax(din_logit, axis=1)
    din_output = tf.squeeze(tf.matmul(tf.expand_dims(din_weight, 1), seq_emb), axis=1)  # [B, E]

    pid_token = tf.zeros_like(din_output)

    rankmix_dense_embeddings = tf.stack(dense_token_embs + [pid_token], axis=1)
    rankmix_dense_feature_names = dense_feature_names + ["PID"]
    rankmix_seq_embeddings = tf.expand_dims(din_output, axis=1)
    rankmix_seq_feature_names = ["din_output"]

    model = RankMixer(
        tokenizer_config=RankMixerTokenizerConfig(
            target_tokens=args.target_tokens,
            d_model=args.d_model,
            embedding_dim=args.embedding_dim,
            version=args.tokenizer_version,
            include_seq_in_tokenization=args.include_seq_in_tokenization,
        ),
        backbone_config=RankMixerBackboneConfig(
            num_layers=args.num_layers,
            d_model=args.d_model,
            num_heads=args.num_heads,
            ffn_mult=args.ffn_mult,
            output_pooling=args.output_pooling,
            ffn_dropout=args.dropout,
            token_mixing_dropout=args.dropout,
            input_dropout=args.dropout,
        ),
    )

    output = model(
        dense_embeddings=rankmix_dense_embeddings,
        dense_feature_names=rankmix_dense_feature_names,
        seq_embeddings=rankmix_seq_embeddings,
        seq_feature_names=rankmix_seq_feature_names,
        training=True,
    )

    pooled = output.pooled_output
    hidden = tf.keras.layers.Dense(
        units=args.head_hidden,
        activation=tf.nn.relu,
        name="task_hidden",
    )(pooled)
    ctr_logit = tf.keras.layers.Dense(units=1, activation=None, name="ctr_logit")(hidden)
    ctr_prob = tf.nn.sigmoid(ctr_logit, name="ctr_prob")

    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=ctr_logit)
    )
    train_op = tf.train.AdamOptimizer(args.lr).minimize(loss)

    return {
        "dense_ids": dense_ids,
        "seq_ids": seq_ids,
        "seq_mask": seq_mask,
        "labels": labels,
        "ctr_prob": ctr_prob,
        "loss": loss,
        "train_op": train_op,
        "token_shape": tf.shape(output.tokens),
        "encoded_shape": tf.shape(output.encoded_tokens),
        "pooled_shape": tf.shape(output.pooled_output),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=5120)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--target_tokens", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ffn_mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--head_hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--epoch", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--use_sid", action="store_true", default=True)
    parser.add_argument("--no_use_sid", action="store_false", dest="use_sid")
    parser.add_argument("--use_pid", action="store_true", default=False)
    parser.add_argument("--output_pooling", type=str, default="mean", choices=["mean", "avg", "cls"])
    parser.add_argument("--tokenizer_version", type=str, default="v3")
    parser.add_argument("--include_seq_in_tokenization", action="store_true", default=True)
    args = parser.parse_args()

    seed = 2026
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    root_path = "/data/cbn01/mid_dataset"
    print(f"Loading data from {root_path}...")

    train_dataset = TaobaoDataset(root_path, mode='train', max_len=args.max_len, use_sid=args.use_sid, use_pid=args.use_pid)
    test_dataset = TaobaoDataset(root_path, mode='test', max_len=args.max_len, use_sid=args.use_sid, use_pid=args.use_pid)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    feature_names_all = list(train_dataset.feature_names)

    dense_feature_names = [
        "129_1", "130_1", "130_2", "130_3", "130_4", "130_5",
        "205", "206", "213", "214",
    ]

    dense_feature_indices = []
    dense_vocab_sizes = []
    for col in dense_feature_names:
        if col in feature_names_all:
            dense_feature_indices.append(feature_names_all.index(col))
            if col in train_dataset.maps:
                dense_vocab_sizes.append(len(train_dataset.maps[col]) + 1)
            else:
                dense_vocab_sizes.append(100000)
        else:
            dense_feature_indices.append(None)
            dense_vocab_sizes.append(2)

    seq_vocab_size = len(train_dataset.maps.get('150_2_180', [])) + 1 if '150_2_180' in train_dataset.maps else 200000

    if args.num_heads != args.target_tokens:
        raise ValueError(
            f"num_heads({args.num_heads}) must equal target_tokens({args.target_tokens}) for RankMix"
        )

    graph = build_graph(args, dense_feature_names, dense_vocab_sizes, seq_vocab_size)

    def prepare_batch(dnn_feat, seq_feat, seq_mask, labels):
        dnn_np = dnn_feat.numpy()
        seq_np = seq_feat.numpy().astype(np.int32)
        mask_np = seq_mask.numpy().astype(np.float32)
        label_np = labels.numpy().reshape(-1, 1).astype(np.float32)

        dense_id_cols = []
        for idx in dense_feature_indices:
            if idx is None:
                dense_id_cols.append(np.zeros((dnn_np.shape[0],), dtype=np.int32))
            else:
                dense_id_cols.append(dnn_np[:, idx].astype(np.int32))
        dense_ids_np = np.stack(dense_id_cols, axis=1)
        return dense_ids_np, seq_np, mask_np, label_np

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.visible_device_list = str(args.gpu)

    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())

        best_auc = 0.0
        for epoch in range(args.epoch):
            total_loss = 0.0
            steps = 0

            for dnn_feat, seq_feat, seq_mask_b, label in train_loader:
                dense_ids_np, seq_np, mask_np, label_np = prepare_batch(dnn_feat, seq_feat, seq_mask_b, label)
                _, loss_v = sess.run(
                    [graph["train_op"], graph["loss"]],
                    feed_dict={
                        graph["dense_ids"]: dense_ids_np,
                        graph["seq_ids"]: seq_np,
                        graph["seq_mask"]: mask_np,
                        graph["labels"]: label_np,
                    },
                )
                total_loss += float(loss_v)
                steps += 1
                if steps % 100 == 0:
                    print(f"Epoch {epoch+1} | Step {steps} | Loss: {loss_v:.4f}", end='\r')

            avg_loss = total_loss / max(1, steps)
            print(f"\nEpoch {epoch+1} Done. Avg Loss: {avg_loss:.4f}")

            preds = []
            gt = []
            for dnn_feat, seq_feat, seq_mask_b, label in test_loader:
                dense_ids_np, seq_np, mask_np, label_np = prepare_batch(dnn_feat, seq_feat, seq_mask_b, label)
                p = sess.run(
                    graph["ctr_prob"],
                    feed_dict={
                        graph["dense_ids"]: dense_ids_np,
                        graph["seq_ids"]: seq_np,
                        graph["seq_mask"]: mask_np,
                        graph["labels"]: label_np,
                    },
                )
                preds.extend(p.reshape(-1).tolist())
                gt.extend(label_np.reshape(-1).tolist())

            auc = roc_auc_score(gt, preds)
            ll = log_loss(gt, preds)
            print(f"test AUC: {auc:.4f}, LogLoss: {ll:.4f}")

            if auc > best_auc:
                best_auc = auc
                print(f"Best AUC updated: {best_auc:.4f}")


if __name__ == "__main__":
    main()
