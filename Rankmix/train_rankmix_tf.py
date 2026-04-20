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


def gelu(x):
    return 0.5 * x * (
        1.0 + tf.tanh(tf.sqrt(2.0 / 3.141592653589793) * (x + 0.044715 * tf.pow(x, 3)))
    )


def safe_layer_norm(x, name, epsilon=1e-6):
    """LayerNorm without fused batch norm kernels.

    Keras LayerNormalization in TF1 graph mode may map to FusedBatchNormV3,
    which can fail on some CUDA/cuDNN environments for large tensors.
    """
    dim = x.shape[-1].value if hasattr(x.shape[-1], "value") else int(x.shape[-1])
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        gamma = tf.get_variable("gamma", shape=[dim], initializer=tf.ones_initializer())
        beta = tf.get_variable("beta", shape=[dim], initializer=tf.zeros_initializer())
    mean, var = tf.nn.moments(x, axes=[-1], keepdims=True)
    normed = (x - mean) / tf.sqrt(var + epsilon)
    return normed * gamma + beta


def build_graph(
    args,
    feature_names,
    feature_vocab_sizes,
    attr_lookup_tables=None,
    sid_lookup_table=None,
    pid_lookup_table=None,
    pid_sim_table=None,
    pid_basis_lookup_table=None,
):
    dense_ids = tf.placeholder(tf.int32, shape=[None, len(feature_names)], name="dense_ids")
    seq_ids = tf.placeholder(tf.int32, shape=[None, None], name="seq_ids")
    seq_mask = tf.placeholder(tf.float32, shape=[None, None], name="seq_mask")
    labels = tf.placeholder(tf.float32, shape=[None, 1], name="labels")

    feature_to_idx = {name: i for i, name in enumerate(feature_names)}
    seq_attr_list = ["206", "213", "214"]

    dense_token_embs = []
    emb_tables = {}
    for i, feat_name in enumerate(feature_names):
        vocab_size = int(max(2, feature_vocab_sizes[i]))
        table = tf.get_variable(
            name="emb_%s" % feat_name,
            shape=[vocab_size, args.embedding_dim],
            initializer=tf.glorot_uniform_initializer(),
        )
        emb_tables[feat_name] = table
        ids_i = tf.clip_by_value(dense_ids[:, i], 0, vocab_size - 1)
        dense_token_embs.append(tf.nn.embedding_lookup(table, ids_i))

    if "205" not in feature_to_idx:
        raise ValueError("Feature '205' is required for DIN alignment.")
    idx_205 = feature_to_idx["205"]

    batch_size = tf.shape(dense_ids)[0]
    seq_len = tf.shape(seq_ids)[1]

    dnn_embs_stack = tf.stack(dense_token_embs, axis=1)

    # 2) Construct Target Item (DIN query)
    target_parts = [dense_token_embs[idx_205]]
    for col in seq_attr_list:
        if col in feature_to_idx:
            target_parts.append(dense_token_embs[feature_to_idx[col]])
        else:
            target_parts.append(tf.zeros([batch_size, args.embedding_dim], dtype=tf.float32))

    # 2.3 SID (target)
    num_sid_cols = 0
    sid_tables = []
    target_sid_parts = []
    sid_lookup_const = None
    if args.use_sid and sid_lookup_table is not None:
        sid_lookup_np = np.asarray(sid_lookup_table, dtype=np.int32)
        if sid_lookup_np.ndim == 2 and sid_lookup_np.shape[1] > 0:
            sid_lookup_const = tf.constant(sid_lookup_np, dtype=tf.int32)
            num_sid_cols = int(sid_lookup_np.shape[1])
            sid_vocab_size = int(max(2, int(np.max(sid_lookup_np)) + 10))
            for i in range(num_sid_cols):
                sid_tables.append(
                    tf.get_variable(
                        name="sid_emb_%d" % i,
                        shape=[sid_vocab_size, args.embedding_dim],
                        initializer=tf.glorot_uniform_initializer(),
                    )
                )

            target_ids = tf.clip_by_value(dense_ids[:, idx_205], 0, sid_lookup_np.shape[0] - 1)
            target_sid_ids = tf.gather(sid_lookup_const, target_ids)
            for i in range(num_sid_cols):
                sid_emb = tf.nn.embedding_lookup(sid_tables[i], target_sid_ids[:, i])
                target_sid_parts.append(sid_emb)

    # 2.4 PID (target)
    use_pid_graph = bool(
        args.use_pid
        and pid_lookup_table is not None
        and pid_sim_table is not None
        and pid_basis_lookup_table is not None
    )
    num_pid_blocks = 1 if use_pid_graph else 0
    pid_aux_loss = tf.constant(0.0, dtype=tf.float32)
    target_pid_emb = None

    if use_pid_graph:
        pid_lookup_np = np.asarray(pid_lookup_table, dtype=np.int32)
        pid_sim_np = np.asarray(pid_sim_table, dtype=np.float32)
        pid_basis_lookup_np = np.asarray(pid_basis_lookup_table, dtype=np.int32)
        pid_lookup_const = tf.constant(pid_lookup_np, dtype=tf.int32)
        pid_sim_const = tf.constant(pid_sim_np, dtype=tf.float32)
        pid_basis_lookup_const = tf.constant(pid_basis_lookup_np, dtype=tf.int32)

        pid_k = int(pid_lookup_np.shape[1])
        item_vocab_size = int(max(2, feature_vocab_sizes[idx_205]))
        pid_basis_vocab_size = int(max(2, int(np.max(pid_basis_lookup_np)) + 1))

        pid_item_embeddings_nonpad = tf.get_variable(
            name="pid_item_embeddings_nonpad",
            shape=[pid_basis_vocab_size - 1, args.embedding_dim],
            initializer=tf.glorot_uniform_initializer(),
        )
        pid_item_embeddings = tf.concat(
            [tf.zeros([1, args.embedding_dim], dtype=tf.float32), pid_item_embeddings_nonpad],
            axis=0,
        )

        pid_linear = tf.keras.layers.Dense(pid_k, activation=None, name="pid_linear")
        pid_proj_fc1 = tf.keras.layers.Dense(args.embedding_dim, activation=None, name="pid_proj_fc1")
        pid_proj_fc2 = tf.keras.layers.Dense(args.embedding_dim, activation=None, name="pid_proj_fc2")

        def pid_embedding_from_ids(item_ids):
            safe_ids = tf.clip_by_value(item_ids, 0, pid_lookup_np.shape[0] - 1)
            flat_safe_ids = tf.reshape(safe_ids, [-1])

            # Shared compute on unique IDs to avoid repeated PID aux loss from duplicates.
            unique_ids, inverse = tf.unique(flat_safe_ids)

            neighbor_ids = tf.gather(pid_lookup_const, unique_ids)
            neighbor_sims = tf.gather(pid_sim_const, unique_ids)

            neighbor_ids_safe = tf.clip_by_value(neighbor_ids, 0, item_vocab_size - 1)
            pid_neighbor_ids = tf.gather(pid_basis_lookup_const, neighbor_ids_safe)
            pid_neighbor_ids = tf.clip_by_value(pid_neighbor_ids, 0, pid_basis_vocab_size - 1)
            neighbor_embs = tf.nn.embedding_lookup(pid_item_embeddings, pid_neighbor_ids)

            sims_2d = tf.reshape(neighbor_sims, [-1, pid_k])
            weights_2d = tf.nn.sigmoid(pid_linear(sims_2d))
            weights = tf.reshape(weights_2d, tf.shape(neighbor_sims))

            diff = weights[..., 1:] - weights[..., :-1]
            mono_loss = tf.reduce_sum(tf.nn.relu(diff))

            mask = tf.cast(tf.not_equal(neighbor_ids, 0), tf.float32)
            weighted_embs = neighbor_embs * tf.expand_dims(weights * mask, axis=-1)
            pid_emb_unique = tf.reduce_sum(weighted_embs, axis=-2)

            pid_emb_2d = tf.reshape(pid_emb_unique, [-1, args.embedding_dim])
            pid_emb_2d = pid_proj_fc1(pid_emb_2d)
            pid_emb_2d = safe_layer_norm(pid_emb_2d, name="pid_proj_ln")
            pid_emb_2d = gelu(pid_emb_2d)
            pid_emb_2d = pid_proj_fc2(pid_emb_2d)
            pid_emb_unique = tf.reshape(pid_emb_2d, tf.shape(pid_emb_unique))

            distill_loss = tf.constant(0.0, dtype=tf.float32)
            if args.pid_distill:
                self_match = tf.equal(neighbor_ids[..., 0], unique_ids)
                self_sim = tf.abs(neighbor_sims[..., 0] - 1.0) < 1e-6
                basis_mask = tf.logical_and(tf.greater(unique_ids, 0), tf.logical_and(self_match, self_sim))
                if pid_k > 1:
                    zero_id_tail = tf.reduce_all(tf.equal(neighbor_ids[..., 1:], 0), axis=-1)
                    zero_sim_tail = tf.reduce_all(tf.abs(neighbor_sims[..., 1:]) < 1e-6, axis=-1)
                    basis_mask = tf.logical_and(basis_mask, tf.logical_and(zero_id_tail, zero_sim_tail))

                basis_pid_raw_emb = neighbor_embs[..., 0, :]
                self_205_emb = tf.nn.embedding_lookup(emb_tables["205"], tf.clip_by_value(unique_ids, 0, item_vocab_size - 1))
                basis_pid_norm = tf.nn.l2_normalize(basis_pid_raw_emb, axis=-1)
                self_205_norm = tf.nn.l2_normalize(self_205_emb, axis=-1)
                cos_sim = tf.reduce_sum(basis_pid_norm * self_205_norm, axis=-1)
                per_item = (1.0 - cos_sim) * tf.cast(basis_mask, tf.float32)
                denom_distill = tf.reduce_sum(tf.cast(basis_mask, tf.float32))
                distill_loss = tf.math.divide_no_nan(tf.reduce_sum(per_item), denom_distill)
                distill_loss = distill_loss * float(args.pid_distill_weight)

            pid_emb_flat = tf.gather(pid_emb_unique, inverse)
            out_shape = tf.concat([tf.shape(item_ids), tf.constant([args.embedding_dim], dtype=tf.int32)], axis=0)
            pid_emb = tf.reshape(pid_emb_flat, out_shape)

            denom = tf.cast(tf.maximum(1, tf.size(unique_ids)), tf.float32)
            mono_loss = mono_loss / denom
            return pid_emb, mono_loss + distill_loss

        target_pid_emb, target_mono = pid_embedding_from_ids(dense_ids[:, idx_205])
        target_parts.append(target_pid_emb)
        pid_aux_loss = pid_aux_loss + target_mono

    # Keep SID blocks at the end of all target features.
    if num_sid_cols > 0:
        target_parts.extend(target_sid_parts)

    target_item_emb = tf.concat(target_parts, axis=-1)

    # 3) Construct Sequence Items (DIN key)
    seq_vocab_205 = int(max(2, feature_vocab_sizes[idx_205]))
    seq_safe_ids_205 = tf.clip_by_value(seq_ids, 0, seq_vocab_205 - 1)
    seq_parts = [tf.nn.embedding_lookup(emb_tables["205"], seq_safe_ids_205)]

    flat_seq_ids = tf.reshape(seq_ids, [-1])

    for col in seq_attr_list:
        if (
            attr_lookup_tables is not None
            and col in attr_lookup_tables
            and col in feature_to_idx
        ):
            lookup_np = np.asarray(attr_lookup_tables[col], dtype=np.int32)
            lookup_const = tf.constant(lookup_np, dtype=tf.int32)
            safe_flat = tf.clip_by_value(flat_seq_ids, 0, lookup_np.shape[0] - 1)
            attr_ids_flat = tf.gather(lookup_const, safe_flat)

            vocab_size = int(max(2, feature_vocab_sizes[feature_to_idx[col]]))
            attr_ids_flat = tf.clip_by_value(attr_ids_flat, 0, vocab_size - 1)
            attr_emb_flat = tf.nn.embedding_lookup(emb_tables[col], attr_ids_flat)
            attr_emb = tf.reshape(attr_emb_flat, [batch_size, seq_len, args.embedding_dim])
            seq_parts.append(attr_emb)
        else:
            seq_parts.append(tf.zeros([batch_size, seq_len, args.embedding_dim], dtype=tf.float32))

    if use_pid_graph:
        seq_pid_emb, seq_mono = pid_embedding_from_ids(seq_ids)
        seq_parts.append(seq_pid_emb)
        pid_aux_loss = pid_aux_loss + seq_mono

    # Keep SID blocks at the end of all sequence features.
    if num_sid_cols > 0 and sid_lookup_const is not None:
        safe_seq_ids_sid = tf.clip_by_value(seq_ids, 0, int(np.asarray(sid_lookup_table).shape[0]) - 1)
        seq_sid_ids = tf.gather(sid_lookup_const, safe_seq_ids_sid)
        for i in range(num_sid_cols):
            sid_emb = tf.nn.embedding_lookup(sid_tables[i], seq_sid_ids[:, :, i])
            seq_parts.append(sid_emb)

    seq_item_emb = tf.concat(seq_parts, axis=-1)

    # DIN attention: concat([hist, target, hist*target]) -> FC -> mask -> softmax -> weighted sum
    target_expand = tf.tile(tf.expand_dims(target_item_emb, axis=1), [1, seq_len, 1])
    din_concat = tf.concat([seq_item_emb, target_expand, seq_item_emb * target_expand], axis=-1)
    item_total_dim = int((1 + len(seq_attr_list) + num_sid_cols + num_pid_blocks) * args.embedding_dim)
    din_h = tf.keras.layers.Dense(item_total_dim, activation=tf.nn.relu, name="din_fc1")(din_concat)
    din_logit = tf.keras.layers.Dense(1, activation=None, name="din_fc2")(din_h)
    din_logit = tf.squeeze(din_logit, axis=-1)
    din_logit = tf.where(seq_mask > 0.5, din_logit, tf.ones_like(din_logit) * (-1e9))
    din_weight = tf.nn.softmax(din_logit, axis=1)
    din_output = tf.squeeze(tf.matmul(tf.expand_dims(din_weight, 1), seq_item_emb), axis=1)

    # Align with DCNv2 blocks for RankMix input
    block_count = 1 + len(seq_attr_list) + num_sid_cols + num_pid_blocks
    rankmix_seq_embeddings = tf.reshape(din_output, [batch_size, block_count, args.embedding_dim])

    rankmix_seq_feature_names = ["din_205", "din_206", "din_213", "din_214"]
    if use_pid_graph:
        rankmix_seq_feature_names.append("din_pid")
    if num_sid_cols > 0:
        rankmix_seq_feature_names.extend(["din_sid_%d" % i for i in range(num_sid_cols)])

    # Move target-side extras to seq branch tail (as extra tokens)
    if target_pid_emb is not None:
        rankmix_seq_embeddings = tf.concat(
            [rankmix_seq_embeddings, tf.expand_dims(target_pid_emb, axis=1)],
            axis=1,
        )
        rankmix_seq_feature_names.append("target_pid")

    if num_sid_cols > 0 and target_sid_parts:
        target_sid_stack = tf.stack(target_sid_parts, axis=1)  # [B, num_sid_cols, E]
        rankmix_seq_embeddings = tf.concat(
            [rankmix_seq_embeddings, target_sid_stack],
            axis=1,
        )
        rankmix_seq_feature_names.extend(["target_sid_%d" % i for i in range(num_sid_cols)])

    # Keep dense branch pure dense
    rankmix_dense_tokens = list(dense_token_embs)
    rankmix_dense_feature_names = list(feature_names)
    rankmix_dense_embeddings = tf.stack(rankmix_dense_tokens, axis=1)
    print("rankmix_dense_embeddings shape:", rankmix_dense_embeddings.shape)
    print("rankmix_seq_embeddings shape:", rankmix_seq_embeddings.shape)

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
    hidden = tf.keras.layers.Dense(units=args.head_hidden, activation=tf.nn.relu, name="task_hidden")(pooled)
    ctr_logit = tf.keras.layers.Dense(units=1, activation=None, name="ctr_logit")(hidden)
    ctr_prob = tf.nn.sigmoid(ctr_logit, name="ctr_prob")

    base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=ctr_logit))
    loss = base_loss + args.pid_aux_coef * pid_aux_loss
    train_op = tf.train.AdamOptimizer(args.lr).minimize(loss)

    return {
        "dense_ids": dense_ids,
        "seq_ids": seq_ids,
        "seq_mask": seq_mask,
        "labels": labels,
        "ctr_prob": ctr_prob,
        "loss": loss,
        "base_loss": base_loss,
        "pid_aux_loss": pid_aux_loss,
        "train_op": train_op,
        "token_shape": tf.shape(output.tokens),
        "encoded_shape": tf.shape(output.encoded_tokens),
        "pooled_shape": tf.shape(output.pooled_output),
    }


def compute_gauc(labels, preds, user_ids):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    user_ids = np.asarray(user_ids)

    total_weight = 0.0
    weighted_auc = 0.0

    for user_id in np.unique(user_ids):
        mask = user_ids == user_id
        user_labels = labels[mask]
        user_preds = preds[mask]

        if user_labels.size < 2:
            continue
        if np.all(user_labels == 0) or np.all(user_labels == 1):
            continue

        auc = roc_auc_score(user_labels, user_preds)
        weight = float(user_labels.size)
        weighted_auc += auc * weight
        total_weight += weight

    return weighted_auc / total_weight if total_weight > 0 else 0.0

#python  train_rankmix_tf.py --use_pid --no_use_sid --pid_distill --pid_distill_weight 0.1 

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
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epoch", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--use_sid", action="store_true", default=True)
    parser.add_argument("--no_use_sid", action="store_false", dest="use_sid")
    parser.add_argument("--use_pid", action="store_true", default=False)
    parser.add_argument("--pid_aux_coef", type=float, default=1.0)
    parser.add_argument("--pid_distill", action="store_true", default=False)
    parser.add_argument("--pid_distill_weight", type=float, default=0.1)
    parser.add_argument("--output_pooling", type=str, default="mean", choices=["mean", "avg", "cls"])
    parser.add_argument("--tokenizer_version", type=str, default="v3")
    parser.add_argument("--include_seq_in_tokenization", action="store_true", default=True)
    parser.add_argument("--gauc_user_col", type=str, default="129_1")
    args = parser.parse_args()

    seed = 2026
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    root_path = "/data/cbn01/mid_dataset"
    print(f"Loading data from {root_path}...")

    train_dataset = TaobaoDataset(root_path, mode="train", max_len=args.max_len, use_sid=args.use_sid, use_pid=args.use_pid)
    test_dataset = TaobaoDataset(root_path, mode="test", max_len=args.max_len, use_sid=args.use_sid, use_pid=args.use_pid)

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

    feature_names = list(train_dataset.feature_names)
    if args.gauc_user_col not in feature_names:
        raise ValueError(f"GAUC user column '{args.gauc_user_col}' not found in dataset features: {feature_names}")
    gauc_user_idx = feature_names.index(args.gauc_user_col)

    feature_vocab_sizes = []
    for col in feature_names:
        if col in train_dataset.maps:
            feature_vocab_sizes.append(len(train_dataset.maps[col]) + 1)
        else:
            feature_vocab_sizes.append(100000)

    attr_lookup_tables = {}
    for key, tensor in getattr(train_dataset, "attr_lookups", {}).items():
        if hasattr(tensor, "numpy"):
            attr_lookup_tables[str(key)] = tensor.numpy().astype(np.int32)
        else:
            attr_lookup_tables[str(key)] = np.asarray(tensor, dtype=np.int32)

    sid_lookup_table = None
    if args.use_sid and getattr(train_dataset, "sid_lookup_table", None) is not None:
        sid_lookup_table = np.asarray(train_dataset.sid_lookup_table, dtype=np.int32)

    pid_lookup_table = None
    pid_sim_table = None
    pid_basis_lookup_table = None
    if args.use_pid:
        if (
            getattr(train_dataset, "pid_lookup_table", None) is not None
            and getattr(train_dataset, "pid_sim_table", None) is not None
            and getattr(train_dataset, "pid_basis_lookup_table", None) is not None
        ):
            pid_lookup_table = np.asarray(train_dataset.pid_lookup_table, dtype=np.int32)
            pid_sim_table = np.asarray(train_dataset.pid_sim_table, dtype=np.float32)
            pid_basis_lookup_table = np.asarray(train_dataset.pid_basis_lookup_table, dtype=np.int32)
        else:
            print("[WARN] use_pid=True but PID lookup/sim/basis tables are unavailable, fallback to no-PID graph.")

    if args.num_heads != args.target_tokens:
        raise ValueError(f"num_heads({args.num_heads}) must equal target_tokens({args.target_tokens}) for RankMix")

    graph = build_graph(
        args,
        feature_names=feature_names,
        feature_vocab_sizes=feature_vocab_sizes,
        attr_lookup_tables=attr_lookup_tables,
        sid_lookup_table=sid_lookup_table,
        pid_lookup_table=pid_lookup_table,
        pid_sim_table=pid_sim_table,
        pid_basis_lookup_table=pid_basis_lookup_table,
    )

    def prepare_batch(dnn_feat, seq_feat, seq_mask, labels):
        dense_ids_np = dnn_feat.numpy().astype(np.int32)
        seq_np = seq_feat.numpy().astype(np.int32)
        mask_np = seq_mask.numpy().astype(np.float32)
        label_np = labels.numpy().reshape(-1, 1).astype(np.float32)
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
                    print(f"Epoch {epoch+1} | Step {steps} | Loss: {loss_v:.4f}", end="\r")

            avg_loss = total_loss / max(1, steps)
            print(f"\nEpoch {epoch+1} Done. Avg Loss: {avg_loss:.4f}")

            preds = []
            gt = []
            user_ids = []
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
                user_ids.extend(dense_ids_np[:, gauc_user_idx].reshape(-1).tolist())

            auc = roc_auc_score(gt, preds)
            ll = log_loss(gt, preds)
            gauc = compute_gauc(gt, preds, user_ids)
            print(f"test AUC: {auc:.4f}, GAUC: {gauc:.4f}, LogLoss: {ll:.4f}")

            if auc > best_auc:
                best_auc = auc
                print(f"Best AUC updated: {best_auc:.4f}")


if __name__ == "__main__":
    main()
