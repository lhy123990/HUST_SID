#!/usr/bin/env python3
# coding: utf-8
"""RankMixer 核心模块。

这个文件保留的是可直接复用的网络部分，不包含训练脚本、数据读取、
动态 embedding lookup 和业务侧特征处理。

如果外部项目已经准备好了 feature embedding，就可以直接使用这里的：

1. `RankMixerTokenizer`
2. `RankMixerBackbone`
3. `RankMixer`

模块依赖 `tensorflow.compat.v1`。
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Optional, Sequence, Tuple

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
logger = tf.compat.v1.logging


def _dense(inputs, units, activation=None, name=None):
    layer = tf.keras.layers.Dense(units=units, activation=activation, name=name)
    return layer(inputs)


DEFAULT_SEMANTIC_GROUP_RULES_V1 = [
    {"name": "core_id", "patterns": [r"^combination_un_id$"]},
    {"name": "seq", "patterns": [r"^seq::", r"^seq_"]},
    {"name": "dpa", "patterns": [r"^dpa_"]},
    {"name": "item_meta", "patterns": [
        r"^brand_name$", r"^first_category$", r"^second_category$", r"^annual_vol$",
        r"^shop_id$", r"^shop_name$", r"^shop_source$",
    ]},
    {"name": "price", "patterns": [
        r"^reserve_price$", r"^final_promotion_price$", r"^commission$", r"^commission_rate$",
    ]},
    {"name": "semantics", "patterns": [r"^title_sem_id$", r"^image_sem_id$"]},
    {"name": "adslot", "patterns": [
        r"^adx_adslot_id$", r"^ssp_adslot_id$", r"^adslot_id$", r"^channel_id$",
        r"^adslot_id_type$", r"^source_adslot_type$", r"^bid_floor$",
        r"^ad_idea_id$", r"^ad_unit_id$", r"^template_id$", r"^template_type$",
        r"^promotion_type$", r"^target_type$",
    ]},
    {"name": "app", "patterns": [
        r"^app_pkg_src$", r"^app_pkg$", r"^app_src_", r"^package_name$",
        r"^app_first_type$", r"^app_second_type$",
    ]},
    {"name": "device", "patterns": [
        r"^device_", r"^network$", r"^ip_region$", r"^ip_city$", r"^device_size$", r"^city_level$",
    ]},
    {"name": "strategy", "patterns": [
        r"^model_type$", r"^dispatch_center_id$", r"^rta_type$", r"^crowd_type$", r"^is_new_item$",
    ]},
    {"name": "time", "patterns": [r"^day_h$"]},
    {"name": "user_stat", "patterns": [r"^user__"]},
    {"name": "item_stat", "patterns": [r"^item__"]},
    {"name": "skuid_key_one", "patterns": [r"^skuid__key_one__"]},
    {"name": "skuid_key_two", "patterns": [r"^skuid__key_two__"]},
    {"name": "skuid_key_three", "patterns": [r"^skuid__key_three__"]},
    {"name": "skuid_key_four", "patterns": [r"^skuid__key_four__"]},
    {"name": "skuid_key_five", "patterns": [r"^skuid__key_five__"]},
    {"name": "skuid_stat", "patterns": [r"^skuid__"]},
    {"name": "tsd_stat", "patterns": [r"^tsd__"]},
    {"name": "isd_stat", "patterns": [r"^isd__"]},
]

# v2/v3 在原仓库里默认规则为空，这里保持兼容。
DEFAULT_SEMANTIC_GROUP_RULES_V2 = []
DEFAULT_SEMANTIC_GROUP_RULES_V3 = []


def gelu(x):
    """GELU activation.

    Args:
        x:
            `tf.Tensor`，任意 shape。

    Returns:
        与输入同 shape 的 `tf.Tensor`。
    """
    return 0.5 * x * (
        1.0 + tf.tanh(tf.sqrt(2.0 / 3.141592653589793) * (x + 0.044715 * tf.pow(x, 3)))
    )


def _shape_dim(tensor: Optional[tf.Tensor], axis: int) -> Optional[int]:
    """Return static dim when available, otherwise `None`."""
    if tensor is None:
        return None
    dim = tensor.shape[axis]
    if hasattr(dim, "value"):
        return dim.value
    try:
        return int(dim)
    except (TypeError, ValueError):
        return None


def _normalize_feature_names(
    tensor: Optional[tf.Tensor],
    feature_names: Optional[Sequence[str]],
    prefix: str,
) -> Tuple[str, ...]:
    """Normalize feature names and validate length."""
    if tensor is None:
        return tuple()

    size = _shape_dim(tensor, 1)
    if feature_names is None:
        if size is None:
            raise ValueError(
                "%s_feature_names is required when tensor.shape[1] is dynamic." % prefix
            )
        return tuple("%s_%d" % (prefix, i) for i in range(size))

    names = tuple(str(name) for name in feature_names)
    if size is not None and len(names) != size:
        raise ValueError(
            "%s_feature_names length=%d does not match tensor.shape[1]=%d."
            % (prefix, len(names), size)
        )
    return names


def _validate_last_dim(tensor: Optional[tf.Tensor], expected_dim: int, name: str) -> None:
    """Validate tensor last dim when statically known."""
    if tensor is None:
        return
    actual_dim = _shape_dim(tensor, -1)
    if actual_dim is not None and actual_dim != expected_dim:
        raise ValueError("%s last dim must be %d, got %d." % (name, expected_dim, actual_dim))


def _resolve_token_count(tokens: tf.Tensor, token_count: Optional[int]) -> int:
    """Resolve token count from explicit argument or static shape."""
    if token_count is not None:
        return int(token_count)
    static_count = _shape_dim(tokens, 1)
    if static_count is None:
        raise ValueError("token_count is required when tokens.shape[1] is dynamic.")
    return int(static_count)


def _sanitize_group_name(name: str) -> str:
    """Sanitize group name for TensorFlow scope usage."""
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
    return safe or "group"


def _looks_like_regex(pattern: str) -> bool:
    """Heuristically detect whether a string should be treated as regex."""
    if pattern.startswith("re:"):
        return True
    for token in ("^", "$", ".*", "[", "]", "(", ")", "|", "?"):
        if token in pattern:
            return True
    return False


def _normalize_groups(semantic_groups: Optional[Any]) -> list:
    """Normalize semantic group config to `[(name, feature_list), ...]`."""
    if not semantic_groups:
        return []
    if isinstance(semantic_groups, dict):
        return [(str(key), list(value)) for key, value in semantic_groups.items()]
    if isinstance(semantic_groups, (list, tuple)):
        groups = []
        for index, item in enumerate(semantic_groups):
            if isinstance(item, dict):
                name = item.get("name", "group_%d" % index)
                feats = item.get("features") or item.get("patterns") or []
                groups.append((str(name), list(feats)))
            elif isinstance(item, (list, tuple)):
                groups.append(("group_%d" % index, list(item)))
        return groups
    return []


def _compile_group_rules(
    group_rules: Optional[Any],
    default_group_rules: Optional[Sequence[Dict[str, Any]]] = None,
) -> list:
    """Compile group rules into regex patterns."""
    rules = group_rules if group_rules is not None else (default_group_rules or [])
    compiled = []
    for rule in rules:
        name = _sanitize_group_name(rule.get("name", "group"))
        patterns = [pattern for pattern in rule.get("patterns", []) if pattern]
        if not patterns:
            continue
        compiled.append((name, [re.compile(pattern) for pattern in patterns]))
    return compiled


def _assign_semantic_groups(
    feature_names: Sequence[str],
    group_rules: Optional[Any],
    default_group_rules: Optional[Sequence[Dict[str, Any]]] = None,
) -> list:
    """Return reordered feature indices according to semantic group rules."""
    compiled = _compile_group_rules(group_rules, default_group_rules)
    grouped = []
    used = set()
    for _, patterns in compiled:
        indices = []
        for index, feature_name in enumerate(feature_names):
            if index in used:
                continue
            for pattern in patterns:
                if pattern.search(feature_name):
                    indices.append(index)
                    used.add(index)
                    break
        if indices:
            grouped.extend(indices)
    for index in range(len(feature_names)):
        if index not in used:
            grouped.append(index)
    return grouped


class ParameterFreeTokenMixer(tf.keras.layers.Layer):
    """Parameter-free token mixing.

    这个层对应 RankMixer 里的 token mixing 部分，不引入额外可学习投影，
    只对 token 维和 channel 维做重排。

    Args:
        num_tokens:
            token 数 `T`。
        d_model:
            hidden dim `D`。
        num_heads:
            head 数 `H`。当前实现要求 `H == T`。
        dropout:
            输出 dropout。
        name:
            TensorFlow layer 名称。

    Call args:
        x:
            `tf.Tensor`，shape `[B, T, D]`。
        training:
            是否训练态。

    Returns:
        `tf.Tensor`，shape `[B, T, D]`。

    Raises:
        ValueError:
            当 `num_heads != num_tokens` 或 `d_model % num_heads != 0` 时抛出。
    """

    def __init__(self, num_tokens, d_model, num_heads=None, dropout=0.0, name=None):
        super(ParameterFreeTokenMixer, self).__init__(name=name)
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads) if num_heads is not None else int(num_tokens)
        self.dropout = float(dropout)

    def build(self, input_shape):
        if self.num_heads != self.num_tokens:
            raise ValueError("ParameterFreeTokenMixer requires num_heads == num_tokens.")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads, got d_model=%d num_heads=%d"
                % (self.d_model, self.num_heads)
            )
        self.d_head = self.d_model // self.num_heads
        super(ParameterFreeTokenMixer, self).build(input_shape)

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        split = tf.reshape(x, [batch_size, self.num_tokens, self.num_heads, self.d_head])
        shuffled = tf.transpose(split, [0, 2, 1, 3])
        merged = tf.reshape(shuffled, [batch_size, self.num_heads, self.num_tokens * self.d_head])
        mixed = tf.reshape(merged, [batch_size, self.num_tokens, self.d_model])
        if self.dropout and training:
            mixed = tf.nn.dropout(mixed, keep_prob=1.0 - self.dropout)
        return mixed


class PerTokenFFN(tf.keras.layers.Layer):
    """Per-token FFN.

    每个 token 位置都有自己独立的一组 FFN 参数，不共享权重。

    Args:
        num_tokens:
            token 数 `T`。
        d_model:
            hidden dim `D`。
        mult:
            隐层扩张倍数，隐层维度为 `D * mult`。
        dropout:
            FFN 内部 dropout。
        name:
            TensorFlow layer 名称。

    Call args:
        x:
            `tf.Tensor`，shape `[B, T, D]`。
        training:
            是否训练态。

    Returns:
        `tf.Tensor`，shape `[B, T, D]`。
    """

    def __init__(self, num_tokens, d_model, mult=4, dropout=0.0, name=None):
        super(PerTokenFFN, self).__init__(name=name)
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.mult = int(mult)
        self.dropout = float(dropout)

    def build(self, input_shape):
        hidden_dim = self.d_model * self.mult
        init = tf.variance_scaling_initializer(scale=2.0)
        self.W1 = self.add_weight(name="W1", shape=[self.num_tokens, self.d_model, hidden_dim], initializer=init)
        self.b1 = self.add_weight(name="b1", shape=[self.num_tokens, hidden_dim], initializer=tf.zeros_initializer())
        self.W2 = self.add_weight(name="W2", shape=[self.num_tokens, hidden_dim, self.d_model], initializer=init)
        self.b2 = self.add_weight(name="b2", shape=[self.num_tokens, self.d_model], initializer=tf.zeros_initializer())
        super(PerTokenFFN, self).build(input_shape)

    def call(self, x, training=False):
        hidden = tf.einsum("btd,tdh->bth", x, self.W1) + self.b1
        hidden = gelu(hidden)
        if self.dropout and training:
            hidden = tf.nn.dropout(hidden, keep_prob=1.0 - self.dropout)
        output = tf.einsum("bth,thd->btd", hidden, self.W2) + self.b2
        if self.dropout and training:
            output = tf.nn.dropout(output, keep_prob=1.0 - self.dropout)
        return output


class PerTokenSparseMoE(tf.keras.layers.Layer):
    """Per-token sparse MoE.

    这个层和 `PerTokenFFN` 的位置一致，只是把单个 FFN 换成多 expert 路由。

    Args:
        num_tokens:
            token 数 `T`。
        d_model:
            hidden dim `D`。
        mult:
            expert 内部隐层扩张倍数。
        num_experts:
            expert 数。
        dropout:
            expert 前后 dropout。
        l1_coef:
            稀疏正则系数。
        sparsity_ratio:
            稀疏率缩放项。
        use_dtsi:
            是否启用 DTSI 推理路由。
        routing_type:
            路由类型，支持 `"relu_dtsi"` 和 `"relu"`。
        name:
            TensorFlow layer 名称。

    Call args:
        x:
            `tf.Tensor`，shape `[B, T, D]`。
        training:
            是否训练态。

    Returns:
        tuple:
            - `y`: `tf.Tensor`，shape `[B, T, D]`
            - `l1_loss`: 标量 tensor
    """

    def __init__(
        self,
        num_tokens,
        d_model,
        mult=4,
        num_experts=4,
        dropout=0.0,
        l1_coef=0.0,
        sparsity_ratio=1.0,
        use_dtsi=True,
        routing_type="relu_dtsi",
        name=None,
    ):
        super(PerTokenSparseMoE, self).__init__(name=name)
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.mult = int(mult)
        self.num_experts = int(num_experts)
        self.dropout = float(dropout)
        self.l1_coef = float(l1_coef)
        self.sparsity_ratio = float(sparsity_ratio) if sparsity_ratio else 1.0
        self.use_dtsi = bool(use_dtsi)
        self.routing_type = str(routing_type).lower()

    def build(self, input_shape):
        hidden_dim = self.d_model * self.mult
        init = tf.variance_scaling_initializer(scale=2.0)
        self.W1 = self.add_weight(
            name="W1", shape=[self.num_tokens, self.num_experts, self.d_model, hidden_dim], initializer=init
        )
        self.b1 = self.add_weight(
            name="b1", shape=[self.num_tokens, self.num_experts, hidden_dim], initializer=tf.zeros_initializer()
        )
        self.W2 = self.add_weight(
            name="W2", shape=[self.num_tokens, self.num_experts, hidden_dim, self.d_model], initializer=init
        )
        self.b2 = self.add_weight(
            name="b2", shape=[self.num_tokens, self.num_experts, self.d_model], initializer=tf.zeros_initializer()
        )
        self.gate_w_train = self.add_weight(
            name="gate_w_train", shape=[self.num_tokens, self.d_model, self.num_experts], initializer=init
        )
        self.gate_b_train = self.add_weight(
            name="gate_b_train", shape=[self.num_tokens, self.num_experts], initializer=tf.zeros_initializer()
        )
        if self.use_dtsi:
            self.gate_w_infer = self.add_weight(
                name="gate_w_infer", shape=[self.num_tokens, self.d_model, self.num_experts], initializer=init
            )
            self.gate_b_infer = self.add_weight(
                name="gate_b_infer", shape=[self.num_tokens, self.num_experts], initializer=tf.zeros_initializer()
            )
        super(PerTokenSparseMoE, self).build(input_shape)

    def _router_logits(self, x, weight, bias):
        return tf.einsum("btd,tde->bte", x, weight) + bias

    def call(self, x, training=False):
        hidden = tf.einsum("btd,tedh->bteh", x, self.W1) + self.b1
        hidden = gelu(hidden)
        if self.dropout and training:
            hidden = tf.nn.dropout(hidden, keep_prob=1.0 - self.dropout)
        expert_out = tf.einsum("bteh,tehd->bted", hidden, self.W2) + self.b2
        if self.dropout and training:
            expert_out = tf.nn.dropout(expert_out, keep_prob=1.0 - self.dropout)

        gate_train_logits = self._router_logits(x, self.gate_w_train, self.gate_b_train)
        if self.routing_type == "relu_dtsi":
            gate_train = tf.nn.softmax(gate_train_logits, axis=-1)
        elif self.routing_type == "relu":
            gate_train = tf.nn.relu(gate_train_logits)
        else:
            raise ValueError("Unsupported routing_type: %s" % self.routing_type)

        if self.use_dtsi:
            gate_infer_logits = self._router_logits(x, self.gate_w_infer, self.gate_b_infer)
            gate_infer = tf.nn.relu(gate_infer_logits)
        else:
            gate_infer = gate_train

        gate = gate_train if training else gate_infer
        output = tf.reduce_sum(expert_out * tf.expand_dims(gate, -1), axis=2)

        if self.l1_coef > 0.0:
            scale = 1.0 / max(self.sparsity_ratio, 1e-6)
            l1_loss = self.l1_coef * scale * tf.reduce_mean(tf.reduce_sum(gate_infer, axis=-1))
        else:
            l1_loss = tf.constant(0.0)
        return output, l1_loss


class BaseSemanticTokenizer(object):
    """Semantic tokenizer 的基础实现。

    作用是把 heterogeneous feature embedding 压成固定长度的 token 序列，
    输出给 RankMixer 主干使用。

    约定：
        - `B`: batch size
        - `N_dense`: dense 特征个数
        - `N_seq`: 序列池化后的 token 个数
        - `E`: 输入 embedding 维度
        - `T`: tokenizer 输出 token 数
        - `D`: tokenizer 输出维度

    这里不处理原始字符串序列 lookup、hash 和 pooling。调用方需要先准备好
    `dense_embeddings` / `seq_embeddings`。
    """

    default_group_rules = []

    def __init__(
        self,
        target_tokens,
        d_model,
        embedding_dim,
        semantic_groups=None,
        group_rules=None,
        token_projection="linear",
        require_seq_coverage=False,
        require_dense_coverage=False,
        coverage_log_limit=20,
        name="semantic_tokenizer",
    ):
        self.target_tokens = int(target_tokens)
        self.d_model = int(d_model)
        self.embedding_dim = int(embedding_dim)
        self.semantic_groups = semantic_groups
        self.group_rules = group_rules
        self.token_projection = str(token_projection).lower()
        self.require_seq_coverage = bool(require_seq_coverage)
        self.require_dense_coverage = bool(require_dense_coverage)
        self.coverage_log_limit = int(coverage_log_limit)
        self.name = str(name)
        self.coverage_info = {}

    def _concat_and_project(self, tensors, scope_name):
        if len(tensors) == 1:
            concat = tensors[0]
        else:
            concat = tf.concat(tensors, axis=-1)
        return _dense(concat, units=self.d_model, activation=None, name=scope_name)

    def _pad_or_trim_tokens(self, tokens):
        token_count = tf.shape(tokens)[1]
        if self.target_tokens <= 0:
            return tokens
        if tokens.shape[1] is not None and tokens.shape[1] == self.target_tokens:
            return tokens
        if tokens.shape[1] is not None and tokens.shape[1] > self.target_tokens:
            return tokens[:, : self.target_tokens, :]
        pad_len = self.target_tokens - token_count
        pad = tf.zeros([tf.shape(tokens)[0], pad_len, self.d_model])
        return tf.concat([tokens, pad], axis=1)

    def _build_feature_map(self, dense_embeddings, dense_names, seq_embeddings, seq_names):
        feature_map = {}
        if dense_embeddings is not None and dense_names:
            for index, name in enumerate(dense_names):
                feature_map[name] = dense_embeddings[:, index, :]
        if seq_embeddings is not None and seq_names:
            for index, name in enumerate(seq_names):
                feature_map[name] = seq_embeddings[:, index, :]
        return feature_map

    def _resolve_group_features(self, group_features, available_names):
        resolved = []
        for raw in group_features:
            if raw in available_names:
                resolved.append(raw)
                continue
            pattern = raw[3:] if raw.startswith("re:") else raw
            if _looks_like_regex(raw):
                regex = re.compile(pattern)
                for name in available_names:
                    if regex.search(name) and name not in resolved:
                        resolved.append(name)
        return resolved

    def _coverage_report(self, matched_names, dense_feature_names, seq_feature_names):
        dense_names = list(dense_feature_names or [])
        seq_names = list(seq_feature_names or [])
        dense_set = set(dense_names)
        seq_set = set(seq_names)
        matched_set = set(matched_names)

        dense_missing = sorted(list(dense_set - matched_set))
        seq_missing = sorted(list(seq_set - matched_set))
        dense_ratio = 0.0 if not dense_set else (len(dense_set) - len(dense_missing)) / float(len(dense_set))
        seq_ratio = 0.0 if not seq_set else (len(seq_set) - len(seq_missing)) / float(len(seq_set))

        self.coverage_info = {
            "dense_total": len(dense_set),
            "dense_matched": len(dense_set) - len(dense_missing),
            "dense_missing_count": len(dense_missing),
            "dense_coverage_ratio": dense_ratio,
            "seq_total": len(seq_set),
            "seq_matched": len(seq_set) - len(seq_missing),
            "seq_missing_count": len(seq_missing),
            "seq_coverage_ratio": seq_ratio,
            "dense_missing_sample": dense_missing[: self.coverage_log_limit],
            "seq_missing_sample": seq_missing[: self.coverage_log_limit],
        }

        if self.require_dense_coverage and dense_missing:
            raise ValueError(
                "SemanticTokenizer dense coverage failed. missing=%d sample=%s"
                % (len(dense_missing), ", ".join(self.coverage_info["dense_missing_sample"]))
            )
        if self.require_seq_coverage and seq_missing:
            raise ValueError(
                "SemanticTokenizer seq coverage failed. missing=%d sample=%s"
                % (len(seq_missing), ", ".join(self.coverage_info["seq_missing_sample"]))
            )

        if seq_set:
            logger.info(
                "SemanticTokenizer seq coverage: matched=%d total=%d ratio=%.4f missing=%d",
                self.coverage_info["seq_matched"],
                self.coverage_info["seq_total"],
                self.coverage_info["seq_coverage_ratio"],
                self.coverage_info["seq_missing_count"],
            )

    def tokenize(
        self,
        dense_embeddings,
        dense_feature_names,
        seq_embeddings,
        seq_feature_names,
    ):
        """Build RankMixer tokens from feature embeddings.

        Args:
            dense_embeddings:
                `tf.Tensor`，shape `[B, N_dense, E]`，也可以为 `None`。
            dense_feature_names:
                dense 特征名列表，长度 `N_dense`。
            seq_embeddings:
                `tf.Tensor`，shape `[B, N_seq, E]`，也可以为 `None`。
            seq_feature_names:
                seq 特征名列表，长度 `N_seq`。

        Returns:
            tuple:
                - `tokens`: `tf.Tensor`，shape `[B, T, D]`
                - `token_count`: `int`，等于 `T`

        Raises:
            ValueError:
                当 dense 和 seq 都为空时抛出。
        """
        feature_names = []
        if dense_feature_names:
            feature_names.extend(list(dense_feature_names))
        if seq_feature_names:
            feature_names.extend(list(seq_feature_names))
        if not feature_names:
            raise ValueError("SemanticTokenizer needs at least one feature name.")

        feature_map = self._build_feature_map(
            dense_embeddings, dense_feature_names, seq_embeddings, seq_feature_names
        )
        available_names = list(feature_map.keys())
        groups = _normalize_groups(self.semantic_groups)

        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            if groups:
                tokens = []
                matched_names = set()
                for group_name, group_features in groups:
                    resolved = self._resolve_group_features(group_features, available_names)
                    matched_names.update(resolved)
                    tensors = [feature_map[name] for name in resolved if name in feature_map]
                    if not tensors:
                        ref = list(feature_map.values())[0]
                        tensors = [tf.zeros([tf.shape(ref)[0], self.embedding_dim])]
                    token = self._concat_and_project(
                        tensors, "token_proj_%s" % _sanitize_group_name(group_name)
                    )
                    tokens.append(token)
                self._coverage_report(matched_names, dense_feature_names, seq_feature_names)
                stacked = tf.stack(tokens, axis=1)
                stacked = self._pad_or_trim_tokens(stacked)
                stacked.set_shape([None, self.target_tokens, self.d_model])
                return stacked, self.target_tokens

            self._coverage_report(set(available_names), dense_feature_names, seq_feature_names)
            ordered_names = available_names
            compiled_rules = _compile_group_rules(self.group_rules, self.default_group_rules)
            if compiled_rules:
                ordered_indices = _assign_semantic_groups(
                    available_names, self.group_rules, self.default_group_rules
                )
                ordered_names = [available_names[index] for index in ordered_indices]

            ordered_embeddings = tf.stack([feature_map[name] for name in ordered_names], axis=1)
            feature_count = len(ordered_names)
            target_tokens = self.target_tokens if self.target_tokens > 0 else feature_count
            token_size = int((feature_count + target_tokens - 1) / target_tokens)
            pad_needed = target_tokens * token_size - feature_count
            if pad_needed > 0:
                pad_tensor = tf.zeros([tf.shape(ordered_embeddings)[0], pad_needed, self.embedding_dim])
                ordered_embeddings = tf.concat([ordered_embeddings, pad_tensor], axis=1)
            flat = tf.reshape(
                ordered_embeddings,
                [tf.shape(ordered_embeddings)[0], target_tokens, token_size * self.embedding_dim],
            )
            tokens = _dense(flat, units=self.d_model, activation=None, name="token_proj_chunk")
            tokens.set_shape([None, target_tokens, self.d_model])
            return tokens, target_tokens

    __call__ = tokenize


class SemanticTokenizerV1(BaseSemanticTokenizer):
    """Semantic tokenizer v1 with the original default semantic group rules."""

    default_group_rules = DEFAULT_SEMANTIC_GROUP_RULES_V1


class SemanticTokenizerV2(BaseSemanticTokenizer):
    """Semantic tokenizer v2 with empty default rules."""

    default_group_rules = DEFAULT_SEMANTIC_GROUP_RULES_V2


class SemanticTokenizerV3(BaseSemanticTokenizer):
    """Semantic tokenizer v3 with optional coverage checking."""

    default_group_rules = DEFAULT_SEMANTIC_GROUP_RULES_V3


# 默认 `SemanticTokenizer` 指向 v3。
SemanticTokenizer = SemanticTokenizerV3


@dataclass
class RankMixerTokenizerConfig(object):
    """`RankMixerTokenizer` 的配置。

    这个配置只描述 tokenizer 本身，不包含 RankMixer 主干。

    维度约定：
        - `E`: 输入 embedding 维度
        - `T`: tokenizer 输出 token 数
        - `D`: tokenizer 输出维度

    Args:
        target_tokens:
            输出 token 数 `T`。如果使用显式 `semantic_groups`，通常和分组数一致。
        d_model:
            tokenizer 输出维度 `D`，后续也应当和 RankMixer backbone 的 `d_model` 保持一致。
        embedding_dim:
            输入 embedding 维度 `E`。
        version:
            tokenizer 版本。支持 `"v1"`、`"v2"`、`"v3"`。
        semantic_groups:
            显式分组配置。每个 group 产出一个 token。
        group_rules:
            规则分组配置。没有显式 `semantic_groups` 时，按规则先排序再分桶。
        token_projection:
            token 投影方式。目前保留 `"linear"`。
        include_seq_in_tokenization:
            是否把 `seq_embeddings` 一起送进 tokenizer。
            如果设为 `False`，会先对 dense 部分做 tokenization，再把 seq token 直接拼到后面。
        require_seq_coverage:
            是否要求所有 seq feature name 都被分组逻辑覆盖。
        require_dense_coverage:
            是否要求所有 dense feature name 都被分组逻辑覆盖。
        coverage_log_limit:
            覆盖率日志里最多保留多少个 missing sample。
        name:
            TensorFlow scope 名称。
    """

    target_tokens: int
    d_model: int
    embedding_dim: int
    version: str = "v3"
    semantic_groups: Optional[Any] = None
    group_rules: Optional[Any] = None
    token_projection: str = "linear"
    include_seq_in_tokenization: bool = True
    require_seq_coverage: bool = False
    require_dense_coverage: bool = False
    coverage_log_limit: int = 20
    name: str = "rankmixer_tokenizer"


@dataclass
class TokenizerOutput(object):
    """`RankMixerTokenizer` 的输出。

    Attributes:
        tokens:
            tokenizer 结果，shape `[B, T_out, D]`。
        token_count:
            实际输出 token 数 `T_out`。
        dense_feature_names:
            实际使用的 dense 特征名，长度 `N_dense`。
        seq_feature_names:
            实际使用的 seq 特征名，长度 `N_seq`。
        coverage_info:
            tokenizer 内部记录的覆盖率信息。显式分组或规则分组时，可以用它检查是否有漏掉的特征。
    """

    tokens: tf.Tensor
    token_count: int
    dense_feature_names: Tuple[str, ...]
    seq_feature_names: Tuple[str, ...]
    coverage_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RankMixerBackboneConfig(object):
    """`RankMixerBackbone` 的配置。

    这个配置描述的是 RankMixer 主干，不包含前面的 embedding lookup 和 tokenizer。

    维度约定：
        - `T_model`: 真正进入 backbone 的 token 数
        - `D`: hidden dim

    其中 `T_model` 可能等于输入 token 数 `T`，也可能是 `T + 1`，取决于是否开启 `add_cls_token`。

    Args:
        num_layers:
            encoder block 数。
        d_model:
            hidden dim `D`。
        num_heads:
            token mixing head 数 `H`。当前实现要求 `H == T_model`。
            如果不传，会自动取实际 token 数。
        ffn_mult:
            per-token FFN 扩张倍数，隐层维度为 `D * ffn_mult`。
        token_mixing_dropout:
            token mixing dropout。
        ffn_dropout:
            FFN 或 MoE 内部 dropout。
        ln_style:
            layer norm 放置方式，支持 `"pre"` 和 `"post"`。
        use_final_ln:
            encoder stack 输出后是否再做一层 LayerNorm。
        add_cls_token:
            是否在 token 序列开头拼接一个 `[CLS]` token。
        use_input_ln:
            进入 encoder 前是否先做一层 LayerNorm。
        input_dropout:
            encoder 输入 dropout。
        output_pooling:
            输出池化方式，支持 `"mean"`、`"avg"`、`"cls"`。
        per_token_ffn:
            是否使用 per-token FFN。
        use_moe:
            是否用 sparse MoE 替换 FFN。
        moe_num_experts:
            MoE expert 数。
        moe_l1_coef:
            MoE 路由稀疏正则系数。
        moe_sparsity_ratio:
            MoE 稀疏率。
        moe_use_dtsi:
            是否启用 DTSI 路由。
        moe_routing_type:
            路由类型。
        cls_initializer_stddev:
            `[CLS]` token 初始化标准差。
        name:
            TensorFlow scope 名称。
    """

    num_layers: int
    d_model: int
    num_heads: Optional[int] = None
    ffn_mult: int = 4
    token_mixing_dropout: float = 0.0
    ffn_dropout: float = 0.0
    ln_style: str = "pre"
    use_final_ln: bool = True
    add_cls_token: bool = False
    use_input_ln: bool = False
    input_dropout: float = 0.0
    output_pooling: str = "mean"
    per_token_ffn: bool = True
    use_moe: bool = False
    moe_num_experts: int = 4
    moe_l1_coef: float = 0.0
    moe_sparsity_ratio: float = 1.0
    moe_use_dtsi: bool = True
    moe_routing_type: str = "relu_dtsi"
    cls_initializer_stddev: float = 0.02
    name: str = "rankmixer_backbone"


@dataclass
class RankMixerBackboneOutput(object):
    """`RankMixerBackbone` 的输出。

    Attributes:
        input_tokens:
            真正送进 encoder 的 token，shape `[B, T_model, D]`。
        encoded_tokens:
            encoder 输出，shape `[B, T_model, D]`。
        pooled_output:
            池化后的样本表示，shape `[B, D]`。
        token_count:
            实际进入 backbone 的 token 数 `T_model`。
        moe_loss:
            encoder 内部累计的 MoE 稀疏正则损失。不开启 MoE 时为 `0.0`。
    """

    input_tokens: tf.Tensor
    encoded_tokens: tf.Tensor
    pooled_output: tf.Tensor
    token_count: int
    moe_loss: tf.Tensor


@dataclass
class RankMixerPipelineOutput(object):
    """单入口 `RankMixer` / `RankMixerPipeline` 的输出。

    Attributes:
        tokens:
            tokenizer 输出 token，shape `[B, T, D]`。
        encoded_tokens:
            backbone 输出 token，shape `[B, T_model, D]`。
        pooled_output:
            最终聚合表示，shape `[B, D]`。
        token_count:
            backbone 实际 token 数 `T_model`。
        tokenizer_output:
            完整 tokenizer 输出对象。
        backbone_output:
            完整 backbone 输出对象。
    """

    tokens: tf.Tensor
    encoded_tokens: tf.Tensor
    pooled_output: tf.Tensor
    token_count: int
    tokenizer_output: TokenizerOutput
    backbone_output: RankMixerBackboneOutput


class LayerNorm(tf.keras.layers.Layer):
    """简化版 LayerNorm。

    Args:
        epsilon:
            数值稳定项。
        center:
            是否学习偏置项 `beta`。
        scale:
            是否学习缩放项 `gamma`。
        name:
            TensorFlow layer 名称。

    Call args:
        inputs:
            `tf.Tensor`，shape `[..., D]`。

    Returns:
        与输入同 shape 的 `tf.Tensor`。
    """

    def __init__(self, epsilon=1e-6, center=True, scale=True, name=None):
        super(LayerNorm, self).__init__(name=name)
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

    def build(self, input_shape):
        dim = input_shape[-1].value if hasattr(input_shape[-1], "value") else int(input_shape[-1])
        if self.scale:
            self.gamma = self.add_weight(name="gamma", shape=[dim], initializer=tf.ones_initializer())
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(name="beta", shape=[dim], initializer=tf.zeros_initializer())
        else:
            self.beta = None
        super(LayerNorm, self).build(input_shape)

    def call(self, inputs, **kwargs):
        mean, var = tf.nn.moments(inputs, axes=[-1], keepdims=True)
        outputs = (inputs - mean) / tf.sqrt(var + self.epsilon)
        if self.scale:
            outputs = outputs * self.gamma
        if self.center:
            outputs = outputs + self.beta
        return outputs


class RankMixerBlock(tf.keras.layers.Layer):
    """一个 RankMixer block。

    结构顺序：
        token mixing -> residual -> per-token FFN/MoE -> residual

    Args:
        num_tokens:
            token 数 `T`。
        d_model:
            hidden dim `D`。
        num_heads:
            token mixing head 数。
        ffn_mult:
            FFN 隐层扩张倍数。
        token_dp:
            token mixing dropout。
        ffn_dp:
            FFN 或 MoE dropout。
        ln_style:
            `"pre"` 或 `"post"`。
        use_moe:
            是否启用 sparse MoE。
        moe_experts:
            expert 数。
        moe_l1_coef:
            稀疏正则系数。
        moe_sparsity_ratio:
            稀疏率。
        moe_use_dtsi:
            是否启用 DTSI。
        moe_routing_type:
            路由类型。
        name:
            TensorFlow layer 名称。

    Call args:
        x:
            `tf.Tensor`，shape `[B, T, D]`。
        training:
            是否训练态。

    Returns:
        `tf.Tensor`，shape `[B, T, D]`。
    """

    def __init__(
        self,
        num_tokens,
        d_model,
        num_heads,
        ffn_mult,
        token_dp=0.0,
        ffn_dp=0.0,
        ln_style="pre",
        use_moe=False,
        moe_experts=4,
        moe_l1_coef=0.0,
        moe_sparsity_ratio=1.0,
        moe_use_dtsi=True,
        moe_routing_type="relu_dtsi",
        name=None,
    ):
        super(RankMixerBlock, self).__init__(name=name)
        self.ln1 = LayerNorm(name="ln1")
        self.ln2 = LayerNorm(name="ln2")
        self.ln_style = str(ln_style).lower()
        self.use_moe = bool(use_moe)
        self.token_mixer = ParameterFreeTokenMixer(
            num_tokens=num_tokens,
            d_model=d_model,
            num_heads=num_heads,
            dropout=token_dp,
            name="token_mixer",
        )
        if self.use_moe:
            self.per_token_ffn = PerTokenSparseMoE(
                num_tokens=num_tokens,
                d_model=d_model,
                mult=ffn_mult,
                num_experts=moe_experts,
                dropout=ffn_dp,
                l1_coef=moe_l1_coef,
                sparsity_ratio=moe_sparsity_ratio,
                use_dtsi=moe_use_dtsi,
                routing_type=moe_routing_type,
                name="per_token_moe",
            )
        else:
            self.per_token_ffn = PerTokenFFN(
                num_tokens=num_tokens,
                d_model=d_model,
                mult=ffn_mult,
                dropout=ffn_dp,
                name="per_token_ffn",
            )
        self.moe_loss = tf.constant(0.0)

    def call(self, x, training=False):
        moe_loss = tf.constant(0.0)
        if self.ln_style == "post":
            y = self.token_mixer(x, training=training)
            x = self.ln1(x + y)
            if self.use_moe:
                z, moe_loss = self.per_token_ffn(x, training=training)
            else:
                z = self.per_token_ffn(x, training=training)
            out = self.ln2(x + z)
        else:
            y = self.ln1(x)
            y = self.token_mixer(y, training=training)
            x = x + y
            z = self.ln2(x)
            if self.use_moe:
                z, moe_loss = self.per_token_ffn(z, training=training)
            else:
                z = self.per_token_ffn(z, training=training)
            out = x + z
        self.moe_loss = moe_loss
        return out


class RankMixerEncoder(tf.keras.layers.Layer):
    """堆叠后的 RankMixer encoder。

    Args:
        num_layers:
            block 层数。
        num_tokens:
            token 数 `T`。
        d_model:
            hidden dim `D`。
        num_heads:
            token mixing head 数。
        ffn_mult:
            FFN 隐层扩张倍数。
        token_dp:
            token mixing dropout。
        ffn_dp:
            FFN 或 MoE dropout。
        ln_style:
            `"pre"` 或 `"post"`。
        use_moe:
            是否启用 sparse MoE。
        moe_experts:
            expert 数。
        moe_l1_coef:
            稀疏正则系数。
        moe_sparsity_ratio:
            稀疏率。
        moe_use_dtsi:
            是否启用 DTSI。
        moe_routing_type:
            路由类型。
        use_final_ln:
            输出后是否再做一层 LayerNorm。
        name:
            TensorFlow layer 名称。

    Call args:
        x:
            `tf.Tensor`，shape `[B, T, D]`。
        training:
            是否训练态。

    Returns:
        `tf.Tensor`，shape `[B, T, D]`。
    """

    def __init__(
        self,
        num_layers,
        num_tokens,
        d_model,
        num_heads,
        ffn_mult,
        token_dp=0.0,
        ffn_dp=0.0,
        ln_style="pre",
        use_moe=False,
        moe_experts=4,
        moe_l1_coef=0.0,
        moe_sparsity_ratio=1.0,
        moe_use_dtsi=True,
        moe_routing_type="relu_dtsi",
        use_final_ln=True,
        name=None,
    ):
        super(RankMixerEncoder, self).__init__(name=name)
        self.use_final_ln = bool(use_final_ln)
        self.blocks = [
            RankMixerBlock(
                num_tokens=num_tokens,
                d_model=d_model,
                num_heads=num_heads,
                ffn_mult=ffn_mult,
                token_dp=token_dp,
                ffn_dp=ffn_dp,
                ln_style=ln_style,
                use_moe=use_moe,
                moe_experts=moe_experts,
                moe_l1_coef=moe_l1_coef,
                moe_sparsity_ratio=moe_sparsity_ratio,
                moe_use_dtsi=moe_use_dtsi,
                moe_routing_type=moe_routing_type,
                name="block_%d" % index,
            )
            for index in range(num_layers)
        ]
        self.final_ln = LayerNorm(name="encoder_ln")
        self.moe_loss = tf.constant(0.0)

    def call(self, x, training=False):
        out = x
        moe_losses = []
        for block in self.blocks:
            out = block(out, training=training)
            moe_losses.append(block.moe_loss)
        self.moe_loss = tf.add_n(moe_losses) if moe_losses else tf.constant(0.0)
        return self.final_ln(out) if self.use_final_ln else out


class RankMixerTokenizer(object):
    """Tokenizer 封装。

    这个类负责把外部已经准备好的 dense / seq embedding 变成 RankMixer token。

    输入维度：
        - `dense_embeddings`: `[B, N_dense, E]`
        - `seq_embeddings`: `[B, N_seq, E]`

    输出维度：
        - `tokens`: `[B, T_out, D]`

    其中：
        - `B`: batch size
        - `N_dense`: dense 特征个数
        - `N_seq`: seq token 个数
        - `E`: 输入 embedding 维度
        - `T_out`: tokenizer 最终输出 token 数
        - `D`: tokenizer 输出维度
    """

    _TOKENIZER_MAP = {
        "v1": SemanticTokenizerV1,
        "tokenization_v1": SemanticTokenizerV1,
        "semantic_v1": SemanticTokenizerV1,
        "v2": SemanticTokenizerV2,
        "tokenization_v2": SemanticTokenizerV2,
        "semantic_v2": SemanticTokenizerV2,
        "v3": SemanticTokenizerV3,
        "tokenization_v3": SemanticTokenizerV3,
        "semantic_v3": SemanticTokenizerV3,
    }

    def __init__(self, config: RankMixerTokenizerConfig):
        self.config = config
        version = str(config.version).lower()
        if version not in self._TOKENIZER_MAP:
            raise ValueError("Unsupported tokenizer version: %s" % config.version)
        tokenizer_cls = self._TOKENIZER_MAP[version]
        self._tokenizer = tokenizer_cls(
            target_tokens=config.target_tokens,
            d_model=config.d_model,
            embedding_dim=config.embedding_dim,
            semantic_groups=config.semantic_groups,
            group_rules=config.group_rules,
            token_projection=config.token_projection,
            require_seq_coverage=config.require_seq_coverage,
            require_dense_coverage=config.require_dense_coverage,
            coverage_log_limit=config.coverage_log_limit,
            name=config.name,
        )

    def tokenize(
        self,
        dense_embeddings: Optional[tf.Tensor],
        dense_feature_names: Optional[Sequence[str]] = None,
        seq_embeddings: Optional[tf.Tensor] = None,
        seq_feature_names: Optional[Sequence[str]] = None,
    ) -> TokenizerOutput:
        """把 feature embedding 转成 RankMixer token。

        Args:
            dense_embeddings:
                dense 特征 embedding，shape `[B, N_dense, E]`。可以为 `None`。
            dense_feature_names:
                dense 特征名列表，长度 `N_dense`。如果不传，要求 `N_dense` 可以从静态 shape 推出来。
            seq_embeddings:
                seq 特征 embedding，shape `[B, N_seq, E]`。这里默认外部已经完成 pooling。可以为 `None`。
            seq_feature_names:
                seq 特征名列表，长度 `N_seq`。如果不传，要求 `N_seq` 可以从静态 shape 推出来。

        Returns:
            `TokenizerOutput`

            - `tokens`: `[B, T_out, D]`
            - `token_count`: `T_out`
            - `coverage_info`: 命中分组和漏特征情况

        Raises:
            ValueError:
                当 dense 和 seq 都为空，或者输入维度与配置不一致时抛出。
        """
        if dense_embeddings is None and seq_embeddings is None:
            raise ValueError("At least one of dense_embeddings or seq_embeddings must be provided.")

        _validate_last_dim(dense_embeddings, self.config.embedding_dim, "dense_embeddings")
        _validate_last_dim(seq_embeddings, self.config.embedding_dim, "seq_embeddings")

        dense_feature_names = _normalize_feature_names(dense_embeddings, dense_feature_names, "dense")
        seq_feature_names = _normalize_feature_names(seq_embeddings, seq_feature_names, "seq")

        if self.config.include_seq_in_tokenization:
            tokens, token_count = self._tokenizer.tokenize(
                dense_embeddings,
                dense_feature_names,
                seq_embeddings,
                seq_feature_names,
            )
        else:
            tokens, token_count = self._tokenizer.tokenize(
                dense_embeddings,
                dense_feature_names,
                None,
                None,
            )
            seq_token_count = len(seq_feature_names)
            if seq_embeddings is not None and seq_token_count > 0:
                seq_proj = seq_embeddings
                if self.config.embedding_dim != self.config.d_model:
                    seq_proj = _dense(
                        seq_proj,
                        units=self.config.d_model,
                        activation=None,
                        name="%s_seq_token_proj" % self.config.name,
                    )
                tokens = tf.concat([tokens, seq_proj], axis=1)
                token_count += seq_token_count

        tokens.set_shape([None, token_count, self.config.d_model])
        return TokenizerOutput(
            tokens=tokens,
            token_count=token_count,
            dense_feature_names=dense_feature_names,
            seq_feature_names=seq_feature_names,
            coverage_info=dict(getattr(self._tokenizer, "coverage_info", {}) or {}),
        )

    __call__ = tokenize


class RankMixerBackbone(object):
    """RankMixer 主干封装。

    这个类不关心特征名，也不关心 embedding lookup。它只接收 token 序列，
    输出编码后的 token 和聚合后的样本表示。

    输入维度：
        - `tokens`: `[B, T, D]`

    输出维度：
        - `encoded_tokens`: `[B, T_model, D]`
        - `pooled_output`: `[B, D]`

    其中 `T_model` 可能等于 `T`，也可能因为 `add_cls_token=True` 变成 `T + 1`。
    """

    def __init__(self, config: RankMixerBackboneConfig):
        self.config = config
        self._input_ln = LayerNorm(name="%s_input_ln" % config.name) if config.use_input_ln else None
        self._encoder_cache = {}

    def _get_encoder(self, token_count: int, num_heads: int) -> RankMixerEncoder:
        key = (int(token_count), int(num_heads))
        if key not in self._encoder_cache:
            self._encoder_cache[key] = RankMixerEncoder(
                num_layers=self.config.num_layers,
                num_tokens=token_count,
                d_model=self.config.d_model,
                num_heads=num_heads,
                ffn_mult=self.config.ffn_mult,
                token_dp=self.config.token_mixing_dropout,
                ffn_dp=self.config.ffn_dropout,
                ln_style=self.config.ln_style,
                use_moe=self.config.use_moe,
                moe_experts=self.config.moe_num_experts,
                moe_l1_coef=self.config.moe_l1_coef,
                moe_sparsity_ratio=self.config.moe_sparsity_ratio,
                moe_use_dtsi=self.config.moe_use_dtsi,
                moe_routing_type=self.config.moe_routing_type,
                use_final_ln=self.config.use_final_ln,
                name="%s_t%d_h%d" % (self.config.name, token_count, num_heads),
            )
        return self._encoder_cache[key]

    def encode(
        self,
        tokens: tf.Tensor,
        token_count: Optional[int] = None,
        training: bool = False,
    ) -> RankMixerBackboneOutput:
        """对 token 序列做 RankMixer 编码。

        Args:
            tokens:
                输入 token，shape `[B, T, D]`。
            token_count:
                实际 token 数 `T`。当 `tokens.shape[1]` 是动态维时需要显式传入。
            training:
                是否训练态。会影响 dropout 和 MoE 路由。

        Returns:
            `RankMixerBackboneOutput`

            - `input_tokens`: `[B, T_model, D]`
            - `encoded_tokens`: `[B, T_model, D]`
            - `pooled_output`: `[B, D]`
            - `token_count`: `T_model`
            - `moe_loss`: scalar tensor

        Raises:
            ValueError:
                当 `d_model`、`num_heads`、`output_pooling` 配置不合法时抛出。
        """
        _validate_last_dim(tokens, self.config.d_model, "tokens")

        base_token_count = _resolve_token_count(tokens, token_count)
        model_tokens = tokens
        model_token_count = base_token_count

        if self.config.add_cls_token:
            with tf.compat.v1.variable_scope(self.config.name, reuse=tf.compat.v1.AUTO_REUSE):
                cls_embed = tf.compat.v1.get_variable(
                    "cls_token",
                    shape=[1, 1, self.config.d_model],
                    initializer=tf.random_normal_initializer(stddev=self.config.cls_initializer_stddev),
                )
            cls_token = tf.tile(cls_embed, [tf.shape(tokens)[0], 1, 1])
            model_tokens = tf.concat([cls_token, tokens], axis=1)
            model_token_count += 1

        num_heads = int(self.config.num_heads) if self.config.num_heads is not None else int(model_token_count)
        if num_heads != model_token_count:
            raise ValueError(
                "paper-strict token mixing requires num_heads == token_count, "
                "got num_heads=%d token_count=%d." % (num_heads, model_token_count)
            )
        if not self.config.per_token_ffn and not self.config.use_moe:
            raise ValueError("Enable per_token_ffn=True or use_moe=True.")

        if self._input_ln is not None:
            model_tokens = self._input_ln(model_tokens)
        if self.config.input_dropout and training:
            model_tokens = tf.nn.dropout(model_tokens, keep_prob=1.0 - self.config.input_dropout)

        encoder = self._get_encoder(model_token_count, num_heads)
        encoded_tokens = encoder(model_tokens, training=training)
        encoded_tokens.set_shape([None, model_token_count, self.config.d_model])

        pooling = str(self.config.output_pooling).lower()
        if pooling in ("mean", "avg"):
            pooled_output = tf.reduce_mean(encoded_tokens, axis=1)
        elif pooling == "cls":
            if not self.config.add_cls_token:
                raise ValueError("output_pooling='cls' requires add_cls_token=True.")
            pooled_output = encoded_tokens[:, 0, :]
        else:
            raise ValueError("Unsupported output_pooling: %s" % self.config.output_pooling)

        return RankMixerBackboneOutput(
            input_tokens=model_tokens,
            encoded_tokens=encoded_tokens,
            pooled_output=pooled_output,
            token_count=model_token_count,
            moe_loss=encoder.moe_loss,
        )

    __call__ = encode


class RankMixerPipeline(object):
    """Tokenizer 和 backbone 的组合封装。

    这个类只是把两步串起来，方便外部少接一层。
    如果你希望入口更直观，可以直接用下面的 `RankMixer`。
    """

    def __init__(
        self,
        tokenizer_config: RankMixerTokenizerConfig,
        backbone_config: RankMixerBackboneConfig,
    ):
        self.tokenizer = RankMixerTokenizer(tokenizer_config)
        self.backbone = RankMixerBackbone(backbone_config)

    def forward(
        self,
        dense_embeddings: Optional[tf.Tensor],
        dense_feature_names: Optional[Sequence[str]] = None,
        seq_embeddings: Optional[tf.Tensor] = None,
        seq_feature_names: Optional[Sequence[str]] = None,
        training: bool = False,
    ) -> RankMixerPipelineOutput:
        """一次调用完成 tokenizer 和 backbone。

        Args:
            dense_embeddings:
                dense 特征 embedding，shape `[B, N_dense, E]`。
            dense_feature_names:
                dense 特征名列表，长度 `N_dense`。
            seq_embeddings:
                seq 特征 embedding，shape `[B, N_seq, E]`。
            seq_feature_names:
                seq 特征名列表，长度 `N_seq`。
            training:
                是否训练态。

        Returns:
            `RankMixerPipelineOutput`

            - `tokens`: `[B, T, D]`
            - `encoded_tokens`: `[B, T_model, D]`
            - `pooled_output`: `[B, D]`
        """
        tokenizer_output = self.tokenizer(
            dense_embeddings=dense_embeddings,
            dense_feature_names=dense_feature_names,
            seq_embeddings=seq_embeddings,
            seq_feature_names=seq_feature_names,
        )
        backbone_output = self.backbone(
            tokens=tokenizer_output.tokens,
            token_count=tokenizer_output.token_count,
            training=training,
        )
        return RankMixerPipelineOutput(
            tokens=tokenizer_output.tokens,
            encoded_tokens=backbone_output.encoded_tokens,
            pooled_output=backbone_output.pooled_output,
            token_count=backbone_output.token_count,
            tokenizer_output=tokenizer_output,
            backbone_output=backbone_output,
        )

    __call__ = forward


class RankMixer(RankMixerPipeline):
    """对外推荐使用的单入口类。

    大多数场景下，外部项目只需要准备好 embedding 和 feature name，
    然后直接调用这个类即可。

    初始化参数：
        tokenizer_config:
            `RankMixerTokenizerConfig`
        backbone_config:
            `RankMixerBackboneConfig`

    调用输入：
        - `dense_embeddings`: `[B, N_dense, E]`
        - `dense_feature_names`: 长度 `N_dense`
        - `seq_embeddings`: `[B, N_seq, E]`
        - `seq_feature_names`: 长度 `N_seq`

    调用输出：
        `RankMixerPipelineOutput`

        - `tokens`: `[B, T, D]`
        - `encoded_tokens`: `[B, T_model, D]`
        - `pooled_output`: `[B, D]`

    注意：
        这个类默认只输出 backbone 表征，不直接包含 CTR、CVR、CTCVR 等业务 head。
        如果需要业务预测值，应该在 `pooled_output` 后面继续接自己的任务层。
    """

    pass


__all__ = [
    "gelu",
    "SemanticTokenizer",
    "SemanticTokenizerV1",
    "SemanticTokenizerV2",
    "SemanticTokenizerV3",
    "ParameterFreeTokenMixer",
    "PerTokenFFN",
    "PerTokenSparseMoE",
    "LayerNorm",
    "RankMixerBlock",
    "RankMixerEncoder",
    "TokenizerOutput",
    "RankMixerTokenizerConfig",
    "RankMixerTokenizer",
    "RankMixerBackboneConfig",
    "RankMixerBackboneOutput",
    "RankMixerBackbone",
    "RankMixerPipelineOutput",
    "RankMixerPipeline",
    "RankMixer",
]


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.fc1 = nn.Linear(hidden_size * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hist_emb, target_emb, seq_mask):
        batch_size, seq_len, hidden_size = hist_emb.shape
        if target_emb.dim() == 2:
            target_emb = target_emb.unsqueeze(1)

        target_emb_expanded = target_emb.expand(-1, seq_len, -1)

        concat = torch.cat([
            hist_emb,
            target_emb_expanded,
            hist_emb * target_emb_expanded
        ], dim=-1)

        attn_weights = self.fc2(self.relu(self.fc1(concat))).squeeze(-1)

        padding_mask = seq_mask < 0.5
        attn_weights = attn_weights.masked_fill(padding_mask, -1e9)

        attn_weights = self.softmax(attn_weights)
        output = torch.matmul(attn_weights.unsqueeze(1), hist_emb).squeeze(1)

        return output

