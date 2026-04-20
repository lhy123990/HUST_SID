import numpy as np
import torch
from torch import nn


class DNN(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout=0.0):
        super().__init__()
        layers = []
        in_dim = input_dim
        for out_dim in hidden_units:
            layers.append(nn.Linear(in_dim, out_dim))
            nn.init.xavier_uniform_(layers[-1].weight)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = in_dim

    def forward(self, x):
        return self.net(x)


class CrossNet(nn.Module):
    def __init__(self, input_dim, depth=2):
        super().__init__()
        self.depth = depth
        self.kernels = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, input_dim)) for _ in range(depth)
        ])
        self.bias = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, 1)) for _ in range(depth)
        ])
        for i in range(depth):
            nn.init.xavier_normal_(self.kernels[i])
            nn.init.zeros_(self.bias[i])

    def forward(self, x0):
        xl = x0
        for i in range(self.depth):
            xl_w = torch.mm(xl, self.kernels[i]) + self.bias[i].t()
            xl = x0 * xl_w + xl
        return xl


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hist_emb, target_emb, seq_mask):
        batch_size, seq_len, _ = hist_emb.shape
        target_expand = target_emb.unsqueeze(1).expand(-1, seq_len, -1)
        concat = torch.cat([hist_emb, target_expand, hist_emb * target_expand], dim=-1)

        attn_weights = self.fc2(self.relu(self.fc1(concat))).squeeze(-1)
        padding_mask = seq_mask < 0.5
        attn_weights = attn_weights.masked_fill(padding_mask, -1e9)
        attn_weights = self.softmax(attn_weights)

        output = torch.matmul(attn_weights.unsqueeze(1), hist_emb).squeeze(1)
        return output


class DCNV2(nn.Module):
    """KuaiRec all-feature DCN + DIN model (no SID/PID)."""

    def __init__(self, config):
        super().__init__()
        self.embedding_size = config.embedding_size

        self.register_buffer("user_sparse_lookup", torch.from_numpy(np.array(config.user_sparse_lookup, copy=True)))
        self.register_buffer("user_dense_lookup", torch.from_numpy(np.array(config.user_dense_lookup, copy=True)))
        self.register_buffer("item_sparse_lookup", torch.from_numpy(np.array(config.item_sparse_lookup, copy=True)))
        self.register_buffer("item_dense_lookup", torch.from_numpy(np.array(config.item_dense_lookup, copy=True)))

        sid_lookup = getattr(config, "sid_lookup", None)
        self.use_sid = sid_lookup is not None
        if self.use_sid:
            self.register_buffer("sid_lookup", torch.from_numpy(np.array(sid_lookup, copy=True)))
            self.num_sid_cols = int(self.sid_lookup.size(1))
            max_sid = int(self.sid_lookup.max().item()) if self.sid_lookup.numel() > 0 else 0
            self.sid_vocab_size = max(1, max_sid + 1)
            self.sid_embeddings = nn.ModuleList([
                nn.Embedding(self.sid_vocab_size, self.embedding_size)
                for _ in range(self.num_sid_cols)
            ])
            for emb in self.sid_embeddings:
                nn.init.xavier_uniform_(emb.weight)
        else:
            self.num_sid_cols = 0
            self.sid_vocab_size = 0
            self.sid_embeddings = nn.ModuleList()

        self.user_sparse_vocab_sizes = config.user_sparse_vocab_sizes
        self.item_sparse_vocab_sizes = config.item_sparse_vocab_sizes

        self.user_dense_dim = int(config.user_dense_dim)
        self.item_dense_dim = int(config.item_dense_dim)

        self.num_users = int(self.user_sparse_lookup.size(0))
        self.num_items = int(self.item_sparse_lookup.size(0))

        self.user_id_embedding = nn.Embedding(self.num_users, self.embedding_size)
        self.item_id_embedding = nn.Embedding(self.num_items, self.embedding_size)

        self.user_sparse_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, self.embedding_size)
            for vocab_size in self.user_sparse_vocab_sizes
        ])
        self.item_sparse_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, self.embedding_size)
            for vocab_size in self.item_sparse_vocab_sizes
        ])

        for emb in self.user_sparse_embeddings:
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.item_sparse_embeddings:
            nn.init.xavier_uniform_(emb.weight)

        self.user_dense_projector = nn.Sequential(
            nn.Linear(self.user_dense_dim, self.embedding_size),
            nn.LayerNorm(self.embedding_size),
            nn.ReLU(),
        )
        self.item_dense_projector = nn.Sequential(
            nn.Linear(self.item_dense_dim, self.embedding_size),
            nn.LayerNorm(self.embedding_size),
            nn.ReLU(),
        )

        self.user_vec_dim = (1 + len(self.user_sparse_vocab_sizes) + 1) * self.embedding_size
        self.item_vec_dim = (1 + len(self.item_sparse_vocab_sizes) + 1 + self.num_sid_cols) * self.embedding_size

        self.attention = AttentionLayer(self.item_vec_dim)

        self.total_input_dim = self.user_vec_dim + self.item_vec_dim + self.item_vec_dim 
        self.cross_net = CrossNet(self.total_input_dim, depth=config.cross_depth)
        self.dnn = DNN(self.total_input_dim, config.mlp_hidden_units, dropout=config.dropout)

        final_dim = self.total_input_dim + self.dnn.output_dim
        self.linear = nn.Linear(final_dim, 1)

    def _safe_index(self, x, max_size):
        x = x.long()
        x = torch.where(x < 0, torch.zeros_like(x), x)
        x = torch.where(x >= max_size, torch.zeros_like(x), x)
        return x

    def _encode_user(self, user_ids):
        user_ids = self._safe_index(user_ids, self.user_sparse_lookup.size(0))

        user_id_emb = self.user_id_embedding(user_ids)

        sparse_ids = self.user_sparse_lookup[user_ids]  # [B, Uc]
        sparse_parts = []
        for i, emb in enumerate(self.user_sparse_embeddings):
            col_ids = self._safe_index(sparse_ids[:, i], emb.num_embeddings)
            sparse_parts.append(emb(col_ids))

        dense_vals = self.user_dense_lookup[user_ids].float()  # [B, Ud]
        dense_proj = self.user_dense_projector(dense_vals)

        return torch.cat([user_id_emb] + sparse_parts + [dense_proj], dim=-1)

    def _encode_item_target(self, item_ids):
        item_ids = self._safe_index(item_ids, self.item_sparse_lookup.size(0))

        item_id_emb = self.item_id_embedding(item_ids)

        sparse_ids = self.item_sparse_lookup[item_ids]  # [B, Ic]
        sparse_parts = []
        for i, emb in enumerate(self.item_sparse_embeddings):
            col_ids = self._safe_index(sparse_ids[:, i], emb.num_embeddings)
            sparse_parts.append(emb(col_ids))

        dense_vals = self.item_dense_lookup[item_ids].float()  # [B, Id]
        dense_proj = self.item_dense_projector(dense_vals)

        sid_parts = []
        if self.use_sid:
            sid_ids = self.sid_lookup[item_ids]  # [B, S]
            for i, emb in enumerate(self.sid_embeddings):
                col_ids = self._safe_index(sid_ids[:, i], emb.num_embeddings)
                sid_parts.append(emb(col_ids))

        target_item_vec = torch.cat([item_id_emb] + sparse_parts + [dense_proj] + sid_parts, dim=-1)
        target_sid_vec = torch.cat(sid_parts, dim=-1) if len(sid_parts) > 0 else None
        return target_item_vec, target_sid_vec

    def _encode_item_seq(self, seq_item_ids):
        batch_size, seq_len = seq_item_ids.shape
        safe_seq_ids = self._safe_index(seq_item_ids, self.item_sparse_lookup.size(0))

        item_id_emb = self.item_id_embedding(safe_seq_ids)  # [B, L, E]

        flat_ids = safe_seq_ids.view(-1)
        sparse_ids = self.item_sparse_lookup[flat_ids].view(batch_size, seq_len, -1)  # [B, L, Ic]

        sparse_parts = []
        for i, emb in enumerate(self.item_sparse_embeddings):
            col_ids = self._safe_index(sparse_ids[:, :, i], emb.num_embeddings)
            sparse_parts.append(emb(col_ids))

        dense_vals = self.item_dense_lookup[flat_ids].float().view(batch_size, seq_len, -1)  # [B, L, Id]
        dense_proj = self.item_dense_projector(dense_vals.view(-1, dense_vals.size(-1))).view(batch_size, seq_len, -1)

        sid_parts = []
        if self.use_sid:
            sid_ids = self.sid_lookup[flat_ids].view(batch_size, seq_len, -1)  # [B, L, S]
            for i, emb in enumerate(self.sid_embeddings):
                col_ids = self._safe_index(sid_ids[:, :, i], emb.num_embeddings)
                sid_parts.append(emb(col_ids))

        return torch.cat([item_id_emb] + sparse_parts + [dense_proj] + sid_parts, dim=-1)

    def forward(self, batch):
        user_ids = batch["user_id"]
        target_item_ids = batch["target_video_id"]
        hist_item_ids = batch["hist_video_ids"]
        hist_mask = batch["hist_mask"]

        user_vec = self._encode_user(user_ids)
        target_item_vec, _ = self._encode_item_target(target_item_ids)
        hist_item_vec = self._encode_item_seq(hist_item_ids)

        din_output = self.attention(hist_item_vec, target_item_vec, hist_mask)

        doc_list = [user_vec, target_item_vec, din_output]
        total_emb = torch.cat(doc_list, dim=1)
        stack = torch.cat([self.cross_net(total_emb), self.dnn(total_emb)], dim=1)
        pred = torch.sigmoid(self.linear(stack))
        return pred
