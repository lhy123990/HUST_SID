import torch
from torch import nn


class DNN(nn.Module):
    """标准 MLP"""
    def __init__(self, input_dim, hidden_units, dropout=0.0):
        super(DNN, self).__init__()
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
    """DCNv2 Cross Network"""
    def __init__(self, input_dim, depth=2):
        super(CrossNet, self).__init__()
        self.depth = depth
        self.kernels = nn.ParameterList(
            [nn.Parameter(torch.Tensor(input_dim, input_dim)) for _ in range(depth)]
        )
        self.bias = nn.ParameterList(
            [nn.Parameter(torch.Tensor(input_dim, 1)) for _ in range(depth)]
        )
        for i in range(depth):
            nn.init.xavier_normal_(self.kernels[i])
            nn.init.zeros_(self.bias[i])

    def forward(self, x0):
        x_l = x0
        for i in range(self.depth):
            xl_w = torch.mm(x_l, self.kernels[i]) + self.bias[i].t()
            x_l = x0 * xl_w + x_l
        return x_l


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


class DCNV2(nn.Module):
    def __init__(self, config):
        super(DCNV2, self).__init__()

        self.field_dims = config.field_dims
        self.embedding_size = config.embedding_size
        self.feature_names = config.feature_names

        self.feat_name_to_idx = {name: i for i, name in enumerate(self.feature_names)}

        # 1. 基础特征 Embeddings
        self.dnn_embeddings = nn.ModuleList()
        for i, size in enumerate(self.field_dims):
            self.dnn_embeddings.append(nn.Embedding(size + 1, self.embedding_size))

        # 2. 注册 Attribute 查找表
        if hasattr(config, 'attr_lookups'):
            for key, tensor in config.attr_lookups.items():
                self.register_buffer(f'lookup_{key}', tensor)

        # === 3. SID 初始化 ===
        self.num_sid_cols = 0
        self.sid_embeddings = nn.ModuleList()

        if hasattr(config, 'sid_lookup') and config.sid_lookup is not None:
            self.register_buffer('sid_lookup', torch.from_numpy(config.sid_lookup))
            self.num_sid_cols = config.sid_lookup.shape[1]

            max_sid_val = self.sid_lookup.max().item()
            sid_vocab_size = int(max_sid_val + 10)
            print(f"[DCNv2] Enabled SID with {self.num_sid_cols} columns. Vocab Size: {sid_vocab_size}")

            for _ in range(self.num_sid_cols):
                self.sid_embeddings.append(nn.Embedding(sid_vocab_size, self.embedding_size))

        self.seq_attr_list = ['206', '213', '214']

        # === 4. PID 初始化（提前到这里，先计算维度）===
        self.use_pid = getattr(config, 'use_pid', False)
        self.num_pid_blocks = 0

        if self.use_pid:
            print("[DCNv2] Enabling PID...")
            if hasattr(config, 'pid_lookup') and config.pid_lookup is not None:
                self.register_buffer('pid_lookup', torch.from_numpy(config.pid_lookup))
                self.register_buffer('pid_sim_lookup', torch.from_numpy(config.pid_sim_lookup))
                self.num_pid_k = config.pid_lookup.shape[1]

                self.pid_linear = nn.Linear(self.num_pid_k, self.num_pid_k)
                nn.init.xavier_uniform_(self.pid_linear.weight)

                self.num_pid_blocks = 1
            else:
                raise ValueError("use_pid is True but pid_lookup is missing in config")

        # 5. Item Dimension（DIN里每个 item 的拼接维度）
        # ID本身(1) + 属性(len) + SID(cols) + PID(1)
        self.item_total_dim = (
            1 + len(self.seq_attr_list) + self.num_sid_cols + getattr(self, 'num_pid_blocks', 0)
        ) * self.embedding_size
        print(f"[DCNv2] Item Total Dim per element: {self.item_total_dim} (PID blocks: {self.num_pid_blocks})")

        # 6. Attention（DIN）
        self.attention = AttentionLayer(self.item_total_dim)

        # 7. DCN/DNN 输入维度
        self.num_dnn_fields = len(self.field_dims)
        self.dnn_input_dim = self.num_dnn_fields * self.embedding_size

        # PID 作为额外 block 拼到 total_emb
        if self.use_pid:
            self.pid_dim = self.embedding_size
            self.pid_projector = nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size),
                nn.LayerNorm(self.embedding_size),
                nn.GELU(),
                nn.Linear(self.embedding_size, self.embedding_size)
            )
        else:
            self.pid_dim = 0

        # SID 作为额外 block 拼到 total_emb（target-side SID 再 cat 一次）
        self.sid_dim = (self.num_sid_cols * self.embedding_size) if self.num_sid_cols > 0 else 0

        # [关键修正] total_emb = dnn_flat + din_output + target_sid_vec + target_pid_emb
        self.total_input_dim = self.dnn_input_dim + self.item_total_dim + self.sid_dim + self.pid_dim
        print(f"[DCNv2] Total Input Dim: {self.total_input_dim}")

        # 8. Network
        self.cross_net = CrossNet(self.total_input_dim, depth=config.cross_depth)
        self.dnn = DNN(self.total_input_dim, config.mlp_hidden_units, dropout=config.dropout)

        final_dim = self.total_input_dim + self.dnn.output_dim
        self.linear = nn.Linear(final_dim, 1)

        self.has_printed_neighbor_info = False

    def get_pid_embedding(self, item_ids_input):
        """
        计算 PID Embedding 和 Monotonicity Loss
        item_ids_input: [Batch] or [Batch, SeqLen]
        """
        item_ids_input = item_ids_input.long()

        original_shape = item_ids_input.shape
        flat_ids = item_ids_input.view(-1)

        # 1. 查找相似物品
        max_idx = self.pid_lookup.size(0)
        safe_ids = flat_ids.clone()
        safe_ids[safe_ids >= max_idx] = 0
        safe_ids[safe_ids < 0] = 0

        neighbor_ids = self.pid_lookup[safe_ids]       # [N, k]
        neighbor_sims = self.pid_sim_lookup[safe_ids]  # [N, k]

        # 获取 Embedding 层（共享 Main Item Embedding）
        idx_205 = self.feat_name_to_idx.get('205')
        item_emb_layer = self.dnn_embeddings[idx_205]

        if not self.has_printed_neighbor_info:
            print(f"\n[PID Check]")
            print(f"  Lookup Max Size: {max_idx}")
            print(f"  Item Vocab Size: {item_emb_layer.num_embeddings}")
            print(f"  Sample Neighbor IDs: {neighbor_ids[0].tolist()}")
            print(f"  Max Neighbor ID: {neighbor_ids.max().item()}")
            if neighbor_ids.max().item() >= item_emb_layer.num_embeddings:
                print(f"  [CRITICAL] Neighbor IDs contain values larger than Vocab! Are these Raw IDs?")
            self.has_printed_neighbor_info = True

        neighbor_ids_safe = neighbor_ids.clone()
        neighbor_ids_safe[neighbor_ids_safe >= item_emb_layer.num_embeddings] = 0
        neighbor_ids_safe[neighbor_ids_safe < 0] = 0

        neighbor_embs = item_emb_layer(neighbor_ids_safe)
        neighbor_embs = neighbor_embs.detach()  # 防止长尾反向污染热门 item emb

        # 3. 权重
        weights = torch.sigmoid(self.pid_linear(neighbor_sims))  # [N, k]

        if not getattr(self, 'has_printed_pid_debug', False):
            print(f"\n[PID Runtime Debug]")
            print(f"  Input Simulations (neighbor_sims): \n{neighbor_sims[0].detach().cpu().numpy()}")
            print(f"  Output Weights (Sigmoid): \n{weights[0].detach().cpu().numpy()}")
            w_min, w_max = weights.min().item(), weights.max().item()
            print(f"  Weights Range in batch: Min={w_min:.4f}, Max={w_max:.4f}")
            if hasattr(self, 'pid_linear'):
                print(f"  Linear Layer Weights Sample: \n{self.pid_linear.weight.data[0][:10].cpu().numpy()}...")
                print(f"  Linear Layer Grad Exists: {self.pid_linear.weight.grad is not None}")
            self.has_printed_pid_debug = True

        # 4. Monotonicity Loss
        diff = weights[:, 1:] - weights[:, :-1]
        mono_loss = torch.sum(torch.relu(diff))

        # 5. 加权平均
        mask = (neighbor_ids != 0).float().unsqueeze(-1)  # [N, k, 1]
        weighted_embs = neighbor_embs * weights.unsqueeze(-1) * mask
        pid_emb = torch.sum(weighted_embs, dim=1)  # [N, E]

        # 空间投影（让 PID 更像“原型空间”）
        pid_emb = self.pid_projector(pid_emb)  # [N, E]

        pid_emb = pid_emb.view(*original_shape, -1)
        mono_loss = mono_loss / flat_ids.size(0)

        return pid_emb, mono_loss

    def forward(self, dnn_feat, seq_feat, seq_mask):
        batch_size = dnn_feat.shape[0]
        device = dnn_feat.device
        total_aux_loss = torch.tensor(0.0, device=device)

        # target-side SID 额外 block（默认 0）
        target_sid_vec = None
        if self.sid_dim > 0:
            target_sid_vec = torch.zeros(batch_size, self.sid_dim, device=device)

        # 1) Embedding DNN Features
        dnn_emb_list = []
        for i, emb_layer in enumerate(self.dnn_embeddings):
            dnn_emb_list.append(emb_layer(dnn_feat[:, i]))
        dnn_embs_stack = torch.stack(dnn_emb_list, dim=1)
        dnn_embs_flat = dnn_embs_stack.view(batch_size, -1)

        idx_205 = self.feat_name_to_idx.get('205')

        # --- 2) Construct Target Item (DIN query) ---
        target_parts = []

        # 2.1 Target ID
        if idx_205 is not None:
            target_parts.append(dnn_embs_stack[:, idx_205, :])
        else:
            target_parts.append(torch.zeros(batch_size, self.embedding_size, device=device))

        # 2.2 Attributes (206, 213, 214)
        for col in self.seq_attr_list:
            idx = self.feat_name_to_idx.get(col)
            if idx is not None:
                target_parts.append(dnn_embs_stack[:, idx, :])
            else:
                target_parts.append(torch.zeros(batch_size, self.embedding_size, device=device))

        # 2.3 SID (target)：既进 DIN，也额外生成 target_sid_vec 给 DCN/DNN
        if self.num_sid_cols > 0:
            if idx_205 is not None:
                target_ids = dnn_feat[:, idx_205].long()

                max_len = self.sid_lookup.size(0)
                safe_ids = target_ids.clone()
                safe_ids[safe_ids >= max_len] = 0
                safe_ids[safe_ids < 0] = 0

                target_sids = self.sid_lookup[safe_ids]  # [B, C]

                sid_parts_flat = []
                for i in range(self.num_sid_cols):
                    sid_emb = self.sid_embeddings[i](target_sids[:, i])  # [B, E]
                    target_parts.append(sid_emb)         # 进 DIN
                    sid_parts_flat.append(sid_emb)       # 额外 block

                target_sid_vec = torch.cat(sid_parts_flat, dim=-1)  # [B, C*E]
            else:
                for _ in range(self.num_sid_cols):
                    target_parts.append(torch.zeros(batch_size, self.embedding_size, device=device))
                # target_sid_vec 维持默认 0

        # 2.4 PID（target）
        target_pid_emb = None
        if getattr(self, 'use_pid', False) and idx_205 is not None:
            target_ids = dnn_feat[:, idx_205]
            pid_emb, loss = self.get_pid_embedding(target_ids)
            target_parts.append(pid_emb)
            total_aux_loss = total_aux_loss + loss
            target_pid_emb = pid_emb

        target_item_emb = torch.cat(target_parts, dim=-1)  # [B, item_total_dim]

        # --- 3) Construct Sequence Items (DIN key) ---
        seq_parts = []

        # 3.1 ID
        if idx_205 is not None:
            seq_parts.append(self.dnn_embeddings[idx_205](seq_feat))
        else:
            seq_parts.append(torch.zeros(batch_size, seq_feat.size(1), self.embedding_size, device=device))

        flat_seq_ids = seq_feat.reshape(-1)

        # 3.2 Attributes lookups
        for col in self.seq_attr_list:
            if hasattr(self, f'lookup_{col}'):
                lookup = getattr(self, f'lookup_{col}')
                max_lookup_idx = lookup.size(0)
                safe_ids = flat_seq_ids.clone()
                safe_ids[safe_ids >= max_lookup_idx] = 0
                safe_ids[safe_ids < 0] = 0
                attr_ids = lookup[safe_ids]

                if col in self.feat_name_to_idx:
                    emb_layer = self.dnn_embeddings[self.feat_name_to_idx[col]]
                    attr_emb = emb_layer(attr_ids).view(batch_size, -1, self.embedding_size)
                    seq_parts.append(attr_emb)
                else:
                    seq_parts.append(torch.zeros(batch_size, seq_feat.size(1), self.embedding_size, device=device))
            else:
                seq_parts.append(torch.zeros(batch_size, seq_feat.size(1), self.embedding_size, device=device))

        # 3.3 SID（序列）：仍然只进 DIN（你原逻辑）
        if self.num_sid_cols > 0:
            max_sid_lookup = self.sid_lookup.size(0)
            safe_ids = flat_seq_ids.clone()
            safe_ids[safe_ids >= max_sid_lookup] = 0
            safe_ids[safe_ids < 0] = 0

            sids_all = self.sid_lookup[safe_ids]  # [Total, C]
            for i in range(self.num_sid_cols):
                emb_layer = self.sid_embeddings[i]
                sid_emb = emb_layer(sids_all[:, i]).view(batch_size, -1, self.embedding_size)
                seq_parts.append(sid_emb)

        # 3.4 PID（序列）
        if getattr(self, 'use_pid', False):
            pid_emb, loss = self.get_pid_embedding(seq_feat)
            seq_parts.append(pid_emb)
            total_aux_loss = total_aux_loss + loss

        seq_item_emb = torch.cat(seq_parts, dim=-1)  # [B, L, item_total_dim]
        din_output = self.attention(seq_item_emb, target_item_emb, seq_mask)  # [B, item_total_dim]

        # --- 4) DCN/DNN 输入：dnn_flat + din + sid_extra + pid_target ---
        doc_list = [dnn_embs_flat, din_output]
        if self.sid_dim > 0:
            doc_list.append(target_sid_vec)
        if target_pid_emb is not None:
            doc_list.append(target_pid_emb)

        total_emb = torch.cat(doc_list, dim=1)  # [B, total_input_dim]
        stack = torch.cat([self.cross_net(total_emb), self.dnn(total_emb)], dim=1)
        pred = torch.sigmoid(self.linear(stack))

        if self.training:
            return pred, total_aux_loss
        else:
            return pred