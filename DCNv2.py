import torch
from torch import nn

class DNN(nn.Module):
    """鏍囧噯 MLP"""
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
            #hist_emb - target_emb_expanded,
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
        # 按照 feature_names 的顺序初始化
        for i, size in enumerate(self.field_dims):
            self.dnn_embeddings.append(nn.Embedding(size + 1, self.embedding_size))
        
        # 2. 注册 Attribute 查找表
        if hasattr(config, 'attr_lookups'):
            for key, tensor in config.attr_lookups.items():
                self.register_buffer(f'lookup_{key}', tensor)

        # 3. 注册 SID 查找表
        self.num_sid_cols = 0 
        if hasattr(config, 'sid_lookup') and config.sid_lookup is not None:
            self.register_buffer('sid_lookup', torch.from_numpy(config.sid_lookup))
            self.num_sid_cols = config.sid_lookup.shape[1]
            print(f"[DCNv2] Enabled SID with {self.num_sid_cols} columns.")

        self.seq_attr_list = ['206', '213', '214']
        
        # === 4. PID 初始化 (提前到这里，先计算维度) ===
        self.use_pid = getattr(config, 'use_pid', False)
        # PID 会增加多少个 Embedding 块? (如果是加权平均合并成一个向量，那就是 1)
        self.num_pid_blocks = 0 
        
        if self.use_pid:
            print("[DCNv2] Enabling PID...")
            # 注册 Buffer
            if hasattr(config, 'pid_lookup') and config.pid_lookup is not None:
                self.register_buffer('pid_lookup', torch.from_numpy(config.pid_lookup))
                self.register_buffer('pid_sim_lookup', torch.from_numpy(config.pid_sim_lookup))
                self.num_pid_k = config.pid_lookup.shape[1]
                
                # Linear 变换层
                self.pid_linear = nn.Linear(self.num_pid_k, self.num_pid_k)
                nn.init.xavier_uniform_(self.pid_linear.weight)
                
                # 记录 PID 增加的维度块数
                self.num_pid_blocks = 1
            else:
                raise ValueError("use_pid is True but pid_lookup is missing in config")

        # 5. Item Dimension (重新计算总维度)
        # ID本身(1) + 属性(len) + SID(cols) + PID(1)
        self.item_total_dim = (1 + len(self.seq_attr_list) + self.num_sid_cols + self.num_pid_blocks) * self.embedding_size
        
        print(f"[DCNv2] Item Total Dim per element: {self.item_total_dim} (PID blocks: {self.num_pid_blocks})")
        
        # 6. Attention (现在使用正确的维度初始化)
        self.attention = AttentionLayer(self.item_total_dim)
        
        # 7. DCN Input Dim
        # 原始 DNN 特征的总维度 (Fields * EmbSize)
        self.num_dnn_fields = len(self.field_dims)
        self.dnn_input_dim = self.num_dnn_fields * self.embedding_size
        
        # 加上 Item Total Dim (这里包含了 Target Item 的相关 Embedding 聚合结果)
        # 注意: 之前的逻辑是把 Target Item (Query) 单独拿出来做 Cross 吗？
        # 通常 DCN 的输入是：[所有 DNN ID 特征的 Embedding] 拼接
        
        # 如果 PID 是作为一个额外的 "Field" 加入到 CrossNet 中
        if self.use_pid:
            # PID 输出一个 [B, EmbSize] 的向量，相当于多了一个 Field
            self.pid_dim = self.embedding_size
        else:
            self.pid_dim = 0
            
        self.total_input_dim = self.dnn_input_dim + self.item_total_dim
        # 等等，之前的代码 item_total_dim 好像是给 Attention 用的？
        # 让我们理清 input:
        # CrossNet 的输入通常是：Flatten(DNN Embeddings) + Attention Output
        
        self.total_input_dim = self.dnn_input_dim + self.item_total_dim 
        # 这里 item_total_dim 是 DIN 输出的维度吗？
        # 看 forward: total_emb = torch.cat([dnn_embs_flat, din_output], dim=1)
        # din_output 的维度是 hidden_size (即 embedding_size) 还是 item_total_dim?
        # 看 AttentionLayer: output = matmul(weights, hist_emb).
        # hist_emb 的最后一维是 item_total_dim.
        # 所以 din_output 的维度是 item_total_dim.
        
        # 正确的总维度:
        self.total_input_dim = self.dnn_input_dim + self.item_total_dim
        
        # 8. Network
        self.cross_net = CrossNet(self.total_input_dim, depth=config.cross_depth)
        self.dnn = DNN(self.total_input_dim, config.mlp_hidden_units, dropout=config.dropout)
        
        final_dim = self.total_input_dim + self.dnn.output_dim
        self.linear = nn.Linear(final_dim, 1)

        # === PID 初始化 ===
        self.use_pid = getattr(config, 'use_pid', False)
        
        if self.use_pid:
            print("[DCNv2] Enabling PID...")
            # 注册 Buffer，这样它们会随模型保存到 state_dict
            if hasattr(config, 'pid_lookup') and config.pid_lookup is not None:
                self.register_buffer('pid_lookup', torch.from_numpy(config.pid_lookup))
                self.register_buffer('pid_sim_lookup', torch.from_numpy(config.pid_sim_lookup))
                self.num_pid_k = config.pid_lookup.shape[1]
                
                # Linear 变换层: Sim [k] -> Weights [k]
                # 这里使用 Linear(k, k) 允许不同位置的相似度交互，或者也可以用 Element-wise 
                self.pid_linear = nn.Linear(self.num_pid_k, self.num_pid_k)
                # 初始化为正数比较好，或者 Xavier
                nn.init.xavier_uniform_(self.pid_linear.weight)
                
                # PID Embedding 将被拼接到 Item Embedding 之后
                # Item Total Dim 增加一个 Embedding Size (因为是加权平均，维度不变)
                #self.item_total_dim += self.embedding_size
            else:
                raise ValueError("use_pid is True but pid_lookup is missing in config")
        
    def get_pid_embedding(self, item_ids_input):
        """
        计算 PID Embedding 和 Monotonicity Loss
        item_ids_input: [Batch] or [Batch, SeqLen]
        """
        # 1. 展平处理 (为了同时处理 Target Item 和 Sequence Items)
        original_shape = item_ids_input.shape
        flat_ids = item_ids_input.view(-1)
        
        # 2. 处理越界 ID (比如 0 或未登录词)
        max_idx = self.pid_lookup.size(0)
        safe_ids = flat_ids.clone()
        safe_ids[safe_ids >= max_idx] = 0 # 映射到 0 (Padding)
        
        # 3. 查找相似物品 ID 和 相似度
        # neighboring_ids: [N, k], neighboring_sims: [N, k]
        neighbor_ids = self.pid_lookup[safe_ids]
        neighbor_sims = self.pid_sim_lookup[safe_ids]
        
        # 4. 获取相似物品的 Embedding (Stop Gradient)
        # 获取 Item ID (205) 对应的 Embedding Layer
        idx_205 = self.feat_name_to_idx.get('205')
        item_emb_layer = self.dnn_embeddings[idx_205]
        
        # [N, k, EmbSize]
        neighbor_embs = item_emb_layer(neighbor_ids) 
        neighbor_embs = neighbor_embs.detach() # <--- 关键：Stop Gradient
        
        # 5. 计算权重
        # Linear -> Sigmoid
        # weights: [N, k]
        weights = torch.sigmoid(self.pid_linear(neighbor_sims))
        
        # 6. 计算单调性 Loss
        # 约束: out_{j} >= out_{j+1}  =>  out_{j+1} - out_{j} <= 0
        # 如果 out_{j+1} > out_{j}，则 relu > 0，产生 Loss
        diff = weights[:, 1:] - weights[:, :-1] 
        mono_loss = torch.sum(torch.relu(diff))
        
        # 7. 加权平均
        # [N, k, E] * [N, k, 1] -> [N, k, E] -> sum -> [N, E]
        # 添加 Mask: 如果 neighbor_id 是 0 (padding)，则不参与计算
        mask = (neighbor_ids != 0).float().unsqueeze(-1) # [N, k, 1]
        
        weighted_embs = neighbor_embs * weights.unsqueeze(-1) * mask
        pid_emb = torch.sum(weighted_embs, dim=1) # [N, E] (Sum pooling)
        # 也可以考虑 Mean pooling: pid_emb = pid_emb / (torch.sum(weights * mask.squeeze(-1), dim=1, keepdim=True) + 1e-8)
        
        # 恢复形状
        pid_emb = pid_emb.view(*original_shape, -1)
        
        # Loss 归一化 (可选，避免 batch size 影响太大)
        mono_loss = mono_loss / flat_ids.size(0)
        
        return pid_emb, mono_loss

    def forward(self, dnn_feat, seq_feat, seq_mask):
        batch_size = dnn_feat.shape[0]
        device = dnn_feat.device
        
        # 初始化辅助 Loss
        total_aux_loss = torch.tensor(0.0, device=device)

        # --- 1. Embedding DNN Features ---
        dnn_emb_list = []
        for i, emb_layer in enumerate(self.dnn_embeddings):
            dnn_emb_list.append(emb_layer(dnn_feat[:, i]))
        dnn_embs_stack = torch.stack(dnn_emb_list, dim=1) # [B, F, E]
        dnn_embs_flat = dnn_embs_stack.view(batch_size, -1)
        
        # --- 2. Construct Target Item (Query) ---
        target_parts = []
        
        idx_205 = self.feat_name_to_idx.get('205')
        if idx_205 is not None:
             target_parts.append(dnn_embs_stack[:, idx_205, :])
        else:
             target_parts.append(torch.zeros(batch_size, self.embedding_size, device=device))
            
        for col in self.seq_attr_list:
            idx = self.feat_name_to_idx.get(col)
            if idx is not None:
                target_parts.append(dnn_embs_stack[:, idx, :])
            else:
                target_parts.append(torch.zeros(batch_size, self.embedding_size, device=device))

        for i in range(self.num_sid_cols):
            sid_name = f'sid_{i}'
            idx = self.feat_name_to_idx.get(sid_name)
            if idx is not None:
                target_parts.append(dnn_embs_stack[:, idx, :]) 
            else:
                target_parts.append(torch.zeros(batch_size, self.embedding_size, device=device))

        # === Inject PID for Target Item ===
        if self.use_pid and idx_205 is not None:
             # Target Item ID 在 dnn_feat 中的位置
             target_ids = dnn_feat[:, idx_205] 
             pid_emb, loss = self.get_pid_embedding(target_ids)
             target_parts.append(pid_emb)
             total_aux_loss = total_aux_loss + loss

        target_item_emb = torch.cat(target_parts, dim=-1) 

        # --- 3. Construct Sequence Items (Keys) ---
        seq_parts = []
        
        # 3.1 Sequence ID Embedding [Shared with Target Item 205]
        if idx_205 is not None:
            # Re-use the sparse embedding
            # Note: With sparse=True, this still works fine for dense inputs like seq_feat
            seq_id_emb = self.dnn_embeddings[idx_205](seq_feat) # [B, T, E]
            seq_parts.append(seq_id_emb)
        else:
            seq_parts.append(torch.zeros(batch_size, seq_feat.size(1), self.embedding_size, device=device))
        
        # 3.2 Attributes
        flat_seq_ids = seq_feat.reshape(-1)
        for col in self.seq_attr_list:
            if hasattr(self, f'lookup_{col}'):
                lookup_table = getattr(self, f'lookup_{col}')
                max_lookup_idx = lookup_table.size(0)
                safe_ids = flat_seq_ids.clone()
                safe_ids[safe_ids >= max_lookup_idx] = 0
                attr_ids = lookup_table[safe_ids]
                
                if col in self.feat_name_to_idx:
                    emb_layer_idx = self.feat_name_to_idx[col]
                    emb_layer = self.dnn_embeddings[emb_layer_idx]
                    attr_emb = emb_layer(attr_ids).view(batch_size, -1, self.embedding_size)
                    seq_parts.append(attr_emb)
                else:
                    seq_parts.append(torch.zeros(batch_size, seq_feat.size(1), self.embedding_size, device=device))
            else:
                seq_parts.append(torch.zeros(batch_size, seq_feat.size(1), self.embedding_size, device=device))

        # 3.3 SID
        if hasattr(self, 'sid_lookup') and self.sid_lookup is not None:
            max_sid_lookup = self.sid_lookup.size(0)
            safe_ids = flat_seq_ids.clone()
            safe_ids[safe_ids >= max_sid_lookup] = 0
            
            # Use safe indices for lookup
            sids_all = self.sid_lookup[safe_ids] 
            
            for i in range(self.num_sid_cols):
                 sid_name = f'sid_{i}'
                 if sid_name in self.feat_name_to_idx:
                     emb_layer_idx = self.feat_name_to_idx[sid_name]
                     emb_layer = self.dnn_embeddings[emb_layer_idx]
                     sid_emb = emb_layer(sids_all[:, i]).view(batch_size, -1, self.embedding_size)
                     seq_parts.append(sid_emb)
                 else:
                     seq_parts.append(torch.zeros(batch_size, seq_feat.size(1), self.embedding_size, device=device))

        # === Inject PID for Sequence Items ===
        if self.use_pid:
             pid_emb, loss = self.get_pid_embedding(seq_feat)
             seq_parts.append(pid_emb)
             total_aux_loss = total_aux_loss + loss

        seq_item_emb = torch.cat(seq_parts, dim=-1)
        
        # --- 4. Attention & Prediction ---
        din_output = self.attention(seq_item_emb, target_item_emb, seq_mask)
        total_emb = torch.cat([dnn_embs_flat, din_output], dim=1)
        
        stack = torch.cat([self.cross_net(total_emb), self.dnn(total_emb)], dim=1)
        pred = torch.sigmoid(self.linear(stack))
        
        if self.training:
            return pred, total_aux_loss
        else:
            return pred
