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
            #hist_emb * target_emb_expanded
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

        # === 3. SID 初始化 (关键修复) ===
        self.num_sid_cols = 0 
        self.sid_embeddings = nn.ModuleList() # [Fix] 独立创建 Embedding
        
        if hasattr(config, 'sid_lookup') and config.sid_lookup is not None:
            self.register_buffer('sid_lookup', torch.from_numpy(config.sid_lookup))
            self.num_sid_cols = config.sid_lookup.shape[1]
            
            # 自动计算 SID 词表大小 (也就是聚类簇的数量)
            max_sid_val = self.sid_lookup.max().item()
            sid_vocab_size = int(max_sid_val + 10) # +10 防止边界溢出
            print(f"[DCNv2] Enabled SID with {self.num_sid_cols} columns. Vocab Size: {sid_vocab_size}")
            
            # 为每一列 SID 创建独立的 Embedding 层
            for _ in range(self.num_sid_cols):
                self.sid_embeddings.append(nn.Embedding(sid_vocab_size, self.embedding_size))

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
                
                # 记录 PID 增加的维度块数 (拼接到 Item Emb 中)
                self.num_pid_blocks = 1
            else:
                raise ValueError("use_pid is True but pid_lookup is missing in config")

        # 5. Item Dimension (重新计算总维度)
        # ID本身(1) + 属性(len) + SID(cols) + PID(1)
        self.item_total_dim = (1 + len(self.seq_attr_list) + self.num_sid_cols + getattr(self, 'num_pid_blocks', 0)) * self.embedding_size
        
        print(f"[DCNv2] Item Total Dim per element: {self.item_total_dim} (PID blocks: {self.num_pid_blocks})")
        
        # 6. Attention
        self.attention = AttentionLayer(self.item_total_dim)
        
        # 7. DCN Input Dim
        # 原始 DNN 特征的总维度 (Fields * EmbSize)
        self.num_dnn_fields = len(self.field_dims)
        self.dnn_input_dim = self.num_dnn_fields * self.embedding_size
        
        # 如果 PID 是作为一个额外的 "Field" 加入到 CrossNet 中
        if self.use_pid:
            # PID 输出一个 [B, EmbSize] 的向量，拼接到 dnn_embs_flat 中
            self.pid_dim = self.embedding_size
        else:
            self.pid_dim = 0
            
        # [关键修正] 总输入维度 = DNN特征 + 序列特征(Attention输出) + PID特征(Target Item的PID)
        self.total_input_dim = self.dnn_input_dim + self.item_total_dim + self.pid_dim
        
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
        # [Fix] 1. 确保输入是 Long 类型，防止浮点数索引隐患
        item_ids_input = item_ids_input.long()
        
        original_shape = item_ids_input.shape
        flat_ids = item_ids_input.view(-1)
        
        # 1. 查找相似物品
        max_idx = self.pid_lookup.size(0)
        safe_ids = flat_ids.clone()
        safe_ids[safe_ids >= max_idx] = 0
        safe_ids[safe_ids < 0] = 0
        
        neighbor_ids = self.pid_lookup[safe_ids]      # [N, k]
        neighbor_sims = self.pid_sim_lookup[safe_ids] # [N, k]

        # 获取 Embedding 层 (共享 Main Item Embedding)
        idx_205 = self.feat_name_to_idx.get('205')
        item_emb_layer = self.dnn_embeddings[idx_205]

        # [Debug] 打印一次看看取出的 ID 是什么范围 (检查是否是 Raw ID)
        if not self.has_printed_neighbor_info:
            print(f"\n[PID Check]")
            print(f"  Lookup Max Size: {max_idx}")
            print(f"  Item Vocab Size: {item_emb_layer.num_embeddings}")
            print(f"  Sample Neighbor IDs: {neighbor_ids[0].tolist()}")
            print(f"  Max Neighbor ID: {neighbor_ids.max().item()}")
            if neighbor_ids.max().item() >= item_emb_layer.num_embeddings:
                print(f"  [CRITICAL] Neighbor IDs contain values larger than Vocab! Are these Raw IDs?")
            self.has_printed_neighbor_info = True

        # [N, k, EmbSize]
        neighbor_ids_safe = neighbor_ids.clone()
        # [Safety] 防止越界导致崩溃，这里做个截断保护
        neighbor_ids_safe[neighbor_ids_safe >= item_emb_layer.num_embeddings] = 0
        neighbor_ids_safe[neighbor_ids_safe < 0] = 0 
        
        neighbor_embs = item_emb_layer(neighbor_ids_safe) 
        neighbor_embs = neighbor_embs.detach() # [Correct] 梯度停止
        
        # 3. 计算权重 (Linear -> Sigmoid)
        # 你的 pid_linear 是 (k -> k)，这实际上是一个小的 MLP，允许不同位置的相似度交互，是合理的
        weights = torch.sigmoid(self.pid_linear(neighbor_sims)) # [N, k]
        
        # === [新增 DEBUG 代码开始] ===
        if not getattr(self, 'has_printed_pid_debug', False):
            print(f"\n[PID Runtime Debug]")
            print(f"  Input Simulations (neighbor_sims): \n{neighbor_sims[0].detach().cpu().numpy()}")
            print(f"  Output Weights (Sigmoid): \n{weights[0].detach().cpu().numpy()}")
            
            # 检查是否有梯度的迹象 (weights 是否全为 0.5 或者非常接近)
            w_min, w_max = weights.min().item(), weights.max().item()
            print(f"  Weights Range in batch: Min={w_min:.4f}, Max={w_max:.4f}")
            
            # 检查 Linear 层权重 (如果全0初始化或者未能更新)
            if hasattr(self, 'pid_linear'):
                print(f"  Linear Layer Weights Sample: \n{self.pid_linear.weight.data[0][:10].cpu().numpy()}...")
                print(f"  Linear Layer Grad Exists: {self.pid_linear.weight.grad is not None}")
                
            self.has_printed_pid_debug = True
        # === [新增 DEBUG 代码结束] ===

        # 4. Monotonicity Loss
        diff = weights[:, 1:] - weights[:, :-1] 
        mono_loss = torch.sum(torch.relu(diff))
        
        # 5. 加权平均
        # Mask: 0 是 Padding
        mask = (neighbor_ids != 0).float().unsqueeze(-1) # [N, k, 1]
        
        weighted_embs = neighbor_embs * weights.unsqueeze(-1) * mask
        pid_emb = torch.sum(weighted_embs, dim=1) # [N, E]
        
        pid_emb = pid_emb.view(*original_shape, -1)
        mono_loss = mono_loss / flat_ids.size(0)
        
        return pid_emb, mono_loss

    def forward(self, dnn_feat, seq_feat, seq_mask):
        batch_size = dnn_feat.shape[0]
        device = dnn_feat.device
        total_aux_loss = torch.tensor(0.0, device=device)

        # 1. Embedding DNN Features
        dnn_emb_list = []
        for i, emb_layer in enumerate(self.dnn_embeddings):
            dnn_emb_list.append(emb_layer(dnn_feat[:, i]))
        dnn_embs_stack = torch.stack(dnn_emb_list, dim=1) 
        dnn_embs_flat = dnn_embs_stack.view(batch_size, -1)
        
        # 获取 Target Item ID (205)
        idx_205 = self.feat_name_to_idx.get('205')

        # --- 2. Construct Target Item ---
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

        # === 2.3 SID (关键修复: 查表 + 独立Embedding) ===
        if self.num_sid_cols > 0:
            if idx_205 is not None:
                # 1. 拿到 Item ID
                target_ids = dnn_feat[:, idx_205].long()
                
                # 2. 边界保护
                max_len = self.sid_lookup.size(0)
                safe_ids = target_ids.clone()
                safe_ids[safe_ids >= max_len] = 0
                safe_ids[safe_ids < 0] = 0
                
                # 3. 查表得到 SID [Batch, Cols]
                target_sids = self.sid_lookup[safe_ids] 
                
                # 4. 做 Embedding
                for i in range(self.num_sid_cols):
                    # 使用 self.sid_embeddings 而不是 dnn_embeddings
                    sid_emb = self.sid_embeddings[i](target_sids[:, i]) 
                    target_parts.append(sid_emb)
            else:
                for _ in range(self.num_sid_cols):
                    target_parts.append(torch.zeros(batch_size, self.embedding_size, device=device))

        # 2.4 PID (保持不变)
        target_pid_emb = None  
        if getattr(self, 'use_pid', False) and idx_205 is not None:
             target_ids = dnn_feat[:, idx_205] 
             pid_emb, loss = self.get_pid_embedding(target_ids)
             target_parts.append(pid_emb)
             total_aux_loss = total_aux_loss + loss
             target_pid_emb = pid_emb 

        target_item_emb = torch.cat(target_parts, dim=-1) 

        # --- 3. Construct Sequence Items ---
        seq_parts = []
        
        # 3.1 & 3.2 ID & Attributes (简写，请保留你原本的完整逻辑)
        if idx_205 is not None:
            seq_parts.append(self.dnn_embeddings[idx_205](seq_feat))
        else:
            seq_parts.append(torch.zeros(batch_size, seq_feat.size(1), self.embedding_size, device=device))
            
        flat_seq_ids = seq_feat.reshape(-1)
        # ... (保留 Attribute 处理逻辑) ...
        # 为节省篇幅，假设 Attribute 部分你保留原样，确保对齐即可
        for col in self.seq_attr_list:
             if hasattr(self, f'lookup_{col}'): 
                 # 复制之前的逻辑
                 max_lookup_idx = getattr(self, f'lookup_{col}').size(0)
                 safe_ids = flat_seq_ids.clone()
                 safe_ids[safe_ids >= max_lookup_idx] = 0
                 attr_ids = getattr(self, f'lookup_{col}')[safe_ids]
                 if col in self.feat_name_to_idx:
                     emb_layer = self.dnn_embeddings[self.feat_name_to_idx[col]]
                     attr_emb = emb_layer(attr_ids).view(batch_size, -1, self.embedding_size)
                     seq_parts.append(attr_emb)
                 else:
                     seq_parts.append(torch.zeros(batch_size, seq_feat.size(1), self.embedding_size, device=device))
             else:
                 seq_parts.append(torch.zeros(batch_size, seq_feat.size(1), self.embedding_size, device=device))


        # === 3.3 SID (序列部分修复) ===
        if self.num_sid_cols > 0:
            max_sid_lookup = self.sid_lookup.size(0)
            safe_ids = flat_seq_ids.clone()
            safe_ids[safe_ids >= max_sid_lookup] = 0
            
            sids_all = self.sid_lookup[safe_ids] # [Total_Items, Cols]
            
            for i in range(self.num_sid_cols):
                 # [Fix] 使用 sid_embeddings
                 emb_layer = self.sid_embeddings[i]
                 sid_emb = emb_layer(sids_all[:, i]).view(batch_size, -1, self.embedding_size)
                 seq_parts.append(sid_emb)

        # 3.4 PID (保持不变)
        if getattr(self, 'use_pid', False):
             pid_emb, loss = self.get_pid_embedding(seq_feat)
             seq_parts.append(pid_emb)
             total_aux_loss = total_aux_loss + loss

        # ... (后续拼接和 CrossNet 逻辑保持不变)
        seq_item_emb = torch.cat(seq_parts, dim=-1)
        din_output = self.attention(seq_item_emb, target_item_emb, seq_mask)
        
        doc_list = [dnn_embs_flat, din_output]
        if target_pid_emb is not None:
             doc_list.append(target_pid_emb)
            
        total_emb = torch.cat(doc_list, dim=1)
        stack = torch.cat([self.cross_net(total_emb), self.dnn(total_emb)], dim=1)
        pred = torch.sigmoid(self.linear(stack))
        
        if self.training:
            return pred, total_aux_loss
        else:
            return pred