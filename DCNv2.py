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
        self.fc1 = nn.Linear(hidden_size * 4, hidden_size)
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
            hist_emb - target_emb_expanded,
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
            feat_name = self.feature_names[i]
            # [Fix] Set 205 (Item ID) to sparse to reduce memory usage during backward
            if feat_name == '205':
                print(f"[DCNv2] Setting sparse=True for feature {feat_name} (Size: {size})")
                self.dnn_embeddings.append(nn.Embedding(size + 1, self.embedding_size, sparse=True))
            else:
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
        
        # 4. Item Dimension
        self.item_total_dim = (1 + len(self.seq_attr_list) + self.num_sid_cols) * self.embedding_size
        
        # 5. Attention
        self.attention = AttentionLayer(self.item_total_dim)
        
        # 6. DCN Input Dim
        self.num_dnn_fields = len(self.field_dims)
        self.total_input_dim = (self.num_dnn_fields * self.embedding_size) + self.item_total_dim
        
        # 7. Network
        self.cross_net = CrossNet(self.total_input_dim, depth=config.cross_depth)
        self.dnn = DNN(self.total_input_dim, config.mlp_hidden_units, dropout=config.dropout)
        
        final_dim = self.total_input_dim + self.dnn.output_dim
        self.linear = nn.Linear(final_dim, 1)

    def forward(self, dnn_feat, seq_feat, seq_mask):
        batch_size = dnn_feat.shape[0]
        
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
             target_parts.append(torch.zeros(batch_size, self.embedding_size, device=dnn_feat.device))
            
        for col in self.seq_attr_list:
            idx = self.feat_name_to_idx.get(col)
            if idx is not None:
                target_parts.append(dnn_embs_stack[:, idx, :])
            else:
                target_parts.append(torch.zeros(batch_size, self.embedding_size, device=dnn_feat.device))

        for i in range(self.num_sid_cols):
            sid_name = f'sid_{i}'
            idx = self.feat_name_to_idx.get(sid_name)
            if idx is not None:
                target_parts.append(dnn_embs_stack[:, idx, :]) 
            else:
                target_parts.append(torch.zeros(batch_size, self.embedding_size, device=dnn_feat.device))

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
            seq_parts.append(torch.zeros(batch_size, seq_feat.size(1), self.embedding_size, device=dnn_feat.device))
        
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
                    seq_parts.append(torch.zeros(batch_size, seq_feat.size(1), self.embedding_size, device=dnn_feat.device))
            else:
                seq_parts.append(torch.zeros(batch_size, seq_feat.size(1), self.embedding_size, device=dnn_feat.device))

        # 3.3 SID
        if hasattr(self, 'sid_lookup'):
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
                     seq_parts.append(torch.zeros(batch_size, seq_feat.size(1), self.embedding_size, device=dnn_feat.device))

        seq_item_emb = torch.cat(seq_parts, dim=-1)
        
        # --- 4. Attention & Prediction ---
        din_output = self.attention(seq_item_emb, target_item_emb, seq_mask)
        total_emb = torch.cat([dnn_embs_flat, din_output], dim=1)
        
        stack = torch.cat([self.cross_net(total_emb), self.dnn(total_emb)], dim=1)
        return torch.sigmoid(self.linear(stack))