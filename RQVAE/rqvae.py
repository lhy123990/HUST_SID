"""
选手可参考以下流程，使用提供的 RQ-VAE 框架代码将多模态emb数据转换为Semantic Id:
1. 使用 MmEmbDataset 读取不同特征 ID 的多模态emb数据.
2. 训练 RQ-VAE 模型, 训练完成后将数据转换为Semantic Id.
3. 参照 Item Sparse 特征格式处理Semantic Id，作为新特征加入Baseline模型训练.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from sklearn.cluster import KMeans
import numpy as np

# =============================================================================
# 1. 辅助函数与层
# =============================================================================

def activation_layer(activation_name="relu", emb_dim=None):
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_name.lower() == "relu":
            activation = nn.ReLU()
        elif activation_name.lower() == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name.lower() == "none":
            activation = None
        else:
            raise NotImplementedError(
                "activation function {} is not implemented".format(activation_name)
            )
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            "activation function {} is not implemented".format(activation_name)
        )

    return activation

def kmeans(samples, num_clusters, num_iters=10):
    # samples: [B, D]
    B, dim = samples.shape
    dtype, device = samples.dtype, samples.device
    
    # 转为 numpy 进行 K-Means
    x = samples.cpu().detach().numpy()
    
    # 防止样本数小于聚类数
    if B < num_clusters:
        # 样本不足时，直接采样作为中心
        indices = np.random.choice(B, num_clusters, replace=True)
        centers = x[indices]
    else:
        # 使用 sklearn 的 KMeans
        cluster = KMeans(n_clusters=num_clusters, max_iter=num_iters, n_init='auto').fit(x)
        centers = cluster.cluster_centers_

    tensor_centers = torch.from_numpy(centers).to(device).type(dtype)
    return tensor_centers

@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    # distances: [B, K]
    # Q = exp(- dist / epsilon)
    # 为了数值稳定性，可以考虑先减去最小值 (LogSumExp trick 变种)，但这里保持原样
    Q = torch.exp(- distances / epsilon)

    B = Q.shape[0] # number of samples
    K = Q.shape[1] # number of centroids

    # Initial normalization to sum to 1
    sum_Q = Q.sum()
    Q /= sum_Q

    for _ in range(sinkhorn_iterations):
        # normalize each column: sum over K (centroids) must be 1/B
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= B

        # normalize each row: sum over B (samples) must be 1/K
        sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
        Q /= sum_of_cols
        Q /= K

    Q *= B # Columns sum to 1 => Valid assignment probability distribution
    return Q

class MLPLayers(nn.Module):
    def __init__(self, layers, dropout=0.0, activation="relu", bn=False):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Linear(input_size, output_size))

            # BN 和 Activation 通常在 Linear 之后
            if self.use_bn and idx != (len(self.layers) - 2):
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))

            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None and idx != (len(self.layers) - 2):
                mlp_modules.append(activation_func)
                
            if self.dropout > 0:
                 mlp_modules.append(nn.Dropout(p=self.dropout))

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


# =============================================================================
# 2. 核心模块: VectorQuantizer (包含 Sinkhorn)
# =============================================================================

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim,
                 beta=0.25, kmeans_init=False, kmeans_iters=10,
                 sk_epsilon=0.003, sk_iters=100):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        
        # 初始化状态标记
        self.register_buffer('initted', torch.tensor(False))
        
        if not kmeans_init:
            self.initted.fill_(True)
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted.fill_(False)
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q

    def init_emb(self, data):
        print(f"Initializing codebook with K-Means (k={self.n_e})...")
        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )
        self.embedding.weight.data.copy_(centers)
        self.initted.fill_(True) # 标记为已初始化

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, use_sk=True):
        # Flatten input: [B, D]
        latent = x.view(-1, self.e_dim)

        # 训练阶段进行 K-Means 初始化
        if self.training and not self.initted:
            self.init_emb(latent)

        # Calculate distances: x^2 + e^2 - 2xe
        # [B, 1] + [1, K] - [B, K]
        # 注意：使用 t() 转置 embedding weight
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1).unsqueeze(0) - \
            2 * torch.matmul(latent, self.embedding.weight.t())

        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            # Sinkhorn-Knopp
            d_centered = self.center_distance_for_constraint(d)
            # 需要转为 float/double 计算 exp 避免溢出
            Q = sinkhorn_algorithm(d_centered, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
                # Fallback
                indices = torch.argmin(d, dim=-1)
            else:
                indices = torch.argmax(Q, dim=-1) # Q is probability, so max

        # [B, D]
        x_q = self.embedding(indices).view(x.shape)

        # Loss calculation
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # Straight-through estimator
        x_q = x + (x_q - x).detach()
        
        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices


# =============================================================================
# 3. 核心模块: ResidualVectorQuantizer
# =============================================================================

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, n_e_list, e_dim, sk_epsilons=None, beta=0.25,
                 kmeans_init=False, kmeans_iters=100, sk_iters=100):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_iters = sk_iters
        
        if sk_epsilons is None:
            self.sk_epsilons = [None] * self.num_quantizers 
            # 注意：如果是用 None 表示不开启，确保 VectorQuantizer 内部处理了
            # 或者在这里默认设为 0.0
            self.sk_epsilons = [0.0] * self.num_quantizers
        else:
            self.sk_epsilons = sk_epsilons

        self.vq_layers = nn.ModuleList([
            VectorQuantizer(n_e, e_dim,
                            beta=self.beta,
                            kmeans_init=self.kmeans_init,
                            kmeans_iters=self.kmeans_iters,
                            sk_epsilon=sk_eps,
                            sk_iters=self.sk_iters)
            for n_e, sk_eps in zip(n_e_list, self.sk_epsilons)
        ])

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x, use_sk=True):
        all_losses = []
        all_indices = []

        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

        # Mean of losses across layers
        mean_losses = torch.stack(all_losses).mean()
        # Stack indices: [B, Num_Layers]
        all_indices = torch.stack(all_indices, dim=1)

        return x_q, mean_losses, all_indices


# =============================================================================
# 4. 主模型: RQVAE
# =============================================================================

class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=128,
                 num_emb_list=None, # e.g. [256, 128, 128]
                 e_dim=16,
                 layers=None,       # e.g. [128, 128] (hidden layers)
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons=None,
                 sk_iters=100,
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list if num_emb_list is not None else [256, 256, 256]
        self.e_dim = e_dim
        self.layers = layers if layers is not None else [128]
        
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        # Encoder: [In_Dim] -> [Hidden...] -> [E_Dim]
        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob, bn=self.bn, activation='relu')

        # Residual Quantizer
        self.rq = ResidualVectorQuantizer(
            n_e_list=self.num_emb_list, 
            e_dim=self.e_dim,
            sk_epsilons=self.sk_epsilons,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_iters=self.sk_iters
        )

        # Decoder: [E_Dim] -> [Hidden_Reverse...] -> [In_Dim]
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                 dropout=self.dropout_prob, bn=self.bn, activation='relu')

    def forward(self, x, use_sk=True):
        # x: [B, In_Dim]
        
        # 1. Encode
        z_e = self.encoder(x)
        
        # 2. Residual Quantization
        # x_q: [B, E_Dim]
        # rq_loss: scalar (mean of layers)
        # indices: [B, Num_Codebooks]
        z_q, rq_loss, indices = self.rq(z_e, use_sk=use_sk)
        
        # 3. Decode
        out = self.decoder(z_q)

        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        """只获取索引，不进行 Decoder (用于生成 Semantic ID)"""
        if self.training:
            self.eval()
            
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, use_sk=use_sk)
        return indices

    def compute_loss(self, out, quant_loss, xs=None):
        """计算总 Loss"""
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon
