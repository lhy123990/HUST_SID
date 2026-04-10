import argparse
import random 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, log_loss
import numpy as np
import os
import sys

# 引入�???改后�??? Model �??? Dataloader
# 假�?�你�??? model 文件名为 DCNv2.py, dataloader 文件名为 dataloaderx.py
from DCNv2 import DCNV2
from dataloaderx import TaobaoDataset 

class Config: 
    def __init__(self, args):
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.embedding_size = args.embedding_size
        
        # DCN & DNN 参数
        self.cross_depth = args.depth
        self.mlp_hidden_units = args.mlp
        self.dropout = args.dropout
        
        self.epoch = args.epoch
        self.l2_reg = args.l2_reg
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        
        # 特征参数 (�??? Dataset 动态填�???)
        self.field_dims = []      # 每个 DNN 特征�??? Vocab Size
        self.target_field_idx = -1 # Target Item �??? DNN 特征�???的索�???
        self.seq_vocab_size = 0   # 序列特征�??? Vocab Size
        self.max_len = args.max_len
        self.use_pid = False      # �??否使�?? PID
        self.pid_use_independent_emb = False
        self.pid_distill = False
        self.pid_distill_weight = 0.0

        # RankMix params
        self.rankmix_d_model = args.rankmix_d_model
        self.rankmix_layers = args.rankmix_layers
        self.rankmix_heads = args.rankmix_heads
        self.rankmix_dropout = args.rankmix_dropout
        self.rankmix_head_hidden = args.rankmix_head_hidden

def train(args):
    config = Config(args)
    
    # 1. 加载数据
    #root_path = "/var/tmp/lhy_datasets/Taobao-MM/mini_dataset" # �???需要这一级目�???
    root_path = "/data/cbn01/mid_dataset" # 直接指向 mini_dataset
    print(f"Loading data from {root_path}...")
    
    # 加载两个 Dataset (会消耗较多时间加�??? Maps)
    train_dataset = TaobaoDataset(root_path, mode='train', max_len=config.max_len, use_sid=args.use_sid, use_pid=args.use_pid)
    test_dataset = TaobaoDataset(root_path, mode='test', max_len=config.max_len, use_sid=args.use_sid, use_pid=args.use_pid)
    
    # 2.? Dataset 获取维度配置
    print("Configuring Model Dimensions...")
    
    # 计算标量特征的维�??? (field_dims)
    # feature_names 已经�??? Dataset 处理好了顺序
    config.field_dims = []
    
    # 遍历所�??? DNN 特征列名，查找�?�应�??? Map 大小
    for col in train_dataset.feature_names:
        if col in train_dataset.maps:
            # Map 大小 + 1 (留给 0 �??? Padding/Unknown)
            vocab = len(train_dataset.maps[col]) + 1
            config.field_dims.append(vocab)
        else:
            # 万一没有 map (理�?�上不应发生)，给�???默�?�大�???
            print(f"Warning: No map found for {col}, using default 100000")
            config.field_dims.append(100000)
            
    # === 新�??: 传递特征名和查找表 ===
    config.feature_names = train_dataset.feature_names
    config.attr_lookups = train_dataset.attr_lookups
    config.sid_lookup = train_dataset.sid_lookup_table
    
    # 找到 Target Item ID (205) 在特征列表中的索�???
    # DIN 需要知道哪一列是 Target Item，以此作�??? Query 注意力机�???
    target_col_name = '205' 
    try:
        config.target_field_idx = train_dataset.feature_names.index(target_col_name)
    except ValueError:
        raise ValueError(f"Target Item Column '{target_col_name}' not found in dataset features: {train_dataset.feature_names}")
        
    # 计算序列特征维度
    # 序列列名 '150_2_180'
    seq_col_name = '150_2_180'
    if seq_col_name in train_dataset.maps:
        config.seq_vocab_size = len(train_dataset.maps[seq_col_name]) + 1
    else:
        config.seq_vocab_size = 200000 # Fallback
        
    print(f"Num Fields: {len(config.field_dims)}")
    print(f"Target Index: {config.target_field_idx} ({target_col_name})")
    print(f"Seq Vocab Size: {config.seq_vocab_size}")

    # 传�? PID 数据�?? Config
    config.use_pid = args.use_pid
    if args.use_pid:
        config.pid_lookup = train_dataset.pid_lookup_table
        config.pid_sim_lookup = train_dataset.pid_sim_table
        config.pid_basis_lookup = train_dataset.pid_basis_lookup_table
        config.pid_use_independent_emb = args.pid_use_independent_emb
        config.pid_distill = args.pid_distill
        config.pid_distill_weight = args.pid_distill_weight
        if args.pid_use_independent_emb and config.pid_basis_lookup is None:
            raise ValueError("pid_use_independent_emb is enabled but pid_basis_lookup_table is missing from the dataset.")
        
    # 3. 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # 4. 初�?�化模型
    model = DCNV2(config).to(config.device)
    
    if args.warm_start_path:
        if os.path.exists(args.warm_start_path):
            print(f"Loading pretrained embeddings from {args.warm_start_path}...")
            checkpoint = torch.load(args.warm_start_path, map_location=config.device)
            model_dict = model.state_dict()
            
            # �??加载 dnn_embeddings（过滤掉改变了结构的网络其他部分�??
            pretrained_emb_dict = {
                k: v for k, v in checkpoint.items() 
                if 'dnn_embeddings' in k and k in model_dict and v.size() == model_dict[k].size()
            }
            
            if len(pretrained_emb_dict) > 0:
                model_dict.update(pretrained_emb_dict)
                model.load_state_dict(model_dict)
                print(f"Successfully warm-started {len(pretrained_emb_dict)} embedding layer tensors.")
                
                if args.freeze_emb:
                    print("Freezing warmed-up dnn_embeddings...")
                    for name, param in model.named_parameters():
                        if 'dnn_embeddings' in name:
                            param.requires_grad = False
            else:
                print("Warning: No matching dnn_embeddings found.")
        else:
            print(f"Warning: Pretrained path {args.warm_start_path} does not exist.")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config.learning_rate, 
        weight_decay=config.l2_reg
    )
    criterion = nn.BCELoss()

    best_auc = 0.0

    # 定义模型保存�??�?? (�??改这�??)
    save_dir = "/data/cbn01/checkpoints" # 使用大�?�量�??
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'best_model.pth')
    
    print("Start Training...")
    for epoch in range(config.epoch):
        model.train()

        if config.use_pid:
     

            # �?? DCNv2 在本 epoch �??一�?? PID 前向时打印一�?? case
            model.has_printed_pid_debug = False

        total_loss = 0
        steps = 0
        
        # 数据解包: dnn_feat, seq_feat, seq_mask, label
        for dnn_feat, seq_feat, seq_mask, label in train_loader:
            dnn_feat = dnn_feat.to(config.device)
            seq_feat = seq_feat.to(config.device)
            seq_mask = seq_mask.to(config.device)
            label = label.to(config.device).float().unsqueeze(1) # [B, 1]

            optimizer.zero_grad()
            
            # Forward 接收 Loss
            if config.use_pid:
                pred, aux_loss = model(dnn_feat, seq_feat, seq_mask)
                loss = criterion(pred, label) + aux_loss # 直接相加，也�??以加权重 alpha
            else:
                # 兼�?�旧逻辑
                result = model(dnn_feat, seq_feat, seq_mask)
                if isinstance(result, tuple):
                    pred, _ = result
                else:
                    pred = result
                loss = criterion(pred, label)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
            if steps % 100 == 0:
                print(f"Epoch {epoch+1} | Step {steps} | Loss: {loss.item():.4f}", end='\r')

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} Done. Avg Loss: {avg_loss:.4f}")

        # Validation
        auc, ll = evaluate(model, test_loader, config.device)
        print(f"test AUC: {auc:.4f}, LogLoss: {ll:.4f}")

        if auc > best_auc:
            best_auc = auc
            # �??改保存路�??
            torch.save(model.state_dict(), save_path) 
            print(f"Best model saved to {save_path}")

def evaluate(model, loader, device):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for dnn_feat, seq_feat, seq_mask, label in loader:
            dnn_feat = dnn_feat.to(device)
            seq_feat = seq_feat.to(device)
            seq_mask = seq_mask.to(device)
            
            pred = model(dnn_feat, seq_feat, seq_mask)
            
            if isinstance(pred, tuple):
                pred = pred[0]
                
            preds.extend(pred.cpu().numpy().flatten())
            labels.extend(label.numpy().flatten())

    return roc_auc_score(labels, preds), log_loss(labels, preds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=5120)
    parser.add_argument('--embedding_size', type=int, default=16)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--mlp', type=int, nargs='+', default=[512, 512,512])
    parser.add_argument('--dropout', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=4)
    parser.add_argument('--l2_reg', type=float, default=1e-5) 
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=200) # 序列最大长�??
    parser.add_argument('--num_workers', type=int, default=0,
                    help='DataLoader workers (set 0 to disable multiprocessing)')
    parser.add_argument('--pin_memory', action='store_true', default=False,
                    help='Enable DataLoader pin_memory')
    parser.add_argument('--use_sid', action='store_true', default=True, 
                    help='Use Semantic ID (default: enabled)')
    parser.add_argument('--no_use_sid', action='store_false', dest='use_sid',
                    help='Disable Semantic ID (override default enabled)')
    parser.add_argument('--use_pid', action='store_true', default=False)
    parser.add_argument('--pid_use_independent_emb', action='store_true', default=False,
                        help='Use a separate embedding table for PID basis ids instead of reusing the 205 item embedding')
    parser.add_argument('--pid_distill', action='store_true', default=False,
                        help='Add cosine distillation loss between basis PID embeddings and their 205 item embeddings')
    parser.add_argument('--pid_distill_weight', type=float, default=0.01,
                        help='Weight for the PID basis distillation loss')

    # RankMix hyper-parameters
    parser.add_argument('--rankmix_d_model', type=int, default=64,
                        help='RankMix token/channel dim D')
    parser.add_argument('--rankmix_layers', type=int, default=2,
                        help='Number of RankMix encoder layers')
    parser.add_argument('--rankmix_heads', type=int, default=4,
                        help='Attention heads in RankMix encoder')
    parser.add_argument('--rankmix_dropout', type=float, default=0.01,
                        help='Dropout used in RankMix encoder/head')
    parser.add_argument('--rankmix_head_hidden', type=int, default=256,
                        help='Hidden size of RankMix CTR task head')
    
    # === 新�?�控制热�??的参�?? ===
    parser.add_argument('--warm_start_path', type=str, default=False, 
                    help='/data/cbn01/checkpoints/cold_model.pth   False')
    parser.add_argument('--freeze_emb', action='store_true', default=False, 
                    help='Freeze the warm-started DNN embeddings')
                    
    args = parser.parse_args()
    
    # 互斥检�??
    if args.use_sid and args.use_pid:
        print("Warning: Both SID and PID enabled. Combining them or prioritizing PID depends on implementation details.")
        # 这里建�??强制互斥
        args.use_sid = False 
        print("Disabling SID to use PID.")

    if args.pid_distill and not args.use_pid:
        print("Warning: pid_distill is enabled but use_pid is False. Disabling pid_distill.")
        args.pid_distill = False

    seed = 2026
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False   
    os.environ['PYTHONHASHSEED'] = str(seed)
    train(args)

