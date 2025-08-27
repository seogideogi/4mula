### dataset.py

import os
import pandas as pd
import numpy as np
import torch
import networkx as nx
from typing import Optional, Callable
from torch_geometric.data import Data, InMemoryDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from networkx.algorithms import community
import warnings
warnings.filterwarnings('ignore')

class AMLtoGraph(InMemoryDataset):
    def __init__(self, root: str, edge_window_size: int = 10,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.edge_window_size = edge_window_size
        super().__init__(root, transform, pre_transform)
        print("AMLtoGraph initialized!")
        self.data_train, _, self.data_val, _, self.data_test, _ = torch.load(self.processed_paths[0])
        print("Processed data loaded!")

    @property
    def raw_file_names(self): return 'hf_trns_tran_new.csv'
    
    @property
    def processed_file_names(self):
        return 'data.pt'
    
    def detect_communities(self, df):
        print("커뮤니티 탐지 (Train 기반)")
        G = nx.Graph()
        for _, row in df.iterrows():
            G.add_edge(row['wd_fc_ac'], row['dps_fc_ac'], weight=row['tran_amt'])

        communities = community.louvain_communities(G, weight='weight')
        community_map = {}
        for idx, nodes in enumerate(communities):
            for node in nodes:
                community_map[node] = idx

        return community_map

    def add_degree_feature(self, df):
        print("Degree 계산 (Train 기반)")
        G = nx.Graph()
        G.add_edges_from(zip(df['wd_fc_ac'], df['dps_fc_ac']))
        degree_dict = dict(G.degree())
        
        return degree_dict

    def preprocess(self, df):
        print("Preprocessing 시작")
        df.fillna(0, inplace=True)
        df['ff_sp_ai'] = df['ff_sp_ai'].apply(lambda x: 1 if x == 'SP' else 0)

        df["tran_dt_raw"] = pd.to_datetime(df["tran_dt"], format="%Y%m%d")

        train_df = df[(df['tran_dt_raw'].dt.year < 2023) |
                      ((df['tran_dt_raw'].dt.year == 2023) & (df['tran_dt_raw'].dt.month <= 10))].copy()
        val_df = df[(df['tran_dt_raw'].dt.year == 2023) & (df['tran_dt_raw'].dt.month == 11)].copy()
        test_df = df[(df['tran_dt_raw'].dt.year == 2023) & (df['tran_dt_raw'].dt.month > 11)].copy()

        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # 커뮤니티/degree 계산은 train 기준으로만 수행
        community_map = self.detect_communities(train_df)
        degree_map = self.add_degree_feature(train_df)
        
        for split_df in [train_df, val_df, test_df]:
            split_df['community'] = split_df['dps_fc_ac'].map(community_map)
            split_df['community'] = split_df['community'].fillna(0).astype(int)
            
            split_df['degree_dps'] = split_df['dps_fc_ac'].map(degree_map).fillna(0)
            split_df.loc[:, 'tran_dt'] = split_df['tran_dt_raw'].values.astype(np.int64) // 10**9

        scaler = MinMaxScaler()
        train_df.loc[:, 'tran_dt'] = scaler.fit_transform(train_df[['tran_dt']])
        val_df.loc[:, 'tran_dt'] = scaler.transform(val_df[['tran_dt']])
        test_df.loc[:, 'tran_dt'] = scaler.transform(test_df[['tran_dt']])

        return train_df, val_df, test_df

    def sample_data(self, df, verbose: bool = False):
        print("금액대별 1% 샘플링 (정상 중심 + 사기 오버샘플링) 시작")

        bins = [0, 1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, np.inf]
        labels = ['<1K', '1K-10K', '10K-50K', '50K-100K', '100K-500K', '500K-1M', '>1M']
        df['amt_bin'] = pd.cut(df['tran_amt'], bins=bins, labels=labels, include_lowest=True)

        all_samples = []

        for bin_label in labels:
            bin_df = df[df['amt_bin'] == bin_label]
            normal = bin_df[bin_df['ff_sp_ai'] == 0]
            fraud = bin_df[bin_df['ff_sp_ai'] == 1]

            #정상 거래 샘플링 (1%)
            n_normal_sample = int(len(normal) * 0.01)
            if n_normal_sample > 0:
                normal_sampled = normal.sample(n=n_normal_sample, random_state=42)
            else:
                normal_sampled = normal.iloc[0:0]

            #사기 거래 복원 샘플링 → 정상과 개수 맞춤
            if len(normal_sampled) > 0 and len(fraud) > 0:
                fraud_sampled = resample(
                    fraud,
                    replace=True,
                    n_samples=len(normal_sampled),
                    random_state=42
                )
            else:
                fraud_sampled = fraud.iloc[0:0]

            bin_sampled = pd.concat([normal_sampled, fraud_sampled], ignore_index=True)
            all_samples.append(bin_sampled)

            print(f"[{bin_label}] 정상 {len(normal_sampled)} / 사기 {len(fraud_sampled)}")

        df_sampled = pd.concat(all_samples, ignore_index=True).sample(frac=1, random_state=42)

        if verbose:
            def print_graph_stats(df_part, label=""):
                G = nx.Graph()
                for _, row in df_part.iterrows():
                    G.add_edge(row['wd_fc_ac'], row['dps_fc_ac'], weight=row['tran_amt'])
                print(f"[{label}] 노드 수: {G.number_of_nodes()}, 엣지 수: {G.number_of_edges()}")

            print_graph_stats(df, "샘플링 전 전체 데이터")
            print_graph_stats(df_sampled, "샘플링 후 데이터")

        return df_sampled

    def create_account_mapping_fast(self, df):
        """벡터화 연산으로 빠른 계좌 특성 생성"""
        print("계좌 매핑 및 노드 특성 생성 (최적화)")
        
        # 모든 계좌 ID 수집
        all_accounts = pd.concat([df['wd_fc_ac'], df['dps_fc_ac']]).unique()
        account_to_idx = {acc: idx for idx, acc in enumerate(all_accounts)}
        
        # 벡터화 집계 - 출금계좌별
        wd_agg = df.groupby('wd_fc_ac').agg({
            'tran_amt': ['count', 'mean', 'sum'],
            'tran_dt': 'max',
            'community': 'first',
            'degree_dps': 'first'
        }).fillna(0)
        wd_agg.columns = ['wd_count', 'wd_avg_amt', 'wd_total_amt', 'wd_last_dt', 'wd_community', 'wd_degree']
        
        # 벡터화 집계 - 입금계좌별  
        dps_agg = df.groupby('dps_fc_ac').agg({
            'tran_amt': ['count', 'mean', 'sum'],
            'tran_dt': 'max',
            'community': 'first',
            'degree_dps': 'first'
        }).fillna(0)
        dps_agg.columns = ['dps_count', 'dps_avg_amt', 'dps_total_amt', 'dps_last_dt', 'dps_community', 'dps_degree']
        
        # 계좌별 특성 벡터 생성
        account_features = []
        for acc in all_accounts:
            # 출금 정보
            if acc in wd_agg.index:
                wd_info = wd_agg.loc[acc].values
            else:
                wd_info = [0, 0, 0, 0, 0, 0]
                
            # 입금 정보  
            if acc in dps_agg.index:
                dps_info = dps_agg.loc[acc].values
            else:
                dps_info = [0, 0, 0, 0, 0, 0]
            
            # 전체 거래 수
            total_txns = wd_info[0] + dps_info[0]
            
            features = [
                total_txns,        # 총 거래 수
                wd_info[0],        # 출금 거래 수
                dps_info[0],       # 입금 거래 수
                wd_info[1],        # 출금 평균 금액
                dps_info[1],       # 입금 평균 금액
                wd_info[2],        # 출금 총 금액
                dps_info[2],       # 입금 총 금액
                max(wd_info[3], dps_info[3]),  # 최근 거래일
                max(wd_info[4], dps_info[4]),  # 커뮤니티 (더 큰 값)
                max(wd_info[5], dps_info[5])   # degree (더 큰 값)
            ]
            
            account_features.append(features)
        
        account_features = np.array(account_features, dtype=np.float32)
        account_features = np.nan_to_num(account_features, 0)
        
        print(f"생성된 노드 수: {len(all_accounts)}, 노드 특성 차원: {account_features.shape[1]}")
        
        return account_to_idx, torch.tensor(account_features, dtype=torch.float)

    def get_edge_data(self, df, account_to_idx):
        """엣지 인덱스, 엣지 특성, 엣지 라벨 생성"""
        print("엣지 데이터 생성")
        
        # 벡터화 매핑 - pandas map이 list comprehension보다 빠름
        src_nodes = df['wd_fc_ac'].map(account_to_idx).values
        dst_nodes = df['dps_fc_ac'].map(account_to_idx).values
        
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        
        # 엣지 특성 - 필수 컬럼만 사용해서 메모리 절약
        edge_feat_cols = [
            'tran_amt', 'tran_dt', 'tran_tmrg',
            'dps_fc_ac_fnd_amt', 'dps_fc_ac_fnd_cnt', 
            'wd_fc_ac_fnd_amt', 'wd_fc_ac_fnd_cnt'  # 일부 컬럼 제거
        ]
        
        edge_attr = torch.tensor(df[edge_feat_cols].values, dtype=torch.float)
        edge_labels = torch.tensor(df['ff_sp_ai'].values, dtype=torch.float)
        
        print(f"엣지 수: {edge_index.shape[1]}, 엣지 특성 차원: {edge_attr.shape[1]}")
        
        return edge_index, edge_attr, edge_labels

    def process(self):
        print("데이터 처리 시작")
        print(f"CSV 로딩 경로: {self.raw_paths[0]}")
        df = pd.read_csv(self.raw_paths[0], low_memory=False, dtype={'ff_sp_ai': str})
        train_df, val_df, test_df = self.preprocess(df)

        datasets = {
            'train': self.sample_data(train_df),
            'val': self.sample_data(val_df),
            'test': test_df  # 테스트 샘플링 제외
        }

        print("그래프 변환 시작")
        processed_data = {}
        
        # Train 기준으로 계좌 매핑 생성 (최적화된 버전)
        train_account_to_idx, train_node_features = self.create_account_mapping_fast(datasets['train'])
        
        for split_name, split_df in datasets.items():
            print(f"\n{split_name} 데이터 처리 중...")
            
            if split_name == 'train':
                account_to_idx = train_account_to_idx
                node_features = train_node_features
            else:
                # 빠른 필터링 - isin 사용
                valid_mask = (split_df['wd_fc_ac'].isin(train_account_to_idx.keys()) & 
                             split_df['dps_fc_ac'].isin(train_account_to_idx.keys()))
                split_df = split_df[valid_mask].reset_index(drop=True)
                
                account_to_idx = train_account_to_idx
                node_features = train_node_features
                
                print(f"필터링 후 {split_name} 거래 수: {len(split_df)}")
            
            edge_index, edge_attr, edge_labels = self.get_edge_data(split_df, account_to_idx)
            
            # PyG Data 객체 생성
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=edge_labels
            )
            
            processed_data[split_name] = data
        
        # 저장
        torch.save((processed_data['train'], torch.tensor([]),
                   processed_data['val'], torch.tensor([]),
                   processed_data['test'], torch.tensor([])), self.processed_paths[0])
        print("데이터 저장 완료!")

    def create_graph_data(self, df):
        """Deprecated"""
        pass
	
	
	
#### model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Linear
import warnings
warnings.filterwarnings('ignore')

class EdgeGATOptimized(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_channels=64, heads=4):  # hidden 축소
        super().__init__()
        
        # 노드 특성을 hidden dimension으로 변환
        self.node_in = Linear(node_feat_dim, hidden_channels)
        
        # GAT layer 1: multi-head attention
        self.gat1 = GATv2Conv(
            hidden_channels, hidden_channels, 
            heads=heads, dropout=0.2,  # dropout 감소
            edge_dim=edge_feat_dim, 
            concat=True
        )
        
        # GAT layer 2: single-head output  
        self.gat2 = GATv2Conv(
            hidden_channels * heads, hidden_channels,
            heads=1, dropout=0.2,
            edge_dim=edge_feat_dim,
            concat=False
        )
        
        # Batch normalization
        self.norm1 = nn.BatchNorm1d(hidden_channels * heads)
        self.norm2 = nn.BatchNorm1d(hidden_channels)
        
        # 간단한 MLP head (2층으로 축소)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2 + edge_feat_dim, hidden_channels),
            nn.ReLU(inplace=True),  # inplace 연산으로 메모리 절약
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1)
        )
        
    def forward(self, x, edge_index, edge_attr):
        # 노드 특성 초기 변환
        x = F.relu(self.node_in(x), inplace=True)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # GAT layer 1 (residual connection 제거로 속도 향상)
        h1 = self.gat1(x, edge_index, edge_attr)
        h1 = F.relu(self.norm1(h1), inplace=True)
        h1 = F.dropout(h1, p=0.2, training=self.training)
        
        # GAT layer 2 
        h2 = self.gat2(h1, edge_index, edge_attr)
        h2 = F.relu(self.norm2(h2), inplace=True)
        
        # 엣지별 스코어링 (최적화된 indexing)
        src_nodes, dst_nodes = edge_index
        h_src = h2[src_nodes]
        h_dst = h2[dst_nodes]
        
        # concat 한 번에 처리
        edge_features = torch.cat([h_src, h_dst, edge_attr], dim=1)
        
        # MLP로 최종 사기 확률 로짓 계산
        edge_logits = self.edge_mlp(edge_features).squeeze(-1)
        
        return edge_logits


class EdgeGATLightweight(nn.Module):
    """더 가벼운 버전 - 성능 vs 속도 트레이드오프"""
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_channels=32, heads=2):
        super().__init__()
        
        self.node_in = Linear(node_feat_dim, hidden_channels)
        
        # Single GAT layer
        self.gat = GATv2Conv(
            hidden_channels, hidden_channels, 
            heads=heads, dropout=0.1,
            edge_dim=edge_feat_dim, 
            concat=True
        )
        
        self.norm = nn.BatchNorm1d(hidden_channels * heads)
        
        # 매우 간단한 MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels * heads * 2 + edge_feat_dim, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1)
        )
        
    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.node_in(x), inplace=True)
        
        # Single GAT layer
        h = self.gat(x, edge_index, edge_attr)
        h = F.relu(self.norm(h), inplace=True)
        
        # Edge scoring
        src_nodes, dst_nodes = edge_index
        edge_features = torch.cat([h[src_nodes], h[dst_nodes], edge_attr], dim=1)
        
        return self.edge_mlp(edge_features).squeeze(-1)


# 호환성 유지
EdgeGAT = EdgeGATOptimized
GATImproved = EdgeGATOptimized


## train.py
import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from model import EdgeGAT
from dataset import AMLtoGraph
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.95, gamma=1, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# --------- Top-K 평가 함수 ----------
def precision_at_k(y_true, y_pred_proba, k):
    """Precision@K 계산"""
    if len(y_true) < k:
        k = len(y_true)
    top_k_idx = np.argsort(y_pred_proba)[::-1][:k]
    return np.sum(y_true[top_k_idx]) / k

def recall_at_k(y_true, y_pred_proba, k):
    """Recall@K 계산"""
    if len(y_true) < k:
        k = len(y_true)
    top_k_idx = np.argsort(y_pred_proba)[::-1][:k]
    total_positive = np.sum(y_true)
    if total_positive == 0:
        return 0
    return np.sum(y_true[top_k_idx]) / total_positive

def threshold_at_k(y_true, y_pred_proba, k):
    """Threshold@K 계산 - 상위 K번째 샘플의 확률값 반환"""
    if len(y_true) < k:
        k = len(y_true)
    top_k_idx = np.argsort(y_pred_proba)[::-1][:k]
    threshold_index = top_k_idx[-1]  # K번째 인덱스
    threshold = y_pred_proba[threshold_index]
    return threshold

def f1_score_at_k(y_true, y_pred_proba, k):
    """F1-Score@K 계산"""
    precision_k = precision_at_k(y_true, y_pred_proba, k)
    recall_k = recall_at_k(y_true, y_pred_proba, k)
    
    if precision_k + recall_k == 0:
        return 0
    
    f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
    return f1_k

# --------- 하이퍼파라미터 설정 ----------
batch_size = 1024
lr = 0.01
epoch_val = 30
patience = 3

k = 150

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Device: {device}")

# --------- 데이터 로딩 ----------
dataset = AMLtoGraph("./lej_dataset_002")
train_loader = DataLoader([dataset.data_train], batch_size=batch_size, shuffle=True)
val_loader = DataLoader([dataset.data_val], batch_size=batch_size)

# --------- 모델 및 Optimizer ----------
model = EdgeGAT(
    node_feat_dim=dataset.data_train.x.shape[1],
    edge_feat_dim=dataset.data_train.edge_attr.shape[1],
    hidden_channels=128,
    heads=4
).to(device)

print("\n 모델 학습 시작")
loss_fn = FocalLoss(alpha=0.95, gamma=1, reduction='mean').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

best_f1 = 0
patience_counter = 0
model_saved = False  # 모델 저장 여부 추적

# --------- 학습 루프 ----------
for epoch in range(epoch_val):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # --------- 검증 ----------
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr)
            preds.extend(torch.sigmoid(out).cpu().numpy())
            labels.extend(data.y.cpu().numpy())

    preds = np.array(preds)
    labels = np.array(labels)
    bin_preds = (preds > 0.5).astype(int)

    # 기본 이진 분류 지표
    f1 = f1_score(labels, bin_preds)
    precision = precision_score(labels, bin_preds)
    recall = recall_score(labels, bin_preds)
    roc = roc_auc_score(labels, preds)

    # Top-K 지표들
    precision_k = precision_at_k(labels, preds, k)
    recall_k = recall_at_k(labels, preds, k)
    threshold_k = threshold_at_k(labels, preds, k)
    f1_k = f1_score_at_k(labels, preds, k)

    scheduler.step(f1)

    print(f"Epoch {epoch:02d} | Loss: {total_loss:.4f}")
    print(f"bnry values - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc:.4f}")
    print(f"topK values - Precision@{k}: {precision_k:.4f}, Recall@{k}: {recall_k:.4f}, F1@{k}: {f1_k:.4f}, Threshold@{k}: {threshold_k:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        patience_counter = 0
        torch.save(model.state_dict(), "best_edge_model.pth")
        np.save("val_preds.npy", preds)
        np.save("val_labels.npy", labels)
        model_saved = True  # 저장 성공 표시

        print("Best model and predictions saved.")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

print(f"\n학습 완료! Best F1: {best_f1:.4f}")

# ===== 테스트 평가 =====
print("\n=== 테스트 평가 시작 ===")

# 모델 로딩 - 안전하게 처리
if model_saved and os.path.exists("best_edge_model.pth"):
    model.load_state_dict(torch.load("best_edge_model.pth"))
    print("저장된 최고 모델 로딩 완료")
else:
    print("저장된 모델이 없어서 현재 모델로 테스트합니다")

model.eval()

test_loader = DataLoader([dataset.data_test], batch_size=batch_size)
test_preds, test_labels = [], []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)
        test_preds.extend(torch.sigmoid(out).cpu().numpy())
        test_labels.extend(data.y.cpu().numpy())

test_preds = np.array(test_preds)
test_labels = np.array(test_labels)

# 테스트 메트릭 계산
test_bin_preds = (test_preds > 0.5).astype(int)

test_f1 = f1_score(test_labels, test_bin_preds)
test_precision = precision_score(test_labels, test_bin_preds)
test_recall = recall_score(test_labels, test_bin_preds)
test_roc = roc_auc_score(test_labels, test_preds)

test_precision_k = precision_at_k(test_labels, test_preds, k)
test_recall_k = recall_at_k(test_labels, test_preds, k)
test_threshold_k = threshold_at_k(test_labels, test_preds, k)
test_f1_k = f1_score_at_k(test_labels, test_preds, k)

print(f"\n=== 최종 테스트 결과 ===")
print(f"Test data info:")
print(f"  Edges: {len(test_labels)}")
print(f"  Fraud ratio: {test_labels.mean():.4f}")
print(f"Test Binary - Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}, ROC-AUC: {test_roc:.4f}")
print(f"Test Top-{k} - Precision@{k}: {test_precision_k:.4f}, Recall@{k}: {test_recall_k:.4f}, F1@{k}: {test_f1_k:.4f}, Threshold@{k}: {test_threshold_k:.4f}")

# 최종 결과 저장
np.save("test_preds.npy", test_preds)
np.save("test_labels.npy", test_labels)
print("\n테스트 예측 결과 저장 완료!")
