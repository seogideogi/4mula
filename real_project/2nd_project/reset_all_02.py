## dataset.py (기존 코드 + 최소 수정)
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
        try:
            # 계좌 ID 안전 처리
            df['wd_fc_ac'] = df['wd_fc_ac'].astype(str).str.strip()
            df['dps_fc_ac'] = df['dps_fc_ac'].astype(str).str.strip()
            
            G = nx.Graph()
            for _, row in df.iterrows():
                wd_ac, dps_ac = row['wd_fc_ac'], row['dps_fc_ac']
                if wd_ac and dps_ac and str(wd_ac) != 'nan' and str(dps_ac) != 'nan':
                    G.add_edge(wd_ac, dps_ac, weight=float(row['tran_amt']))

            if G.number_of_nodes() == 0:
                print("WARNING 그래프에 노드가 없음, 빈 community_map 반환")
                return {}
                
            communities = community.louvain_communities(G, weight='weight')
            community_map = {}
            for idx, nodes in enumerate(communities):
                for node in nodes:
                    community_map[node] = idx

            print(f"SUCCESS 커뮤니티 {len(communities)}개 탐지, 노드 {len(community_map)}개")
            return community_map
            
        except Exception as e:
            print(f"ERROR 커뮤니티 탐지 중 오류: {e}")
            return {}

    def add_degree_feature(self, df):
        print("Degree 계산 (Train 기반)")
        G = nx.Graph()
        G.add_edges_from(zip(df['wd_fc_ac'], df['dps_fc_ac']))
        degree_dict = dict(G.degree())
        
        return degree_dict

    #노드 임베딩 생성 함수 추가
    def create_node_embeddings(self, df, community_map, degree_map):
        """계좌 단위 노드 임베딩 생성"""
        print("노드 임베딩 생성 (계좌 단위)")
        
        try:
            # 계좌 ID 문자열 안전 처리
            df['wd_fc_ac'] = df['wd_fc_ac'].astype(str).str.strip()
            df['dps_fc_ac'] = df['dps_fc_ac'].astype(str).str.strip()
            
            # 모든 계좌 수집
            all_accounts = set(df['wd_fc_ac'].unique()) | set(df['dps_fc_ac'].unique())
            # None, nan, 빈 문자열 제거
            all_accounts = {acc for acc in all_accounts if acc and str(acc).strip() and str(acc) != 'nan'}
            
            if len(all_accounts) == 0:
                raise ValueError("유효한 계좌가 없습니다!")
                
            account_to_idx = {acc: idx for idx, acc in enumerate(sorted(all_accounts))}
            print(f"총 계좌 수: {len(all_accounts)}")
            
            # 계좌별 집계 특성 생성
            node_features = []
            for account in sorted(all_accounts):
                wd_txns = df[df['wd_fc_ac'] == account]
                dps_txns = df[df['dps_fc_ac'] == account]
                
                # 안전한 집계 (빈 시리즈 처리)
                features = [
                    float(wd_txns['tran_amt'].sum()) if len(wd_txns) > 0 else 0.0,
                    float(dps_txns['tran_amt'].sum()) if len(dps_txns) > 0 else 0.0,
                    float(len(wd_txns)),
                    float(len(dps_txns)),
                    float(wd_txns['tran_tmrg'].mean()) if len(wd_txns) > 0 and not wd_txns['tran_tmrg'].isna().all() else 0.0,
                    float(community_map.get(account, 0)),
                    float(degree_map.get(account, 0)),
                ]
                node_features.append(features)
            
            # 안전한 numpy 변환
            node_features = np.array(node_features, dtype=np.float32)
            node_features = np.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            print(f"노드 특성 shape: {node_features.shape}")
            return torch.tensor(node_features, dtype=torch.float), account_to_idx
            
        except Exception as e:
            print(f"ERROR 노드 임베딩 생성 중 오류: {e}")
            print(f"계좌 샘플: {list(df['wd_fc_ac'].head())}")
            raise

    def preprocess(self, df):
        print("Preprocessing 시작")
        df.fillna(0, inplace=True)
        df['ff_sp_ai'] = df['ff_sp_ai'].apply(lambda x: 1 if x == 'SP' else 0)

        df = df[df["tran_amt"] >= 10000]
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

        return train_df, val_df, test_df, community_map, degree_map  # 🆕 추가 반환

    def sample_data(self, df, verbose: bool = False):
        print("샘플링 시작")
        def print_graph_stats(df_part, label=""):
            G = nx.Graph()
            for _, row in df_part.iterrows():
                G.add_edge(row['wd_fc_ac'], row['dps_fc_ac'], weight=row['tran_amt'])
            print(f"[{label}] 노드 수: {G.number_of_nodes()}, 엣지 수: {G.number_of_edges()}")

        normal_data = df[df['ff_sp_ai'] == 0]
        fraud_data = df[df['ff_sp_ai'] == 1]

        print(f"원래 정상 거래 수: {len(normal_data)}, 사기 거래 수: {len(fraud_data)}")

        normal_sampled = normal_data.sample(frac=0.01, random_state=42)
        fraud_sampled = resample(fraud_data, replace=True, n_samples=len(normal_sampled), random_state=42)

        df_sampled = pd.concat([normal_sampled, fraud_sampled], ignore_index=True).sample(frac=1, random_state=42)

        print(f"샘플링 후 정상 거래 수: {len(normal_sampled)}, 사기 거래 수: {len(fraud_sampled)}")

        if verbose:
            print_graph_stats(df, "샘플링 전 전체 데이터")
            print_graph_stats(df_sampled, "샘플링 후 데이터")

        return df_sampled

    #기존 get_edge_df 함수 수정 (account_to_idx 사용)
    def get_edge_df(self, df, account_to_idx):
        print("엣지 생성 시작")
        
        try:
            # 계좌 ID 안전 처리
            df['wd_fc_ac'] = df['wd_fc_ac'].astype(str).str.strip()
            df['dps_fc_ac'] = df['dps_fc_ac'].astype(str).str.strip()
            
            # account_to_idx에 없는 계좌 필터링
            valid_mask = df['wd_fc_ac'].isin(account_to_idx) & df['dps_fc_ac'].isin(account_to_idx)
            if not valid_mask.any():
                raise ValueError("유효한 엣지가 없습니다!")
                
            df_valid = df[valid_mask].copy()
            print(f"유효한 엣지: {len(df_valid)}/{len(df)}")
            
            # 계좌를 노드 인덱스로 매핑
            wd_indices = [account_to_idx[acc] for acc in df_valid['wd_fc_ac']]
            dps_indices = [account_to_idx[acc] for acc in df_valid['dps_fc_ac']]
            
            edge_index = torch.tensor([wd_indices, dps_indices], dtype=torch.long)

            edge_feat_cols = [
                'tran_amt', 'tran_dt', 'tran_tmrg',
                'dps_fc_ac_fnd_amt', 'dps_fc_ac_fnd_cnt', 'dps_fc_ac_md_amt', 'dps_fc_ac_md_cnt',
                'wd_fc_ac_fnd_amt', 'wd_fc_ac_fnd_cnt', 'wd_fc_ac_md_amt', 'wd_fc_ac_md_cnt'
            ]
            
            # 엣지 특성 안전 추출
            edge_features = []
            for col in edge_feat_cols:
                if col in df_valid.columns:
                    values = pd.to_numeric(df_valid[col], errors='coerce').fillna(0).values
                else:
                    print(f"WARNING 컬럼 '{col}' 없음, 0으로 대체")
                    values = np.zeros(len(df_valid))
                edge_features.append(values)
            
            edge_attr = torch.tensor(np.column_stack(edge_features), dtype=torch.float)
            
            print(f"엣지 수: {edge_index.shape[1]}, 엣지 특성 차원: {edge_attr.shape[1]}")
            return edge_attr, edge_index, df_valid  # 필터된 df도 반환
            
        except Exception as e:
            print(f"ERROR 엣지 생성 중 오류: {e}")
            print(f"account_to_idx 샘플: {list(account_to_idx.keys())[:5]}")
            print(f"wd_fc_ac 샘플: {df['wd_fc_ac'].head().tolist()}")
            raise

    #기존 get_node_attr 함수는 삭제하고 create_node_embeddings 사용

    def process(self):
        print("데이터 처리 시작")
        print(f"CSV 로딩 경로: {self.raw_paths[0]}")
        
        try:
            # CSV 안전 로딩
            df = pd.read_csv(self.raw_paths[0], low_memory=False, dtype={'ff_sp_ai': str}, 
                           encoding='utf-8', on_bad_lines='warn')
            print(f"SUCCESS 원본 데이터 로딩: {len(df):,}행")
            
        except UnicodeDecodeError:
            print("WARNING UTF-8 실패, CP949로 재시도...")
            df = pd.read_csv(self.raw_paths[0], low_memory=False, dtype={'ff_sp_ai': str}, 
                           encoding='cp949', on_bad_lines='warn')
            print(f"SUCCESS 원본 데이터 로딩: {len(df):,}행")
            
        except Exception as e:
            print(f"ERROR CSV 로딩 실패: {e}")
            raise
        
        try:
            train_df, val_df, test_df, community_map, degree_map = self.preprocess(df)

            # 샘플링 전에 노드 임베딩 처리
            print("노드 임베딩 생성 (샘플링 전)")
            _, account_to_idx_train = self.create_node_embeddings(train_df, community_map, degree_map)
            _, account_to_idx_val = self.create_node_embeddings(val_df, community_map, degree_map)
            _, account_to_idx_test = self.create_node_embeddings(test_df, community_map, degree_map)

            datasets = {
                'train': self.sample_data(train_df),
                'val': self.sample_data(val_df),
                'test': self.sample_data(test_df)
            }

            print("그래프 변환 시작")
            processed_data = {}
            for split_name, split_df in datasets.items():
                print(f"INFO 처리 중: {split_name}")
                # 샘플링된 데이터로 노드 임베딩 재생성
                node_features, account_to_idx = self.create_node_embeddings(split_df, community_map, degree_map)
                processed_data[split_name] = self.create_graph_data(split_df, node_features, account_to_idx)
            
            torch.save((processed_data['train'], torch.tensor([]),
                        processed_data['val'], torch.tensor([]),
                        processed_data['test'], torch.tensor([])), self.processed_paths[0])
            print("SUCCESS 데이터 저장 완료!")
            
        except Exception as e:
            print(f"ERROR 데이터 처리 중 오류: {e}")
            import traceback
            traceback.print_exc()
            raise

    #create_graph_data 함수 수정
    def create_graph_data(self, df, node_features, account_to_idx):
        try:
            edge_attr, edge_index, df_valid = self.get_edge_df(df, account_to_idx)
            edge_labels = torch.tensor(df_valid['ff_sp_ai'].values, dtype=torch.float)

            # 데이터 검증
            if edge_index.shape[1] != len(edge_labels):
                raise ValueError(f"엣지 수 불일치: {edge_index.shape[1]} vs {len(edge_labels)}")
            
            if node_features.shape[0] == 0:
                raise ValueError("노드가 없습니다!")
            
            print(f"SUCCESS 그래프 데이터: 노드={node_features.shape[0]}, 엣지={edge_index.shape[1]}")

            # PyG의 Data 객체 생성
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=edge_labels
            )

            return data
            
        except Exception as e:
            print(f"ERROR 그래프 데이터 생성 중 오류: {e}")
            raise


## model.py (기존 모델을 EdgeGAT로 교체)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Linear
import warnings
warnings.filterwarnings('ignore')

class EdgeGAT(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels=64, out_channels=1, heads=4):
        super().__init__()
        self.node_in = Linear(in_channels, hidden_channels)

        # GAT layer 1
        self.gat1 = GATv2Conv(hidden_channels, hidden_channels, heads=heads,
                              dropout=0.3, edge_dim=edge_dim)
        self.norm1 = nn.BatchNorm1d(hidden_channels * heads)

        # GAT layer 2
        self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, concat=False,
                              dropout=0.3, edge_dim=edge_dim)
        self.norm2 = nn.BatchNorm1d(hidden_channels)

        #엣지 스코어링 헤드 추가
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2 + edge_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        # 노드 임베딩 생성
        x = F.relu(self.node_in(x))
        x = F.dropout(x, p=0.3, training=self.training)

        # GAT layer 1
        x1 = self.gat1(x, edge_index, edge_attr)
        x = F.relu(self.norm1(x1))
        x = F.dropout(x, p=0.3, training=self.training)

        # GAT layer 2
        h = self.gat2(x, edge_index, edge_attr)
        h = F.relu(self.norm2(h))

        #엣지 스코어링
        src, dst = edge_index
        h_src, h_dst = h[src], h[dst]
        edge_feat = torch.cat([h_src, h_dst, edge_attr], dim=1)
        logit = self.edge_mlp(edge_feat).squeeze(-1)
        
        return logit  # BCEWithLogitsLoss 사용


## train.py (모델명만 변경)
import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from model import EdgeGAT  #모델명 변경
from dataset import AMLtoGraph
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class FocalLoss(torch.nn.Module) :
    def __init__(self, alpha=0.95, gamma=1, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets) :
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean' :
            return loss.mean()
        elif self.reduction == 'sum' :
            return loss.sum()
        else:
            return loss

# --------- Top-K 평가 함수 ----------
def precision_at_k(y_true, y_pred_proba, k):
    """Precision@K 계산"""
    top_k_idx = np.argsort(y_pred_proba)[::-1][:k]
    return np.sum(y_true[top_k_idx]) / k

def recall_at_k(y_true, y_pred_proba, k):
    """Recall@K 계산"""
    top_k_idx = np.argsort(y_pred_proba)[::-1][:k]
    return np.sum(y_true[top_k_idx]) / np.sum(y_true)

def threshold_at_k(y_true, y_pred_proba, k):
    """Threshold@K 계산 - 상위 K번째 샘플의 확률값 반환"""
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
model = EdgeGAT(  #모델명 변경
    in_channels=dataset.data_train.num_node_features,
    edge_dim=dataset.data_train.edge_attr.shape[1],
    out_channels=1
).to(device)

print("\n 모델 학습 시작")
loss_fn = FocalLoss(alpha=0.95, gamma=1, reduction='mean').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

best_f1 = 0
patience_counter = 0

# --------- 학습 루프 ----------
for epoch in range(epoch_val):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)  #EdgeGAT 출력
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
        torch.save(model.state_dict(), "best_model.pth")
        np.save("val_preds.npy", preds)
        np.save("val_labels.npy", labels)

        print("Best model and predictions saved.")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
            
            