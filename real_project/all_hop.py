#### dataset.py
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

        ## df = df[df["tran_amt"] >= 10000] ---- 사기데이터 모두 포함하도록 변경. 정상데이터에 대해서는 어떻게 해야 할지... 몰라서 일단 포함.
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

			#split_df['community'] = split_df['dps_fc_ac'].map(community_map).astype(object).fillna(0).astype(int) 에러 나서 아래와 같이 변경
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
        print("금액대별 17% 샘플링 (정상 중심 + 사기 오버샘플링) 시작")

        bins = [0, 1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, np.inf]
        labels = ['<1K', '1K-10K', '10K-50K', '50K-100K', '100K-500K', '500K-1M', '>1M']
        df['amt_bin'] = pd.cut(df['tran_amt'], bins=bins, labels=labels, include_lowest=True)

        all_samples = []

        for bin_label in labels:
            bin_df = df[df['amt_bin'] == bin_label]
            normal = bin_df[bin_df['ff_sp_ai'] == 0]
            fraud = bin_df[bin_df['ff_sp_ai'] == 1]

            #정상 거래 샘플링 (17%)
            n_normal_sample = int(len(normal) * 0.17)
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
        # end of def sample_data
        
        
    def hop_oversample(self, df, hop=1, verbose=False):
        """
        사기 거래 노드를 중심으로 hop 이웃 노드까지 포함하여 오버샘플링하는 함수.
        GNN 학습용 서브그래프 생성에 적합.

        Parameters:
            df (pd.DataFrame): 전체 거래 데이터
            hop (int): 몇 hop까지 이웃을 포함할지
            verbose (bool): 로그 출력 여부

        Returns:
            pd.DataFrame: hop 기반 서브그래프 데이터프레임
        """
        import networkx as nx

        # 그래프 구성
        G = nx.from_pandas_edgelist(df, source='wd_fc_ac', target='dps_fc_ac', create_using=nx.DiGraph())

        # 사기 계좌 기준
        fraud_df = df[df['ff_sp_ai'] == 1]
        fraud_accounts = set(fraud_df['wd_fc_ac']).union(set(fraud_df['dps_fc_ac']))

        # N-hop 이웃 추출
        selected_accounts = set()
        for acc in fraud_accounts:
            try:
                neighbors = nx.single_source_shortest_path_length(G, acc, cutoff=hop).keys()
                selected_accounts.update(neighbors)
            except:
                continue

        # 입출금 계좌가 hop 이웃 안에 포함된 거래만 유지
        hop_df = df[df['wd_fc_ac'].isin(selected_accounts) | df['dps_fc_ac'].isin(selected_accounts)].copy()

        if verbose:
            print(f"[HOP-{hop}] 포함된 계좌 수: {len(selected_accounts)}")
            print(f"[HOP-{hop}] 거래 수: {len(hop_df)}")
            fraud_cnt = hop_df['ff_sp_ai'].sum()
            normal_cnt = len(hop_df) - fraud_cnt
            print(f"[HOP-{hop}] 사기: {fraud_cnt}, 정상: {normal_cnt}")

        return hop_df
        # end of def hop_oversample


    def hop_recover_oversample(self, df, hop=2, verbose=False):
        """
        Hop 기반 샘플링 + 금액대별 정상 17% + 사기 복원샘플링 방식

        Parameters:
            df (pd.DataFrame): 전체 거래 데이터
            hop (int): 몇 hop까지 이웃을 포함할지
            verbose (bool): 로그 출력 여부

        Returns:
            pd.DataFrame: hop 기반 오버샘플링 결과 데이터프레임
        """
        import networkx as nx

        print(f"[HOP-{hop}] 서브그래프 기반 Recover OverSampling 시작")

        # 1. 전체 거래 그래프 생성
        G = nx.from_pandas_edgelist(df, source='wd_fc_ac', target='dps_fc_ac', create_using=nx.DiGraph())

        # 2. 사기 계좌 기준 hop 이웃 추출
        fraud_df_all = df[df['ff_sp_ai'] == 1]
        fraud_accounts = set(fraud_df_all['wd_fc_ac']).union(set(fraud_df_all['dps_fc_ac']))

        selected_accounts = set()
        for acc in fraud_accounts:
            try:
                neighbors = nx.single_source_shortest_path_length(G, acc, cutoff=hop).keys()
                selected_accounts.update(neighbors)
            except:
                continue

        # 3. hop 이웃 내 거래 필터링
        hop_df = df[df['wd_fc_ac'].isin(selected_accounts) | df['dps_fc_ac'].isin(selected_accounts)].copy()
        print(f"[HOP-{hop}] 거래 수: {len(hop_df)}, 계좌 수: {len(selected_accounts)}")

        # 4. 금액대 정의
        bins = [0, 1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, np.inf]
        labels = ['<1K', '1K-10K', '10K-50K', '50K-100K', '100K-500K', '500K-1M', '>1M']
        hop_df['amt_bin'] = pd.cut(hop_df['tran_amt'], bins=bins, labels=labels, include_lowest=True)

        # 5. 금액대별 샘플링 진행
        all_samples = []
        for label in labels:
            bin_df = hop_df[hop_df['amt_bin'] == label]
            normal = bin_df[bin_df['ff_sp_ai'] == 0]
            fraud = bin_df[bin_df['ff_sp_ai'] == 1]

            n_normal_sample = int(len(normal) * 0.17)
            if n_normal_sample > 0:
                normal_sampled = normal.sample(n=n_normal_sample, random_state=42)
            else:
                normal_sampled = normal.iloc[0:0]

            if len(fraud) > 0 and n_normal_sample > 0:
                fraud_sampled = resample(fraud, replace=True, n_samples=n_normal_sample, random_state=42)
            else:
                fraud_sampled = fraud.iloc[0:0]

            bin_sampled = pd.concat([normal_sampled, fraud_sampled], ignore_index=True)
            all_samples.append(bin_sampled)

            if verbose:
                print(f"[{label}] 정상: {len(normal_sampled)} / 사기 복원: {len(fraud_sampled)} → K={n_normal_sample}")

        hop_df_sampled = pd.concat(all_samples, ignore_index=True).sample(frac=1, random_state=42)

        if verbose:
            total_fraud = hop_df_sampled['ff_sp_ai'].sum()
            total_normal = len(hop_df_sampled) - total_fraud
            print(f"[HOP-{hop}] 최종 → 사기: {total_fraud}, 정상: {total_normal}, 비율: {(total_fraud / len(hop_df_sampled)) * 100:.2f}%")

        return hop_df_sampled

        # end of def hop_recover_oversample




    def get_edge_df(self, df):
        print("엣지 생성 시작")
        df['wd_fc_ac'] = df['wd_fc_ac'].astype('category').cat.codes
        df['dps_fc_ac'] = df['dps_fc_ac'].astype('category').cat.codes

        edge_index = torch.tensor([df['wd_fc_ac'].values, df['dps_fc_ac'].values], dtype=torch.long)

        edge_feat_cols = [
            'tran_amt', 'tran_dt', 'tran_tmrg',
            'dps_fc_ac_fnd_amt', 'dps_fc_ac_fnd_cnt', 'dps_fc_ac_md_amt', 'dps_fc_ac_md_cnt',
            'wd_fc_ac_fnd_amt', 'wd_fc_ac_fnd_cnt', 'wd_fc_ac_md_amt', 'wd_fc_ac_md_cnt'
        ]

        edge_attr = torch.tensor(df[edge_feat_cols].values, dtype=torch.float)
        print(f"엣지 수: {edge_index.shape[1]}, 엣지 특성 차원: {edge_attr.shape[1]}")
        return edge_attr, edge_index

    def get_node_attr(self, df):
        print("노드 특성 생성 시작")
        df = df.astype(object).fillna(0)
        node_cols = [
            'tran_amt', 'tran_tmrg', 'md_type', 'fnd_type',
            'prev_dps_fraud_cnt', 'prev_wd_fraud_cnt',
            'dps_fc_ac_fnd_cnt',  #  count 정보 하나 유지
            'dps_fc_ac_fnd_amt',  #  입금 계좌 기반 금액
            'community', 'degree_dps'
        ]
        node_attr = torch.tensor(df[node_cols].values, dtype=torch.float)
        node_label = torch.tensor(df['ff_sp_ai'].values, dtype=torch.float)
        print(f"노드 수: {node_attr.shape[0]}, 노드 특성 차원: {node_attr.shape[1]}")
        return node_attr, node_label

    def process(self):
        print("데이터 처리 시작")
        print(f"CSV 로딩 경로: {self.raw_paths[0]}")
        df = pd.read_csv(self.raw_paths[0], low_memory=False, dtype={'ff_sp_ai': str})
        train_df, val_df, test_df = self.preprocess(df)

        #datasets = {
        #    'train': self.sample_data(train_df),
        #    'val': self.sample_data(val_df),
        #    #'test': self.sample_data(test_df)  # 테스트도 fraction
		#	'test': test_df						# 테스트 샘플링 제외
        #}
        
        datasets = {
            'train': self.hop_recover_oversample(train_df),
            'val': val_df,                      # 검증 샘플링 제외
			'test': test_df						# 테스트 샘플링 제외
        }
        

        print("그래프 변환 시작")
        processed_data = {k: self.create_graph_data(d) for k, d in datasets.items()}
        torch.save((processed_data['train'], torch.tensor([]),
                    processed_data['val'], torch.tensor([]),
                    processed_data['test'], torch.tensor([])), self.processed_paths[0])
        print("데이터 저장 완료!")

    def create_graph_data(self, df):
        edge_attr, edge_index = self.get_edge_df(df)
        node_attr, node_label = self.get_node_attr(df)

        # PyG의 Data 객체 생성
        data = Data(
            x=node_attr,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=node_label
        )

        return data



#### model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Linear
import warnings

warnings.filterwarnings('ignore')

class GATImproved(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels=64, out_channels=1, heads=4):
        super().__init__()
        self.lin_in = Linear(in_channels, hidden_channels)

        # GAT layer 1
        self.gat1 = GATv2Conv(hidden_channels, hidden_channels, heads=heads,
                              dropout=0.3, edge_dim=edge_dim)
        self.res_lin1 = Linear(hidden_channels, hidden_channels * heads)  # residual for GAT1
        self.norm1 = nn.BatchNorm1d(hidden_channels * heads)

        # GAT layer 2
        self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, concat=False,
                              dropout=0.3, edge_dim=edge_dim)
        self.res_lin2 = Linear(hidden_channels * heads, hidden_channels)  # residual for GAT2
        self.norm2 = nn.BatchNorm1d(hidden_channels)

        self.lin_out = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.lin_in(x))
        x = F.dropout(x, p=0.3, training=self.training)

        # GAT layer 1 + Residual
        x1 = self.gat1(x, edge_index, edge_attr)
        x = F.relu(self.norm1(x1 + self.res_lin1(x)))
        x = F.dropout(x, p=0.3, training=self.training)

        # GAT layer 2 + Residual
        x2 = self.gat2(x, edge_index, edge_attr)
        x = F.relu(self.norm2(x2 + self.res_lin2(x)))

        return self.lin_out(x)
		
## train.py
import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from model import GATImproved
from dataset import AMLtoGraph
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

class FocalLoss(torch.nn.Module) :
    def __init__(self, alpha=0.1, gamma=1, reduction='mean'):
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
    top_k_idx = np.argsort(y_pred_proba)[::-1][:k]
    return np.sum(y_true[top_k_idx]) / k

def recall_at_k(y_true, y_pred_proba, k):
    top_k_idx = np.argsort(y_pred_proba)[::-1][:k]
    return np.sum(y_true[top_k_idx]) / np.sum(y_true)

# --------- 하이퍼파라미터 설정 ----------
batch_size = 1024
lr = 0.01
topk = 30
patience = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Device: {device}")

# --------- 데이터 로딩 ----------
dataset = AMLtoGraph("./lej_dataset_002")
train_loader = DataLoader([dataset.data_train], batch_size=batch_size, shuffle=True)
val_loader = DataLoader([dataset.data_val], batch_size=batch_size)

# --------- 모델 및 Optimizer ----------
model = GATImproved(
    in_channels=dataset.data_train.num_node_features,
    edge_dim=dataset.data_train.edge_attr.shape[1],
    out_channels=1
).to(device)


print("\n 모델 학습 시작")
loss_fn = FocalLoss(alpha=0.1, gamma=1, reduction='mean').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

best_f1 = 0
patience_counter = 0

# --------- 학습 루프 ----------
for epoch in range(30):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr).view(-1)
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
            out = model(data.x, data.edge_index, data.edge_attr).view(-1)
            preds.extend(torch.sigmoid(out).cpu().numpy())
            labels.extend(data.y.cpu().numpy())

    preds = np.array(preds)
    labels = np.array(labels)
    bin_preds = (preds > 0.5).astype(int)


    f1 = f1_score(labels, bin_preds)
    precision = precision_score(labels, bin_preds)
    recall = recall_score(labels, bin_preds)
    roc = roc_auc_score(labels, preds)

    scheduler.step(f1)

    print(f"Epoch {epoch:02d} | Loss: {total_loss:.4f}")
    print(f"[Val] F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, ROC-AUC: {roc:.4f}")

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
    
