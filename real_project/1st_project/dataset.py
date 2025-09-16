import datetime
import os
from typing import Optional, Callable
import pandas as pd
from sklearn import preprocessing # 레이블 인코딩과 같은 전처리 작업 수행
import numpy as np
import torch
import networkx as nx # NetworkX 라이브러리 import
from torch_geometric.data import (
    Data,
    InMemoryDataset
) # PyTorch Geometric 의 데이터 구조를 정의
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


class AMLtoGraph(InMemoryDataset):
# PyTorch Geometric의 InMemoryDataset을 상속받아 메모리 내에서 데이터셋을 처리.

    def __init__(self, root: str, edge_window_size: int = 10,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.edge_window_size = edge_window_size
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
				# __init__ : 생성자
        # root: 데이터셋이 저장될 루트 디렉토리.
        # edge_window_size: 간선 데이터를 처리할 때 사용할 윈도우 크기 (현재 코드에서는 사용되지 않음).
        # transform, pre_transform: 데이터 변환 및 전처리 함수.
        
    @property
    def raw_file_names(self) -> str:
        return 'HF_TRNS_TRAN.csv' ## 원본 데이터

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def num_nodes(self) -> int:
        return self._data.edge_index.max().item() + 1
        # num_nodes : 그래프의 노드 수를 반환. edge_index 에서 최대 노드 인덱스를 찾아 계산.
        

    def df_label_encoder(self, df, columns):
        le = preprocessing.LabelEncoder()
        for i in columns:
            df[i] = le.fit_transform(df[i].astype(str))
        return df
	# df_label_encoder : 지정된 열에 대해 레이블 인코딩 수행. 문자열 값을 정수로 변환.

    def preprocess(self, df):
        df = df.fillna('00', inplace=False)
				df.loc[df['ff_sp_ai'] == 'SP', 'ff_sp_ai'] = '04'

        # 문자 숫자를 정수로 변환
        df['ff_sp_ai'] = df['ff_sp_ai'].astype(float)

        df['tran_dt'] = pd.to_datetime(df['tran_dt'])
        df['tran_dt'] = df['tran_dt'].apply(lambda x: x.value)
        df['tran_dt'] = (df['tran_dt'] - df['tran_dt'].min()) / (df['tran_dt'].max() - df['tran_dt'].min()) # 1~0 사이 값으로 정규화

        df['wd_ac_sn'] = df['wd_fc_sn'].astype(str) + '_' + df['wd_ac_sn'].astype(str)
        df['dps_ac_sn'] = df['dps_fc_sn'].astype(str) + '_' + df['dps_ac_sn'].astype(str) # 은행 정보와 결합하여 고유한 계정 식별자 생성
        df = df.sort_values(by=['wd_ac_sn'])

        receiving_df = df[['dps_ac_sn', 'tran_amt', 'fnd_type']] # 송금 및 수취 데이터 분리.
        paying_df = df[['wd_ac_sn', 'tran_amt', 'fnd_type']]

        receiving_df = receiving_df.rename({'dps_ac_sn': 'ac_sn'}, axis=1)
        paying_df = paying_df.rename({'wd_ac_sn': 'ac_sn'}, axis=1)

        fnd_type_ls = sorted(df['fnd_type'].unique()) # 자금유형 목록 생성

        return df, receiving_df, paying_df, fnd_type_ls

    def get_all_account_merge(self, df):
        ldf = df[['wd_ac_sn', 'wd_fc_sn']]
        rdf = df[['dps_ac_sn', 'dps_fc_sn']]
        suspicious = df[df['ff_sp_ai'] != 0]

        s1 = suspicious[['wd_ac_sn', 'ff_sp_ai']]
        s1 = s1.rename({'wd_ac_sn': 'ac_sn'}, axis=1)

        s2 = suspicious[['dps_ac_sn', 'ff_sp_ai']]
        s2 = s2.rename({'dps_ac_sn': 'ac_sn'}, axis=1)
        suspicious_accounts = pd.concat([s1, s2], join='outer')
        suspicious_accounts = suspicious_accounts.drop_duplicates(subset=['ac_sn']) # ac_sn 기준으로 중복 제거

        ldf = ldf.rename({'wd_fc_sn': 'fc_sn', 'wd_ac_sn': 'ac_sn'}, axis=1)
        rdf = rdf.rename({'dps_ac_sn': 'ac_sn', 'dps_fc_sn': 'fc_sn'}, axis=1)
        all_accounts_df = pd.concat([ldf, rdf], join='outer')
        all_accounts_df = all_accounts_df.drop_duplicates()

        all_accounts_df['ff_sp_ai'] = 0

        # merge를 사용하여 suspicious_accounts 의 'ff_sp_ai' 정보를 all_accounts_df 에 업데이트
        df_merged = pd.merge(all_accounts_df, suspicious_accounts, on='ac_sn', how='left', suffixes=('_left', '_right'))

        # 'ff_sp_ai_right' 가 NaN 이 아니면 (즉, suspicious account), 해당 값으로 'ff_sp_ai_left' 업데이트
        df_merged['ff_sp_ai'] = df_merged['ff_sp_ai_right'].fillna(df_merged['ff_sp_ai_left']).astype(int)

        df_merged = df_merged[['ac_sn', 'fc_sn', 'ff_sp_ai']] # 불필요한 컬럼 제거 및 순서 정리

        return df_merged

    def paid_currency_aggregate(self, fnd_type_ls, paying_df, accounts):
        for i in fnd_type_ls:
            temp = paying_df[paying_df['fnd_type'] == i]
            accounts['avg paid '+str(i)] = temp['tran_amt'].groupby(temp['ac_sn']).transform('mean')
        return accounts
    # 각 통화별로 송금 평균 금액을 계산.

    def received_currency_aggregate(self, fnd_type_ls, receiving_df, accounts):
        for i in fnd_type_ls:
            temp = receiving_df[receiving_df['fnd_type'] == i]
            accounts['avg received '+str(i)] = temp['tran_amt'].groupby(temp['ac_sn']).transform('mean')
        accounts = accounts.fillna(0)
        return accounts
    # 각 통화별로 수취 평균 금액을 계산.

    def get_edge_df(self, accounts, df):
        accounts = accounts.reset_index(drop=True)
        accounts['ID'] = accounts.index

        accounts['ID'] = accounts['ID'].index.astype(int)

        print('################ accounts ##################')
        print(accounts.dtypes)
        print(accounts.columns)
        print('############################################')

        mapping_dict = dict(zip(accounts['ac_sn'], accounts['ID']))

        df['wd_ac_sn'] = df['wd_ac_sn'].map(mapping_dict)
        df['dps_ac_sn'] = df['dps_ac_sn'].map(mapping_dict)
        

        edge_index = torch.stack([
            torch.from_numpy(df['wd_ac_sn'].values),
            torch.from_numpy(df['dps_ac_sn'].values)], dim=0)

        df = df.drop(['ff_sp_ai', 'wd_fc_sn', 'dps_fc_sn', 'wd_ac_sn', 'dps_ac_sn'], axis=1)

        print('################## df #####################')
        print(df.dtypes)
        print(df.columns)
        print('############################################')

        edge_attr = torch.from_numpy(df.values).to(torch.float)

        return edge_attr, edge_index

        # ## 간선 데이터프레임 생성.
        # ## 계정을 정수 ID로 매핑.
        # ## edge_index: 송금 계좌와 수취 계좌의 인덱스 쌍.
        # ## edge_attr: 간선 특성 (금액, 통화 등).

    def get_node_attr(self, fnd_type_ls, paying_df, receiving_df, accounts):
        node_df = self.paid_currency_aggregate(fnd_type_ls, paying_df, accounts)
        node_df = self.received_currency_aggregate(fnd_type_ls, receiving_df, node_df)
        node_label = torch.from_numpy(node_df['ff_sp_ai'].values).to(torch.float)
        node_df = node_df.drop(['ff_sp_ai', 'ac_sn'], axis=1)
        node_df = self.df_label_encoder(node_df, ['fc_sn'])
        node_df = torch.from_numpy(node_df.values).to(torch.float)

        return node_df, node_label

        # # 노드 특성 및 레이블 생성.
        # ## node_df: 노드 특성 (평균 송금/수취 금액).
        # ## node_label: 불법 거래 여부.

    ######### networkX 의 계좌중심성 지표(3) / 자금세탁 패턴 지표(6) 함수 시작 ###################################################
    ######### networkX 의 중심성 지표 시작 ######################################################################################
    def get_betweenness_tensor(nx_graph, accounts) :
        betweenness_centrality = nx.betweenness_centrality(nx_graph)
        # 중심성 Feature Node Feature에 추가
        
        betweenness_values = [betweenness_centrality.get(i, 0) for i in range(accounts.shape[0])]   # 각 노드에 대한 betweenness 값 추출, 없는 경우 0으로 처리
        betweenness_tensor = torch.tensor(betweenness_values, dtype=torch.float).unsqueeze(1)       # tensor 로 변환 및 차원 조정

        return betweenness_tensor
    # end of def get_betweenness_tensor

    def get_degree_tensor(nx_graph, accounts):
        degree_centrality = nx.degree_centrality(nx_graph)
        # degree_centrality = nx.degree_centrality(nx_graph) # or use nx.degree(nx_graph) which returns degree directly
        # 연결 중심성 (Degree Centrality)
        
        degree_values = [degree_centrality.get(i, 0) for i in range(accounts.shape[0])]
        degree_tensor = torch.tensor(degree_values, dtype=torch.float).unsqueeze(1)

        return degree_tensor
        # end of def get_degree_tensor

    def get_closeness_tensor(nx_graph, accounts):
        closeness_centrality = {} # disconnected graph 방지 위해 초기화
        # 근접 중심성 (Closeness Centrality)
        try:
            closeness_centrality = nx.closeness_centrality(nx_graph)
        except nx.NetworkXError: # disconnected graph일 경우 처리
            print("Warning: Graph is disconnected. Closeness centrality might not be calculated for all nodes.")
            # disconnected graph 경우, closeness_centrality는 빈 dict {} 로 남게 됨 (아래에서 get() 호출 시 key 가 없으면 0 반환)
            closeness_values = [closeness_centrality.get(i, 0) for i in range(accounts.shape[0])] # get(i, 0) : key i 가 없으면 0 반환
            closeness_tensor = torch.tensor(closeness_values, dtype=torch.float).unsqueeze(1)
            
        return closeness_tensor
    # end of def get_closeness_tensor
## networkX의 계좌중심성 지표(3) 끝 #########################################

######### networkX 의 자금세탁 패턴 지표(6) 함수 시작 ######################################################################################
    def get_transaction_frequency_tensor(nx_graph, accounts) :
        # 1. 거래 빈도 (Transaction Frequency) / 거래량(Volume) (Transaction Volume) (Node Degree 활용)
        # - Degree Centrality 값 (이미 계산됨)을 거래 빈도/량 지표로 재활용
        degree_centrality = nx.degree_centrality(nx_graph) # or use nx.degree(nx_graph) which returns degree directly
        degree_values = [degree_centrality.get(i, 0) for i in range(accounts.shape[0])]
        transaction_frequency_tensor = torch.tensor(degree_values, dtype=torch.float).unsqueeze(1) # tensor 로 변환 및 차원 조정
        return transaction_frequency_tensor
    # end of def get_transaction_frequency_tensor

    def get_in_out_degree_ratio_tensor(nx_digraph, accounts) :
        # 2. In-Degree / Out-Degree 비율 (방향 그래프 고려)
        in_degree_centrality = nx.in_degree_centrality(nx_digraph)
        out_degree_centrality = nx.out_degree_centrality(nx_digraph)

        in_degree_values = [in_degree_centrality.get(i, 0) for i in range(accounts.shape[0])]
        out_degree_values = [out_degree_centrality.get(i, 0) for i in range(accounts.shape[0])]

        in_degree_tensor = torch.tensor(in_degree_values, dtype=torch.float).unsqueeze(1)
        out_degree_tensor = torch.tensor(out_degree_values, dtype=torch.float).unsqueeze(1)

        in_out_degree_ratio_tensor = (out_degree_tensor + 1e-6) / (in_degree_tensor + 1e-6) # 분모가 0이 되는 것 방지 위해 작은 값 더함
        return in_out_degree_ratio_tensor
    # end of def get_in_out_degree_ratio_tensor

    def get_pagerank_tensor(nx_digraph, accounts) :
        # 3. PageRank Centrality
        pagerank_centrality = nx.pagerank(nx_digraph) # 방향 그래프 사용
        pagerank_values = [pagerank_centrality.get(i, 0) for i in range(accounts.shape[0])]
        pagerank_tensor = torch.tensor(pagerank_values, dtype=torch.float).unsqueeze(1)
        return pagerank_tensor
    # end of def get_pagerank_tensor

    def get_clustering_coefficient_tensor(nx_graph, accounts) :
        # 4. Local Clustering Coefficient (클러스터링 계수)
        clustering_coefficient = nx.clustering(nx_graph)
        clustering_values = [clustering_coefficient.get(i, 0) for i in range(accounts.shape[0])]
        clustering_tensor = torch.tensor(clustering_values, dtype=torch.float).unsqueeze(1)
        return clustering_tensor
        # end of def get_clustering_coefficient_tensor

    def get_avg_neighbor_degree_tensor(nx_graph, accounts) :
        # 5. Average Neighbor Degree (평균 이웃 Degree)
        avg_neighbor_degree = nx.average_neighbor_degree(nx_graph) # 평균 Degree
        avg_neighbor_degree_values = [avg_neighbor_degree.get(i, 0) for i in range(accounts.shape[0])]
        avg_neighbor_degree_tensor = torch.tensor(avg_neighbor_degree_values, dtype=torch.float).unsqueeze(1)
        return avg_neighbor_degree_tensor
    # end of def get_avg_neighbor_degree_tensor

    def get_shortest_path_related_tensor(nx_graph, accounts) :
        # 6. Shortest Path Length 관련 지표 (평균 최단 경로 길이, Diameter 등)
        # 평균 최단 경로 길이 (disconnected graph일 경우 에러 발생 가능성 있음)
        try:
            avg_shortest_path = nx.average_shortest_path_length(nx_graph)
        except nx.NetworkXError:     # disconnected graph일 경우 에러 처리 (0 또는 -1 등의 값으로 대체하거나, component 별로 계산)
            avg_shortest_path = 0    # 또는 avg_shortest_path = -1

        # 네트워크 지름 (disconnected graph일 경우 에러 발생 가능성 있음)
        try:
            diameter = nx.diameter(nx_graph)
        except nx.NetworkXError:     # disconnected graph일 경우 에러 처리
            diameter = 0             # 또는 diameter = -1

        avg_shortest_path_values = [avg_shortest_path] * accounts.shape[0] #  모든 노드에 동일 값 적용
        diameter_values = [diameter] * accounts.shape[0]     #  모든 노드에 동일 값 적용

        avg_shortest_path_tensor = torch.tensor(avg_shortest_path_values, dtype=torch.float).unsqueeze(1)
        diameter_tensor = torch.tensor(diameter_values, dtype=torch.float).unsqueeze(1)

        shortest_path_related_tensor = torch.cat([avg_shortest_path_tensor, diameter_tensor], dim=1) # 평균 최단경로길이, 지름 함께 return
        return shortest_path_related_tensor
    # end of def get_shortest_path_related_tensor

    ######### networkX 의 자금세탁 패턴 지표(6) 끝 ######################################################################################
    ######### networkX 의 계좌중심성 지표(3) / 자금세탁 패턴 지표(6) 함수 끝 ###############################################################

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        df, receiving_df, paying_df, fnd_type_ls = self.preprocess(df)
        accounts = self.get_all_account_merge(df)

        print('+++++++++++++++ accounts ++++++++++++++++')
        print(accounts.dtypes)
        print(accounts.columns)
        print('++++++++++++++++++++++++++++++++++++++++++')

        print('+++++++++++++++ df ++++++++++++++++')
        print(df.dtypes)
        print(df.columns)
        print('++++++++++++++++++++++++++++++++++++')

        node_attr, node_label = self.get_node_attr(fnd_type_ls, paying_df, receiving_df, accounts)
        edge_attr, edge_index = self.get_edge_df(accounts, df)

#################################################### 2025.02.19 ########################
    # NetworkX 그래프 생성
    # nx_graph = nx.Graph()     # or nx.DiGraph() depending on your graph directionality
    # edge_list = edge_index.T.tolist() # edge_index 를 NetworkX edge 형식으로 변환
    # nx_graph.add_edges_from(edge_list)

    #### 계좌 중심성 지표 3가지 함수 호출 추가 - 기존 node_attr에 중심성 feature 연결 ###
    # node_attr = torch.cat([node_attr, get_betweenness_tensor(nx_graph, accounts)], dim=1) # 1. 중심성(Centrality) 계산 (Betweenness Centrality)
    # node_attr = torch.cat([node_attr, get_degree_tensor(nx_graph, accounts)], dim=1) # 2. 연결 중심성 (Degree Centrality)
    # node_attr = torch.cat([node_attr, get_closeness_tensor(nx_graph, accounts)], dim=1) # 3. 근접 중심성 (Closeness Centrality)
    #### 정적 중심성 지표 3가지 함수 호출 끝 ###

    #### 자금세탁 패턴 지표 6가지 함수 호출 추가 ###
    # node_attr = torch.cat([node_attr, get_transaction_frequency_tensor(nx_graph, accounts)], dim=1)    # 1. 거래빈도
    # node_attr = torch.cat([node_attr, get_in_out_degree_ratio_tensor(nx_graph, accounts)], dim=1)    # 2. In-Degree / Out-Degree 비율 (방향 그래프 고려)
    # node_attr = torch.cat([node_attr, get_pagerank_tensor(nx_graph, accounts)], dim=1)   # 3. Page Rank Centrality
    # node_attr = torch.cat([node_attr, get_clustering_coefficient_tensor(nx_graph, accounts)], dim=1)    # 4. 클러스터링 계수
    # node_attr = torch.cat([node_attr, get_avg_neighbor_degree_tensor(nx_graph, accounts)], dim=1)       # 5. 평균 이웃 Degree
    # node_attr = torch.cat([node_attr, get_shortest_path_related_tensor(nx_graph, accounts)], dim=1) # 6. 평균 최단 경로 길이
    #### 자금세탁 패턴 지표 6가지 함수 호출 끝 ###

    data = Data(x=node_attr,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=node_label,
                )

        #### downsampling ratio = 0.2 로 다운샘플링 비율 설정 ### [다운샘플링 코드 시작] ####
        # downsample_ratio = 0.2 # 다운샘플링 비율 (label 0 데이터 샘플링 사용, 0.0 ~ 1.0)
        # num_label_0 = (node_label == 0).sum().item() # label 0 갯수
        # num_label_1 = (node_label == 1).sum().item() # label 1 갯수

        # if num_label_0 > 0: # label 0 데이터 있는 경우만 downsample 적용
            # downsampled_label_0_count = int(num_label_0 * downsample_ratio) # 다운샘플링할 label 0 데이터 갯수
            # label_0_indices = (node_label == 0).nonzero(as_tuple=True)[0] # label 0 node index
            # rand_perm = torch.randperm(num_label_0)   # label 0 index 랜덤 섞기
            # downsampled_indices_label_0 = label_0_indices[rand_perm[:downsampled_label_0_count]]  # 다운샘플링된 label_0 index 선택
            # label_1_indices = (node_label == 1).nonzero(as_tuple=True)[0]  # label 1 node index
            # train_indices = torch.cat([downsampled_indices_label_0, label_1_indices]) # 다운샘플링된 label 0 index + label 1 index
            
        # else: # else 구문 추가 (node_label == 1) 경우
            # train_indices = (node_label == 1).nonzero(as_tuple=True)[0] # label 0 데이터가 없으면 label 1 데이터만 사용 (혹은 에러처리)
            
        # train_mask = torch.zeros(node_label.size(0), dtype=torch.bool) # 전체 node mask 생성(False로 초기화)
        # train_mask[train_indices] = True   # 다운샘플링된 index에 해당하는 mask만 True로 설정
        # data.train_mask = train_mask    # train_mask 를 Data 객체에 할당

        # data.val_mask = None   # 다운샘플링 적용: validation/test mask 로 train.py에서 RandomNodeSplit 으로 생성하므로 None 으로 설정
        # data.test_mask = None  
        ############################################### [다운샘플링 코드 끝] ############################################################################################

    data_list = [data]

    if self.pre_filter is not None:
        data_list = [d for d in data_list if self.pre_filter(d)]

    if self.pre_transform is not None:
        data_list = [self.pre_transform(d) for d in data_list]

    data, slices = self.collate(data_list)
    torch.save((data, slices), self.processed_paths[0])     # self.processed_paths=['dataset/processed/data.pt'] # dataset/processed 경로에 pickle 형태로 저장

# process: 데이터 전처리 파이프라인. pd.read_csv 부터 시작하여 최종 Data 객체를 생성하는 모든 과정 포함.
# ## 원본 데이터를 로드하고, 전처리하고, 그래프를 생성하고, Data 객체로 저장하는 전체 과정.