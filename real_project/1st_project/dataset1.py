import datetime
import os
from typing import Optional, Callable
import pandas as pd
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


class AMLtoGraph(InMemoryDataset):
    def __init__(self, root: str, edge_window_size: int = 10,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.edge_window_size = edge_window_size
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return 'HF_TRNS_TRAN.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def preprocess(self, df):
        df = df.fillna('00', inplace=False)
        df['ff_sp_ai'] = df['ff_sp_ai'].apply(lambda x: 1 if x == 'SP' else 0)

        df['tran_dt'] = pd.to_datetime(df['tran_dt'])
        df['tran_dt'] = df['tran_dt'].apply(lambda x: x.value)
        df['tran_dt'] = (df['tran_dt'] - df['tran_dt'].min()) / (df['tran_dt'].max() - df['tran_dt'].min())

        df['wd_ac_sn'] = df['wd_fc_sn'].astype(str) + '_' + df['wd_ac_sn'].astype(str)
        df['dps_ac_sn'] = df['dps_fc_sn'].astype(str) + '_' + df['dps_ac_sn'].astype(str)
        df = df.sort_values(by=['wd_ac_sn'])

        return df

    def get_all_account_merge(self, df):
        """계좌별 데이터 통합 및 사기 계좌 정보 병합"""
        ldf = df[['wd_ac_sn', 'wd_fc_sn']]
        rdf = df[['dps_ac_sn', 'dps_fc_sn']]
        suspicious = df[df['ff_sp_ai'] != 0]

        s1 = suspicious[['wd_ac_sn', 'ff_sp_ai']].rename({'wd_ac_sn': 'ac_sn'}, axis=1)
        s2 = suspicious[['dps_ac_sn', 'ff_sp_ai']].rename({'dps_ac_sn': 'ac_sn'}, axis=1)
        suspicious_accounts = pd.concat([s1, s2], join='outer').drop_duplicates(subset=['ac_sn'])

        ldf = ldf.rename({'wd_fc_sn': 'fc_sn', 'wd_ac_sn': 'ac_sn'}, axis=1)
        rdf = rdf.rename({'dps_ac_sn': 'ac_sn', 'dps_fc_sn': 'fc_sn'}, axis=1)
        all_accounts_df = pd.concat([ldf, rdf], join='outer').drop_duplicates()

        all_accounts_df['ff_sp_ai'] = 0
        df_merged = pd.merge(all_accounts_df, suspicious_accounts, on='ac_sn', how='left', suffixes=('_left', '_right'))
        df_merged['ff_sp_ai'] = df_merged['ff_sp_ai_right'].fillna(df_merged['ff_sp_ai_left']).astype(int)
        df_merged = df_merged[['ac_sn', 'fc_sn', 'ff_sp_ai']]

        return df_merged 

    def get_edge_df(self, df):
        df['wd_ac_sn'] = df['wd_ac_sn'].astype('category').cat.codes
        df['dps_ac_sn'] = df['dps_ac_sn'].astype('category').cat.codes
        edge_index = torch.tensor([df['wd_ac_sn'].values, df['dps_ac_sn'].values], dtype=torch.long)
        edge_attr = torch.tensor(df[['tran_amt']].values, dtype=torch.float)
        return edge_attr, edge_index

    def get_betweenness_tensor(self, nx_graph, accounts):
        betweenness = nx.betweenness_centrality(nx_graph)
        values = [betweenness.get(i, 0) for i in range(accounts.shape[0])]
        return torch.tensor(values, dtype=torch.float).unsqueeze(1)

    def get_degree_tensor(self, nx_graph, accounts):
        degree = nx.degree_centrality(nx_graph)
        values = [degree.get(i, 0) for i in range(accounts.shape[0])]
        return torch.tensor(values, dtype=torch.float).unsqueeze(1)

    def get_closeness_tensor(self, nx_graph, accounts):
        try:
            closeness = nx.closeness_centrality(nx_graph)
        except nx.NetworkXError:
            closeness = {}
        values = [closeness.get(i, 0) for i in range(accounts.shape[0])]
        return torch.tensor(values, dtype=torch.float).unsqueeze(1)

    def get_transaction_frequency_tensor(self, nx_graph, accounts):
        degree = nx.degree_centrality(nx_graph)
        values = [degree.get(i, 0) for i in range(accounts.shape[0])]
        return torch.tensor(values, dtype=torch.float).unsqueeze(1)

    def get_in_out_degree_ratio_tensor(self, nx_graph, accounts):
        in_degree = nx.in_degree_centrality(nx_graph)
        out_degree = nx.out_degree_centrality(nx_graph)
        in_values = [in_degree.get(i, 0) for i in range(accounts.shape[0])]
        out_values = [out_degree.get(i, 0) for i in range(accounts.shape[0])]
        in_tensor = torch.tensor(in_values, dtype=torch.float).unsqueeze(1)
        out_tensor = torch.tensor(out_values, dtype=torch.float).unsqueeze(1)
        return (out_tensor + 1e-6) / (in_tensor + 1e-6)

    def get_pagerank_tensor(self, nx_graph, accounts):
        pagerank = nx.pagerank(nx_graph)
        values = [pagerank.get(i, 0) for i in range(accounts.shape[0])]
        return torch.tensor(values, dtype=torch.float).unsqueeze(1)

    def get_clustering_coefficient_tensor(self, nx_graph, accounts):
        clustering = nx.clustering(nx_graph)
        values = [clustering.get(i, 0) for i in range(accounts.shape[0])]
        return torch.tensor(values, dtype=torch.float).unsqueeze(1)

    def get_avg_neighbor_degree_tensor(self, nx_graph, accounts):
        avg_neighbor_degree = nx.average_neighbor_degree(nx_graph)
        values = [avg_neighbor_degree.get(i, 0) for i in range(accounts.shape[0])]
        return torch.tensor(values, dtype=torch.float).unsqueeze(1)

    def get_shortest_path_related_tensor(self, nx_graph, accounts):
        try:
            avg_shortest_path = nx.average_shortest_path_length(nx_graph)
        except nx.NetworkXError:
            avg_shortest_path = 0
        try:
            diameter = nx.diameter(nx_graph)
        except nx.NetworkXError:
            diameter = 0
        return torch.tensor([[avg_shortest_path, diameter]] * accounts.shape[0], dtype=torch.float)

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        df = self.preprocess(df)
        edge_attr, edge_index = self.get_edge_df(df)
        node_label = torch.tensor(df['ff_sp_ai'].values, dtype=torch.float)
        
        # 노드 속성 추가 (중요!)
        node_attr = torch.tensor(df[['tran_dt', 'tran_amt']].values, dtype=torch.float)

        # # NetworkX 그래프 생성
        # nx_graph = nx.Graph()
        # edges = edge_index.T.tolist()
        # nx_graph.add_edges_from(edges)

        # # 노드 속성 추가
        # node_attr = torch.cat([self.get_betweenness_tensor(nx_graph, df)], dim=1)
        # node_attr = torch.cat([node_attr, self.get_degree_tensor(nx_graph, df)], dim=1)
        # node_attr = torch.cat([node_attr, self.get_closeness_tensor(nx_graph, df)], dim=1)
        # node_attr = torch.cat([node_attr, self.get_transaction_frequency_tensor(nx_graph, df)], dim=1)
        # node_attr = torch.cat([node_attr, self.get_in_out_degree_ratio_tensor(nx_graph, df)], dim=1)
        # node_attr = torch.cat([node_attr, self.get_pagerank_tensor(nx_graph, df)], dim=1)
        # node_attr = torch.cat([node_attr, self.get_clustering_coefficient_tensor(nx_graph, df)], dim=1)
        # node_attr = torch.cat([node_attr, self.get_avg_neighbor_degree_tensor(nx_graph, df)], dim=1)
        # node_attr = torch.cat([node_attr, self.get_shortest_path_related_tensor(nx_graph, df)], dim=1)

        # PyTorch Geometric 데이터 변환
        data = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, y=node_label)

        torch.save((data, None), self.processed_paths[0])