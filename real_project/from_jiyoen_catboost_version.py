# -*- coding: utf-8 -*-

###### 라이브러리 불러오기 ######
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from zoneinfo import ZoneInfo
from sklearn.utils import resample
import networkx as nx
from networkx.algorithms import community

from catboost import CatBoostClassifier, metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, make_scorer, confusion_matrix

###### 데이터 불러오기 #######
df = pd.read_csv('hf_trns_tran_new_12norm.csv', dtype={'ff_sp_ai': str})

# 날짜 컬럼을 문자열 → datetime으로 변환
df['tran_dt'] = pd.to_datetime(df['tran_dt'].astype(str), format='%Y%m%d')

# 월 / 연도 분리 (※ split 기준)
df['year'] = df['tran_dt'].dt.year
df['month'] = df['tran_dt'].dt.month

# timestamp (초단위 정수형) 피처 생성
df['tran_dt_seconds'] = df['tran_dt'].astype('int64') // 10**9

# Label 0/1로 변환
df['ff_sp_ai'] = df['ff_sp_ai'].apply(lambda x: 0 if x == '0' else 1)

### ---------------------------------------------
### 2) 타입변경 (범주형 feature 지정)
### ---------------------------------------------
cat_feats = ['wd_fc_ac', 'dps_fc_ac', 'md_type', 'fnd_type']
for c in cat_feats:
    df[c] = df[c].astype('category')

### ---------------------------------------------
### 3) feature list + train/val/test 분할
### ---------------------------------------------
features = ['wd_fc_ac', 'dps_fc_ac', 'md_type', 'fnd_type', 'tran_amt',
            'tran_dt_seconds', 'tran_tmrg',
            'wd_fc_ac_fnd_cnt', 'wd_fc_ac_fnd_amt', 'wd_fc_ac_md_cnt','wd_fc_ac_md_amt',
            'dps_fc_ac_fnd_cnt','dps_fc_ac_fnd_amt','dps_fc_ac_md_amt','dps_fc_ac_md_cnt',
            'prev_dps_fraud_cnt','prev_wd_fraud_cnt']
target = 'ff_sp_ai'

# 월 기준 split
X_train = df.loc[(df['month'] >= 1) & (df['month'] <= 10), features].copy()
y_train = df.loc[(df['month'] >= 1) & (df['month'] <= 10), target].copy()

X_val = df.loc[df['month'] == 11, features].copy()
y_val = df.loc[df['month'] == 11, target].copy()

X_test = df.loc[df['month'] == 12, features].copy()
y_test = df.loc[df['month'] == 12, target].copy()

print("훈련데이터: ", X_train.shape, y_train.shape)
print("검증데이터: ", X_val.shape, y_val.shape)
print("시험데이터: ", X_test.shape, y_test.shape)

### ---------------------------------------------
### 4) 날짜 timestamp 정규화
### ---------------------------------------------
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# 반드시 train으로 fit
X_train.loc[:, 'tran_dt_seconds'] = scaler.fit_transform(X_train[['tran_dt_seconds']])
X_val.loc[:, 'tran_dt_seconds']   = scaler.transform(X_val[['tran_dt_seconds']])
X_test.loc[:, 'tran_dt_seconds']  = scaler.transform(X_test[['tran_dt_seconds']])

## 커뮤니티 피처 만들기 ##
def detect_communities(df_feat):
    print("커뮤니티 탐지(Train 기반)")
    G = nx.Graph()
    for _, row in df_feat.iterrows():
        G.add_edge(row['wd_fc_ac'], row['dps_fc_ac'], weight=row['tran_amt'])

    communities = community.louvain_communities(G, weight='weight')
    community_map_local = {}
    for idx, nodes in enumerate(communities):
        for node in nodes:
            community_map_local[node] = idx

    return community_map_local

# 커뮤니티 피처를 기존 데이터에 추가 (GAT와 동일: dps_fc_ac 기준 매핑)
community_map = detect_communities(X_train)

for split in (X_train, X_val, X_test):
    split['community'] = split['dps_fc_ac'].map(community_map)
    split['community'] = split['community'].fillna(0).astype('int32')

## Degree 차수 만들기 ##
def add_degree_feature(df_feat):
    print("Degree 계산(Train 기반)")
    G = nx.Graph()
    G.add_edges_from(zip(df_feat['wd_fc_ac'], df_feat['dps_fc_ac']))
    degree_dict = dict(G.degree())
    return degree_dict

# Degree 차수를 기존데이터에 추가 (GAT와 동일: dps_fc_ac 기준)
degree_map = add_degree_feature(X_train)

X_train['degree_dps'] = X_train['dps_fc_ac'].map(degree_map).fillna(0)
X_val['degree_dps']   = X_val['dps_fc_ac'].map(degree_map).fillna(0)
X_test['degree_dps']  = X_test['dps_fc_ac'].map(degree_map).fillna(0)

###### GAT와 동일한 금액대별 1% 샘플링 ######

def sample_data(df, verbose=False, clf=''):
    """
    GAT 최종 샘플링: 정상 거래 1% 샘플링, 사기 거래는 정상과 개수 맞춤
    """
    print(clf + "샘플링 시작")
    
    def print_graph_stats(df_part, label=""):
        G = nx.Graph()
        for _, row in df_part.iterrows():
            G.add_edge(row['wd_fc_ac'], row['dps_fc_ac'], weight=row['tran_amt'])
        print(f"[{label}] 노드 수: {G.number_of_nodes()}, 엣지 수: {G.number_of_edges()}")
    
    normal_data = df[df['ff_sp_ai'] == 0]
    fraud_data = df[df['ff_sp_ai'] == 1]
    print(f"{clf} 원래 정상 거래 수: {len(normal_data)}, 사기 거래 수: {len(fraud_data)}")
    
    # 정상 거래 1% 샘플링
    normal_sampled = normal_data.sample(frac=0.01, random_state=42)
    
    # 사기 거래는 정상과 동일한 개수로 복원 샘플링
    fraud_sampled = resample(fraud_data, replace=True, n_samples=len(normal_sampled), random_state=42)
    
    # 합치고 셔플
    df_sampled = pd.concat([normal_sampled, fraud_sampled], ignore_index=True).sample(frac=1, random_state=42)
    
    print(f"{clf} 샘플링 후 정상 거래 수: {len(normal_sampled)}, 사기 거래 수: {len(fraud_sampled)}")
    
    if verbose:
        print_graph_stats(df, "샘플링 전 전체 데이터")
        print_graph_stats(df_sampled, "샘플링 후 전체 데이터")
    
    return df_sampled

## GAT 스타일 샘플링 적용 ##
print("=== GAT 스타일 1% 샘플링 수행 ===")

# 훈련 데이터 샘플링
# 훈련 데이터 샘플링 (X_train, y_train을 다시 합쳐서 처리)
print("[훈련 데이터]")
df_train_combined = pd.concat([X_train, y_train], axis=1)
df_sampled_tr = sample_data(df_train_combined, verbose=True, clf="[훈련] ")
X_resampled_tr = df_sampled_tr.drop('ff_sp_ai', axis=1)
y_resampled_tr = df_sampled_tr['ff_sp_ai']

# 검증 데이터 샘플링
print("[검증 데이터]")
df_val_combined = pd.concat([X_val, y_val], axis=1)
df_sampled_val = sample_data(df_val_combined, verbose=True, clf="[검증] ")
X_resampled_val = df_sampled_val.drop('ff_sp_ai', axis=1)
y_resampled_val = df_sampled_val['ff_sp_ai']

# 검증 데이터 샘플링
print("[테스트 데이터]")
df_val_combined = pd.concat([X_test, y_test], axis=1)
df_sampled_test = sample_data(df_test_combined, verbose=True, clf="[테스트] ")
X_resampled_test = df_sampled_test.drop('ff_sp_ai', axis=1)
y_resampled_test = df_sampled_test['ff_sp_ai']


###### TopK 성능지표정의 ######

## 모델 예측 후 평가하기 위한 TopK 성능지표 구현 ##
def precision_at_k(y_true, y_pred_proba, k):
    top_k_idx = np.argsort(y_pred_proba)[::-1][:k]
    y_true_arr = np.array(y_true)
    return np.sum(y_true_arr[top_k_idx]) / k

def recall_at_k(y_true, y_pred_proba, k):
    top_k_idx = np.argsort(y_pred_proba)[::-1][:k]
    y_true_arr = np.array(y_true)
    return np.sum(y_true_arr[top_k_idx]) / np.sum(y_true_arr)

def threshold_at_k(y_true, y_pred_proba, k):
    top_k_idx = np.argsort(y_pred_proba)[::-1][:k]
    threshold_index = top_k_idx[-1]
    return y_pred_proba[threshold_index]

def f1_score_at_k(y_true, y_pred_proba, k):
    precision_k = precision_at_k(y_true, y_pred_proba, k)
    recall_k = recall_at_k(y_true, y_pred_proba, k)
    if precision_k + recall_k == 0:
        return 0
    return 2*(precision_k*recall_k)/(precision_k+recision_k)

## CatBoost에서 모델 훈련 때 사용할 F1@300 평가함수 구현 ##
class eval_f1_score_at_k():
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight=None):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        k = 300

        y_pred_proba = 1 / (1 + np.exp(-np.array(approxes[0])))
        all_labels = np.column_stack((target, y_pred_proba))

        top_k_indice = np.argsort(all_labels[:, 1])[::-1][:k]
        top_k_true = all_labels[top_k_indice, 0]
        total_positive = np.sum(target == 1)

        precision_at_k_val = np.sum(top_k_true == 1) / k
        recall_at_k_val = np.sum(top_k_true == 1) / total_positive

        if precision_at_k_val + recall_at_k_val == 0:
            return 0, True

        f1_score_at_k_val = 2*(precision_at_k_val * recall_at_k_val) / (precision_at_k_val + recall_at_k_val)

        return f1_score_at_k_val, True

###### CatBoost 모델 훈련 ######
## CatBoost

# CatBoostClassifier 하이퍼파라미터 설정
cat_params = {
    'loss_function': 'Logloss',
    'random_seed': 0,
    'iterations': 100,
    'learning_rate': 0.1,
    'verbose': 10,
    'cat_features': ['wd_fc_ac', 'dps_fc_ac', 'md_type', 'fnd_type']
}

# CatBoost 모델 생성 및 학습
cbt = CatBoostClassifier(**cat_params, eval_metric=eval_f1_score_at_k())
cbt.fit(X_resampled_tr, y_resampled_tr, eval_set=[(X_resampled_val, y_resampled_val)], use_best_model=True)

# learning curve 시각화
plt.figure(figsize=(10, 7))
plt.plot(cbt.evals_result_["learn"]["eval_f1_score_at_k"], label="Training f1-score@K", linewidth=2)
plt.plot(cbt.evals_result_["validation"]["eval_f1_score_at_k"], label="Validation f1-score@K", linewidth=2)
plt.title("The learning curves for training and testing sets with f1-score@K", fontsize=14)
plt.xlabel("Number of trees", fontsize=12)
plt.ylabel("f1-score@K", fontsize=12)
plt.legend()
plt.show()

###### 예측 수행 ######
cbt_pred = cbt.predict(X_test)
cbt_pred_proba = cbt.predict_proba(X_test)[:, 1]

# 평가 지표 계산
acc = round(accuracy_score(y_resampled_test, cbt_pred), 4)
prec = round(precision_score(y_resampled_test, cbt_pred, zero_division=0), 4)
rec = round(recall_score(y_resampled_test, cbt_pred), 4)
f1 = round(f1_score(y_resampled_test, cbt_pred), 4)
roc_auc = round(roc_auc_score(y_resampled_test, cbt_pred_proba), 4)

# 혼동 행렬 계산
tn, fp, fn, tp = confusion_matrix(y_resampled_test, cbt_pred).ravel()

# FPR 계산
fpr = round(fp / (fp + tn), 6)

# 결과
print("Accuracy", acc, "\n",
      "Precision", prec, "\n",
      "Recall", rec, "\n",
      "F1-score", f1, "\n",
      "ROC-AUC", roc_auc, "\n",
      "FPR", fpr)

## 교차표 시각화 ##
cbt_cm = confusion_matrix(y_resampled_test, cbt_pred)
group_names = ["TN", "FP", "FN", "TP"]
group_counts = [value for value in cbt_cm.flatten()]
group_percentages = [f"{value: .4%}" for value in cbt_cm.flatten() / np.sum(cbt_cm)]
labels = [f"{name}\n{count}\n{percent}" for name, count, percent in zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)

sns.heatmap(cbt_cm, annot=labels, fmt='', cmap='Blues')
plt.title('Confusion Matrix(CatBoost)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

## TopK 성능평가 ##
results = []
k_list = [30, 150, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]

for k in k_list:
    thresh_k = round(threshold_at_k(y_resampled_test, cbt_pred_proba, k), 4)
    prec_k = round(precision_at_k(y_resampled_test, cbt_pred_proba, k), 4)
    rec_k = round(recall_at_k(y_resampled_test, cbt_pred_proba, k), 4)
    f1_k = round(f1_score_at_k(y_resampled_test, cbt_pred_proba, k), 4)

    results.append({'k': k, 'thresh': thresh_k, 'pre_at_k': prec_k, 'rec_at_k': rec_k, 'f1_at_k': f1_k})

results = pd.DataFrame(results)

print("TopK 성능지표 확인")
print(results)

## 성능지표 시각화 - TopK ##
plt.figure(figsize=(15, 6))
plt.plot(results['k'], results['pre_at_k'], label='precision@k', marker='o', markersize=4, linewidth=1.2)
plt.plot(results['k'], results['rec_at_k'], label='recall@k', marker='^', markersize=4, linewidth=1.2)
plt.plot(results['k'], results['f1_at_k'], label='f1@k', marker='s', markersize=4, linewidth=1.2)
plt.title("Performance Metrics(Top K evaluations)", fontsize=15)
plt.xlabel("k", fontsize=13)
plt.ylabel("Metric Score", fontsize=13)
plt.legend(title="Metrics")
plt.tick_params(axis="both", labelsize=14)
plt.axvspan(150, 300, color='red', alpha=0.3)
plt.text(150, -0.1, '150', ha='center', va='bottom', color='red', fontweight='bold', rotation=45, fontsize=14)
plt.text(300, -0.1, '300', ha='center', va='bottom', color='red', fontweight='bold', rotation=45, fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, which='major', axis='x', linestyle='--', alpha=0.7)
plt.show()

## 성능지표 시각화 - threshold ##
plt.figure(figsize=(15, 6))
plt.plot(results['k'], results['thresh'], label='thresh@k', marker='o', linewidth=2)
plt.title("Threshold@K", fontsize=15)
plt.xlabel("k", fontsize=13)
plt.ylabel("thresholds", fontsize=13)
plt.tick_params(axis='both', labelsize=14)
plt.axvspan(150, 300, color='red', alpha=0.3)
plt.text(150, 0.1, '150', ha='center', va='bottom', color='red', fontweight='bold', rotation=45, fontsize=14)
plt.text(300, 0.1, '300', ha='center', va='bottom', color='red', fontweight='bold', rotation=45, fontsize=14)
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, which='major', axis='x', linestyle='--', alpha=0.7)
plt.show()
