# -*- coding: utf-8 -*-

###### 라이브러리 불러오기 ######
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from zoneinfo import ZoneInfo
from imblearn.under_sampling import RandomUnderSampler
import networkx as nx
from networkx.algorithm import community

from catboost import CatBoostClassifier, metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, make_scorer, confusion_matrix

###### 데이터 불러오기 #######
df_new = pd.read_csv('hf_trns_tran_new_12norm.csv', dtype = {'ff_sp_ai': str} )
df_new.info()

## 데이터 복사 ##
df = df_new.copy()

## 데이터 결측치 확인 ##
df.isna().sum()

## 사기건수 확인(0 또는 SP) ##
df['ff_sp_ai'].value_counts(dropna=False)

## 사기건수 확인(SP -> 1) ##
def sp_or_not(x):
	if x == '0':
		return 0
	else:
	  return 1

df['ff_sp_ai'] = df['ff_sp_ai'].apply(sp_or_not)
df['ff_sp_ai'].value_counts()

###### 데이터 전처리 및 분리 ######
## 자료타입 변경 ##
df['tran_dt'] = pd.to_datetime(df['tran_dt'])   # 정수 -> 날짜

# 정수 -> 카테고리
df['wd_fc_ac'] = df['wd_fc_ac'].astype('category')
df['dps_fc_ac'] = df['dps_fc_ac'].astype('category')
df['md_type'] = df['md_type'].astype('category')
df['fnd_type'] = df['fnd_type'].astype('category')

# month 분리 - 훈련, 검증, 테스트 데이터로 분리할 때 사용
df['month'] = df['tran_dt'].dt.month

# 날짜를 초단위로 변환(이후 정규화진행)
df['tran_dt_seconds'] = df['tran_dt'].astype('int64') // 10**9

## 10000원 이상 금액 적용 ##
df_all = df.copy()
df = df[df['tran_amt'] >= 10000]

## feature, target
features = ['wd_fc_ac', 'dps_fc_ac', 'md_type', 'fnd_type', 'tran_amt',
						'tran_dt_seconds', 'tran_tmrg',
						'wd_fc_ac_fnd_cnt', 'wd_fc_ac_fnd_amt',
						'wd_fc_ac_md_cnt', 'wd_fc_ac_md_amt',
						'dps_fc_ac_fnd_cnt', 'dps_fc_ac_fnd_amt',
						'dps_fc_ac_md_amt', 'dps_fc_ac_md_cnt',
						'prev_dps_fraud_cnt', 'prev_wd_fraud_cnt']
target = 'ff_sp_ai'

## 데이터 나누기 ##
X_train = df.loc[(df['month'] >= 1) & (df['month'] <= 10), features]
X_val = df.loc[df['month'] == 11, features]
X_test = df.loc[df['month'] == 12, features]

y_train = df.loc[(df['month'] >= 1) & (df['month'] <= 10), target]
y_val = df.loc[df['month'] == 11, target]
y_test = df.loc[df['month'] == 12, target]

print("훈련데이터: ", X_train.shape, y_train.shape)
print("검증데이터: ", X_val.shape, y_val.shape)
print("시험데이터: ", X_test.shape, y_test.shape)

## 날짜 데이터 정규화 ##
scaler = MinMaxScaler()
X_train.loc[:, 'tran_dt_seconds'] = scaler.fit_transform(X_train[['tran_dt_seconds']])
X_val.loc[:, 'tran_dt_seconds'] = scaler.transform(X_val[['tran_dt_seconds']])
X_test.loc[:, 'tran_dt_seconds'] = scaler.transform(X_test[['tran_dt_seconds']])

## 커뮤니티 피처 만들기 ##
def detect_communities(df)"
	print("커뮤니티 탐지(Train 기반)")
	G = nx.Graph()
	for _, row in df.iterrows():
		G.add_edge(row['wd_fc_ac'], row['dps_fc_ac'], weight=row['tran_amt'])

	communities = community.louvain_communities(G, weight='weight')
	community_map = {}
	for idx, nodes in enumerate(communities):
		for node in nodes:
			community_map[node] = idx

	return community_map

# 커뮤니티 피처를 기존 데이터에 추가
community_map = detect_communities(X_train)

X_train['community'] = X_train['wd_fc_ac'].map(community_map).fillna(0)
X_val['community'] = X_val['wd_fc_ac'].map(community_map).fillna(0)
X_test['community'] = X_test['wd_fc_ac'].map(community_map).fillna(0)

## Degree 차수 만들기 ##
def add_degree_feature(df):
	print("Degree 계산(Train 기반)")
	G = nx.Graph()
	G.add_edges_from(zip(df['wd_fc_ac'], df['dps_fc_ac']))
	degree_dict = dict(G.degree())
	return degree_dict

# Degree 차수를 기존데이터에 추가
degree_map = add_degree_feature(X_train)

X_train['degree_dps'] = X_train['dps_fc_ac'].map(degree_map).fillna(0)
X_val['degree_dps'] = X_val['dps_fc_ac'].map(degree_map).fillna(0)
X_test['degree_dps'] = X_test['dps_fc_ac'].map(degree_map).fillna(0)

###### 언더샘플링 ######

## RandomUnderSampling - 훈련데이터 ##
ratio = 0.01

# 의심거래개수
num_minority = np.sum(y_train == 1)

# 정상거래의 목표 개수 설정
num_majority = int(np.sum(y_train == 0)*ratio)

# RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy = {0:num_majority, 1:num_minority}, random_state=42)
X_resampled_tr, y_resampled_tr = rus.fit_resample(X_train, y_train)

# 데이터 개수 확인
print(f"정상거래 샘플링비율 {ratio:.2f}, 정상거래건수: {np.sum(y_resampled_tr ==0)}, 사기거래건수: {np.sum(y_resampled_tr ==1)}")

## RandomUnderSampling - 검증데이터 ##

# 의심거래개수
num_minority = np.sum(y_val == 1)

# 정상거래의 목표 개수 설정
num_majority = int(np.sum(y_val == 0)*ratio)

# RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy = {0:num_majority, 1:num_minority}, random_state=42)
X_resampled_val, y_resampled_val = rus.fit_resample(X_val, y_val)

# 데이터 개수 확인
print(f"정상거래 샘플링비율 {ratio:.2f}, 정상거래건수: {np.sum(y_resampled_val ==0)}, 사기거래건수: {np.sum(y_resampled_val ==1)}")

###### TopK 성능지표정의 ######

## 모델 예측 후 평가하기 위한 TopK 성능지표 구현 ##

def precision_at_k(y_true, y_pred_proba, k):
	"""Precision@K 계산"""

	all_labels = np.column_stack((y_test, cbt_pred_proba))

	# 확률 기준으로 상위 k개 인덱스 선택
	top_k_indices = np.argsort(all_labels[:, 1])[::-1][:k]
	top_k_true = all_labels[top_k_indices, 0]

	return np.sum(top_k_true) / k

def recall_at_k(y_true, y_pred_proba, k):
	"""Recall@K 계산"""

	all_labels = np.column_stack((y_test, cbt_pred_proba))

	top_k_indices = np.argsort(all_labels[:, 1])[::-1][:k]
	top_k_true = all_labels[top_k_indices, 0]
	total_positives = np.sum(y_true)

	return np.sum(top_k_true) / total_positives

def threshold_at_k(y_true, y_pred_proba, k):
	"""threshold@K 계산"""

	all_labels = np.column_stack((y_test, cbt_pred_proba))

	top_k_indices = np.argsort(all_labels[:, 1])[::-1][:k]
	threshold_index = top_k_indices[-1]
	threshold = all_labels[threshold_index, 1]

	return threshold

def f1_score_at_k(y_true, y_pred_proba, k=300):
	"""f1_score@K 계산"""

	all_labels = np.column_stack((y_true, y_pred_proba))

	top_k_indices = np.argsort(all_labels[:, 1])[::-1][:k]
	top_k_true = all_labels[top_k_indices, 0]
	total_positives = np.sum(y_true)
	precision_at_k = np.sum(top_k_true) / k
	recall_at_k = np.sum(top_k_true) / total_positives

	if precision_at_k + recall_at_k == 0:
		return 0

	f1_score_at_k = 2*(precision_at_k*recall_at_k) / (precision_at_k + recall_at_k)

	return f1_score_at_k

## CatBoost에서 모델 훈련 때 사용할 F1@300 평가함수 구현 ##

class eval_f1_score_at_k():
	def get_final_error(self, error, weight):
		return error / (weight + 1e-38)

	def is_max_optimal(self):
		return True

	def evaluate(self, approxes, target, weight=None):
		assert len(approxes) == 1
		assert len(target) == len(approxes[0])

		k=300

		y_pred_proba = 1 / (1 + np.exp(-np.array(approxes[0])))
		all_labels = np.column_stack((target, y_pred_proba))

		top_k_indice = np.argsort(all_labels[:, 1])[::-1][:k]
		top_k_true = all_labels[top_k_indice, 0]
		total_positive = np.sum(target == 1)

		precision_at_k = np.sum(top_k_true == 1) / k
		recall_at_k = np.sum(top_k_true == 1) / total_positive

		if precision_at_k + recall_at_k == 0:
			return 0, True

		f1_score_at_k = 2*(precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)

		return f1_score_at_k, True

###### CatBoost 모델 훈련 ######
## CatBoost

# CatBoostClassifier 하이퍼파라미터 설정
cat_params = {'loss_function': 'Logloss',
							'random_seed': 0,
							'iterations': 100,
							'learning_rate': 0.1,
							'verbose': 10,
							'cat_features': ['wd_fc_ac', 'dps_fc_ac', 'md_type', 'fnd_type']}

# CatBoost 모델 생성 및 학습
cbt = CatBoostClassifier(**cat_params, eval_metric = eval_f1_score_at_k())
cbt.fit(X_resampled_tr, y_resampled_tr, eval_set = [(X_resampled_val, y_resampled_val)], use_best_model=True)

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
cbt_pred_proba = cbt.predict_proba(X_test)[:,1]

# 평가 지표 계산
acc = round(accuracy_score(y_test, cbt_pred), 4)
prec = round(precision_score(y_test, cbt_pred, zero_division = 0), 4)
rec = round(recall_score(y_test, cbt_pred), 4)
f1 = round(f1_score(y_test, cbt_pred), 4)
roc_auc = round(roc_auc_score(y_test, cbt_pred_proba), 4)

# 혼동 행렬 계산
tn, fp, fn, tp =  confusion_matrix(y_test, cbt_pred).ravel()

# FPR 계산
fpr = round(fp / (fp+tn), 6)

# 결과
print("Accuracy", acc, "\n",
			"Precision", prec, "\n",
			"Recall", rec, "\n",
			"F1-score", f1, "\n",
			"ROC-AUC", roc_auc, "\n",
			"FPR", fpr)

## 교차표 시각화 ##
cbt_cm = confusion_matrix(y_test, cbt_pred)
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
	thresh_k = round(threshold_at_k(y_test, cbt_pred_proba, k), 4)
	prec_k = round(precision_at_k(y_test, cbt_pred_proba, k), 4)
	rec_k = round(recall_at_k(y_test, cbt_pred_proba, k), 4)
	f1_k = round(f1_score_at_k(y_test, cbt_pred_proba, k), 4)

	results.append({'k':k, 'thresh': thresh_k, 'pre_at_k':prec_k, 'rec_at_k':rec_k, 'f1_at_k':f1_k})


results = pd.DataFrame(results)

print("TopK 성능지표 확인")
results

## 성능지표 시각화 - TopK ##

plt.figure(figsize=(15, 6))

plt.plot(results['k'], results['pre_at_k']. label='precision@k', marker = 'o', markersize=4, linewidth=1.2)
plt.plot(results['k'], results['rec_at_k']. label='recall@k', marker = '^', markersize=4, linewidth=1.2)
plt.plot(results['k'], results['f1_at_k']. label='f1@k', marker = 's', markersize=4, linewidth=1.2)

plt.title("Performance Metrics(Top K evaluations)", fontsize = 15)
plt.xlabel("k", fontsize=13)
plt.ylabel("Metric Score", fontsize=13)
plt.legend(title="Metrics")
plt.tick_params(axis="both", labelsize=14)

# 평가점수가 높은 구간 표시
plt.axvspan(150, 300, color='red', alpha=0.3)
plt.text(150, -0.1, '150', ha='center', va='bottom', color='red', fontweight= 'bold', rotation=45, fontsize=14)
plt.text(300, -0.1, '300', ha='center', va='bottom', color='red', fontweight= 'bold', rotation=45, fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, which='major',axis='x', linestyle='--', alpha=0.7)

plt.show()

## 성능지표 시각화 - threshold ##

plt.figure(figsize=(15, 6))
plt.plot(results['k'], results['thresh'], label='thresh@k', marker='o', linewidth=2, color='Purple')

# 그래프 제목 및 라벨 설정
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
