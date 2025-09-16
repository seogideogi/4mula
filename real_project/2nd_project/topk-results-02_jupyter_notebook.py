import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from model import GATImproved


device = torch.device(cuda if torch.cuda.is_available() else cpu)
print(Device, device)

# 1) data.pt 직접 로드
data_path = os.path.join(.lej_dataset_002, processed, data.pt)
data_train, _, data_val, _, data_test, _ = torch.load(data_path, map_location=device)

# 2) 모델 로드 (train과 동일한 하이퍼 파라미터 차원 사용)
in_channels = data_train.num_node_features
edge_dim    = data_train.edge_attr.shape[1]
model = EdgeGAT(in_channels=in_channels, edge_dim=edge_dim, out_channels=1).to(device)

state = torch.load(best_model.pth, map_location=device)
model.load_state_dict(state)
model.eval()

def run_and_save(split_name, data_obj)
    if data_obj is None
        print(f[{split_name}] 없음, skip)
        return
    loader = DataLoader([data_obj], batch_size=1, shuffle=False)  # 단일 그래프
    preds, labels = [], []
    with torch.no_grad()
        for batch in loader
            batch = batch.to(device)
            logit = model(batch.x, batch.edge_index, batch.edge_attr).view(-1)
            preds.extend(torch.sigmoid(logit).cpu().numpy())
            labels.extend(batch.y.cpu().numpy())

    preds = np.asarray(preds, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    np.save(f{split_name}_preds.npy, preds)
    np.save(f{split_name}_labels.npy, labels)
    pd.DataFrame({score preds, label labels}).to_csv(f{split_name}_predictions.csv, index=False)
    print(f[{split_name}] saved {split_name}_preds.npy {preds.shape}, {split_name}_labels.npy {labels.shape})

# 검증테스트 예측 생성
run_and_save(val,  data_val)
run_and_save(test, data_test)

#-----------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

def precision_at_k(y_true, y_score, k)
    idx = np.argsort(y_score)[-1][k]
    return (y_true[idx].sum()  k) if k  0 else 0.0

def recall_at_k(y_true, y_score, k)
    idx = np.argsort(y_score)[-1][k]
    pos = y_true.sum()
    return (y_true[idx].sum()  pos) if pos  0 else 0.0

def f1_at_k(y_true, y_score, k)
    p = precision_at_k(y_true, y_score, k)
    r = recall_at_k(y_true, y_score, k)
    return (2pr  (p+r+1e-8)) if (p+r)  0 else 0.0

def threshold_at_k(y_score, k)
    s = np.sort(y_score)[-1]
    return s[k-1] if 1 = k = len(s) else (s[-1] if len(s) else 0.0)
    
    
    
    
    

def analyze_split(split_name, k_list=None)
    if k_list is None
        k_list = [30, 150, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]

    # 1) 예측 결과 로드
    y_score = np.load(f{split_name}_preds.npy)
    y_true  = np.load(f{split_name}_labels.npy)

    # 2) 메트릭 계산
    precision_list = [precision_at_k(y_true, y_score, k) for k in k_list]
    recall_list    = [recall_at_k(y_true, y_score, k)    for k in k_list]
    f1_list        = [f1_at_k(y_true, y_score, k)        for k in k_list]
    thr_list       = [threshold_at_k(y_score, k)         for k in k_list]

    # 3) 그래프 1 PrecisionRecallF1@K
    plt.figure(figsize=(12,5))
    plt.plot(k_list, precision_list, marker='o', label='precision@k')
    plt.plot(k_list, recall_list,    marker='^', label='recall@k')
    plt.plot(k_list, f1_list,        marker='s', label='f1@k')
    plt.title(f{split_name.upper()} - Top-K Metrics)
    plt.xlabel(k); plt.ylabel(score); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()

    # 4) 그래프 2 Thresholds by K
    plt.figure(figsize=(10,3))
    plt.plot(k_list, thr_list, marker='o')
    plt.title(f{split_name.upper()} - Thresholds by K)
    plt.xlabel(k); plt.ylabel(threshold); plt.grid(True); plt.tight_layout()
    plt.show()

    # 5) 수치 출력 (테이블 형태)
    df_metrics = pd.DataFrame({
        K k_list,
        Precision@K precision_list,
        Recall@K recall_list,
        F1@K f1_list,
        Threshold@K thr_list
    })
    print(fn=== {split_name.upper()} 상세 결과 ===)
    print(df_metrics.to_string(index=False, float_format=lambda x f{x.4f}))

    # 6) @0.5 기준 이진 분류 지표도 출력
    y_bin = (y_score  0.5).astype(int)
    print(fn[{split_name}] @0.5  Precision={precision_score(y_true, y_bin).4f} 
          fRecall={recall_score(y_true, y_bin).4f} 
          fF1={f1_score(y_true, y_bin).4f} 
          fROC-AUC={roc_auc_score(y_true, y_score).4f})

# 실행 예시
analyze_split(val)
analyze_split(test)

