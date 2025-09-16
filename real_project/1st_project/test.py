# test_eval.py
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from model13 import GATImproved
from dataset16 import AMLtoGraph

def precision_at_k(y_true, y_pred_proba, k):
    top_k_idx = np.argsort(y_pred_proba)[::-1][:k]
    return np.sum(y_true[top_k_idx]) / k

def recall_at_k(y_true, y_pred_proba, k):
    top_k_idx = np.argsort(y_pred_proba)[::-1][:k]
    return np.sum(y_true[top_k_idx]) / np.sum(y_true)

# 환경 설정
batch_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 저장된 data.pt 로드만 수행
data_path = "./lej_dataset_002/processed/data.pt"
_, _, _, _, data_test, _ = torch.load(data_path)
test_loader = DataLoader([data_test], batch_size=batch_size)

# 모델 로드
model = GATImproved(
    in_channels=data_test.num_node_features,
    edge_dim=data_test.edge_attr.shape[1],
    out_channels=1
).to(device)

model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# 예측
test_preds, test_labels = [], []
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr).view(-1)
        test_preds.extend(torch.sigmoid(out).cpu().numpy())
        test_labels.extend(data.y.cpu().numpy())

test_preds = np.array(test_preds)
test_labels = np.array(test_labels)

# 저장
np.save("test_preds.npy", test_preds)
np.save("test_labels.npy", test_labels)
np.save("test_x.npy", data_test.x.cpu().numpy())  # community, degree 등 분석용
print("test_x.npy 저장 완료!")
df_test = pd.DataFrame({"score": test_preds, "label": test_labels})
df_test.to_csv("test_predictions.csv", index=False)
print("Test 저장 완료!")

# 평가
binary_preds = (test_preds > 0.5).astype(int)
print(f"[Test] F1: {f1_score(test_labels, binary_preds):.4f}")
print(f"[Test] Precision: {precision_score(test_labels, binary_preds):.4f}")
print(f"[Test] Recall: {recall_score(test_labels, binary_preds):.4f}")
print(f"[Test] ROC AUC: {roc_auc_score(test_labels, test_preds):.4f}")

# Top-K 성능
print("\n[Top-K 성능]")
for k in [30, 150, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]:
    p_k = precision_at_k(test_labels, test_preds, k)
    r_k = recall_at_k(test_labels, test_preds, k)
    print(f"Top-{k:4d} | Precision: {p_k:.4f} | Recall: {r_k:.4f}")
