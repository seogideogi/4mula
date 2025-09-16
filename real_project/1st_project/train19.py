import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from model13 import GATImproved
from dataset16 import AMLtoGraph
import numpy as np
import pandas as pd

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
loss_fn = FocalLoss(alpha=0.95, gamma=1, reduction='mean').to(device)
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
        