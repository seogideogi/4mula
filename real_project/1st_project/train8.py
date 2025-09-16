import torch
import numpy as np
from model3 import GAT
from dataset8 import AMLtoGraph
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score, roc_auc_score, precision_score

#  1. Device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

#  2. 데이터셋 로드
dataset = AMLtoGraph('./lej_dataset_002')
train_data = dataset.data_train.to(device)
val_data = dataset.data_val.to(device)
test_data = dataset.data_test.to(device)

#  3. 모델 정의
model = GAT(
    in_channels=train_data.num_features,
    edge_dim=train_data.edge_attr.shape[1],
    hidden_channels=16,
    out_channels=1,
    heads=8
).to(device)

#  4. 손실 함수 및 옵티마이저 설정 (불균형 보정)
# fraud_ratio = (train_data.y == 1).sum().item() / train_data.y.shape[0]
# pos_weight = torch.tensor([1.0 / fraud_ratio]).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

#  5. NeighborLoader 정의
train_loader = NeighborLoader(train_data, num_neighbors=[30]*2, batch_size=256, shuffle=True)
val_loader = NeighborLoader(val_data, num_neighbors=[30]*2, batch_size=256, shuffle=False)
test_loader = NeighborLoader(test_data, num_neighbors=[30]*2, batch_size=256, shuffle=False)

#  6. 학습 루프
num_epochs = 30
best_f1 = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr).view(-1)
        loss = criterion(pred, batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch:03d} | Loss: {total_loss:.4f}")

    #  7. Validation 평가
    model.eval()
    all_preds, all_labels, all_dates, all_tmrg = [], [], [], []

    with torch.no_grad():
        for batch in val_loader:
            batch.to(device)
            out = torch.sigmoid(model(batch.x, batch.edge_index, batch.edge_attr).view(-1))
            all_preds.append(out.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
            all_dates.append(batch.x[:, 0].cpu().numpy())  # tran_dt
            all_tmrg.append(batch.x[:, 2].cpu().numpy())   # tran_tmrg

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_dates = np.concatenate(all_dates)
    all_tmrg = np.concatenate(all_tmrg)

    f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), average='binary')
    precision = precision_score(all_labels, (all_preds > 0.5).astype(int), average='binary')
    roc_auc = roc_auc_score(all_labels, all_preds)

    print(f"[Validation] F1: {f1:.4f}, Precision: {precision:.4f}, ROC-AUC: {roc_auc:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "gat_model.pth")
        np.save("all_preds.npy", all_preds)
        np.save("all_labels.npy", all_labels)
        np.save("tran_dates.npy", all_dates)
        np.save("tran_tmrg.npy", all_tmrg)
        print("Best model & predictions saved!")

#  8. Test 데이터셋 평가
print("\n[Test Set Evaluation]")
model.eval()
all_preds, all_labels, all_dates, all_tmrg = [], [], [], []

with torch.no_grad():
    for batch in test_loader:
        batch.to(device)
        out = torch.sigmoid(model(batch.x, batch.edge_index, batch.edge_attr).view(-1))
        all_preds.append(out.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
        all_dates.append(batch.x[:, 0].cpu().numpy())  # tran_dt
        all_tmrg.append(batch.x[:, 2].cpu().numpy())   # tran_tmrg

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
all_dates = np.concatenate(all_dates)
all_tmrg = np.concatenate(all_tmrg)

f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), average='binary')
precision = precision_score(all_labels, (all_preds > 0.5).astype(int), average='binary')
roc_auc = roc_auc_score(all_labels, all_preds)

print(f"[Test] F1: {f1:.4f}, Precision: {precision:.4f}, ROC-AUC: {roc_auc:.4f}")
