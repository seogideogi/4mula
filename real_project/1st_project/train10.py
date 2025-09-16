import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, precision_score, roc_auc_score
from model11 import GATImproved
from dataset11 import AMLtoGraph
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = AMLtoGraph("./lej_dataset_002")
train_loader = DataLoader([dataset.data_train], batch_size=1, shuffle=True)
val_loader = DataLoader([dataset.data_val], batch_size=1, shuffle=False)

model = GATImproved(
    in_channels=dataset.data_train.num_node_features,
    edge_dim=dataset.data_train.edge_attr.shape[1],
    out_channels=1
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
loss_fn = torch.nn.BCEWithLogitsLoss()

best_f1 = 0
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

    model.eval()
    preds, labels, dates = [], [], []
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr).view(-1)
            score = torch.sigmoid(out).cpu().numpy()
            label = data.y.cpu().numpy()
            date = data.x[:, 0].cpu().numpy()
            preds.extend(score)
            labels.extend(label)
            dates.extend(date)

    bin_preds = (np.array(preds) > 0.5).astype(int)
    f1 = f1_score(labels, bin_preds)
    precision = precision_score(labels, bin_preds)
    roc = roc_auc_score(labels, preds)

    print(f"Epoch {epoch:02d} | Loss: {total_loss:.4f}")
    print(f"[Val] F1: {f1:.4f}, Precision: {precision:.4f}, ROC-AUC: {roc:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "best_model.pth")
        np.save("val_preds.npy", preds)
        np.save("val_labels.npy", labels)
        np.save("val_tran_dt.npy", dates)
        print("Best model and predictions saved.")
