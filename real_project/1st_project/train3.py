import torch
import numpy as np
from model import GAT
from dataset import AMLtoGraph
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score

# GPU/CPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터셋 로드
dataset = AMLtoGraph('./lej_dataset_002')
data = dataset[0].to(device)

# 사기 계좌 비율 확인 (불균형 데이터 보정)
fraud_ratio = (data.y == 1).sum().item() / data.y.shape[0]
pos_weight = torch.tensor([1.0 / fraud_ratio]).to(device)  # 불균형 보정

# 모델 초기화 (edge_attr 활용)
model = GAT(
    in_channels=data.num_features,
    edge_dim=data.edge_attr.shape[1], 
    hidden_channels=16,
    out_channels=1,
    heads=8
).to(device)

# 손실 함수 및 옵티마이저 설정
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # 불균형 보정 적용
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

# 학습/검증 데이터 분할
split = T.RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0)
data = split(data)

# 데이터 로더 설정
train_loader = NeighborLoader(data, num_neighbors=[30] * 2, batch_size=256, shuffle=True)
test_loader = NeighborLoader(data, num_neighbors=[30] * 2, batch_size=256, shuffle=False)

# Top-K 평가 함수
def top_k_indices(y_score, k):
    """ 예측 점수(y_score)에서 상위 K개 인덱스 가져오기 """
    return np.argsort(y_score)[-k:]

def precision_at_k(y_true, y_score, k=30):
    """ Precision@K 계산: 상위 K개 예측 중 실제 사기 비율 """
    top_k_idx = top_k_indices(y_score, k)
    y_top_k_pred = (y_score[top_k_idx] > 0.5).astype(int)  
    y_top_k_true = y_true[top_k_idx]
    return precision_score(y_top_k_true, y_top_k_pred, average='binary')

def accuracy_at_k(y_true, y_score, k=30):
    """ Accuracy@K 계산: 상위 K개 예측 중 맞춘 개수 """
    top_k_idx = top_k_indices(y_score, k)
    y_top_k_pred = (y_score[top_k_idx] > 0.5).astype(int)
    y_top_k_true = y_true[top_k_idx]
    return accuracy_score(y_top_k_true, y_top_k_pred)

# 학습 루프 설정
num_epochs = 30  
best_f1 = 0  # 최고 성능 저장

for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    
    for batch in train_loader:
        optimizer.zero_grad()
        batch.to(device)
        
        # 모델 예측 (sigmoid 미적용)
        pred = model(batch.x, batch.edge_index, batch.edge_attr).view(-1)  
        
        # 손실 계산 (이진 분류)
        loss = criterion(pred, batch.y.float())  
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 🔹 10 epoch마다 평가 실행
    if epoch % 10 == 0:
        print(f'Epoch {epoch:03d} | Loss: {total_loss:.4f}')
        model.eval()
        
        with torch.no_grad():
            all_preds, all_labels = [], []
            
            for batch in test_loader:
                batch.to(device)
                
                # 모델 예측 수행 (sigmoid 적용)
                pred = torch.sigmoid(model(batch.x, batch.edge_index, batch.edge_attr).view(-1))
                all_preds.append(pred.cpu().numpy())
                all_labels.append(batch.y.cpu().numpy())

            # 전체 예측값 및 실제 정답값 정리
            all_preds = np.concatenate(all_preds)  # 확률 값
            all_labels = np.concatenate(all_labels)  # 이미 0과 1로 되어 있음 (변환 불필요)

            # 기본 평가 지표 (이진 분류 기준)
            f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), average='binary')
            precision = precision_score(all_labels, (all_preds > 0.5).astype(int), average='binary')
            roc_auc = roc_auc_score(all_labels, all_preds)  # 확률 값 기준으로 ROC-AUC 계산

            # 추가된 Top-K 평가 지표 (Top-30 기준, 확률 기준으로 평가)
            top_k_acc = accuracy_at_k(all_labels, all_preds, k=30)
            top_k_prec = precision_at_k(all_labels, all_preds, k=30)

            # 결과 출력
            print(f' F1-score: {f1:.4f}, Precision: {precision:.4f}, ROC-AUC: {roc_auc:.4f}')
            print(f' Top-30 Accuracy: {top_k_acc:.4f}, Top-30 Precision: {top_k_prec:.4f}')
