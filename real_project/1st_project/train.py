# train.py
import torch
from model import GAT # model.py에서 정의한 Graph Attention Network(GAT) 모델.
from dataset import AMLtoGraph # dataset.py에서 정의한 데이터셋 클래스.
import torch_geometric.transforms as T # 그래프 데이터 변환을 위한 도구.
from torch_geometric.loader import NeighborLoader #그래프 데이터를 미니배치로 로드하기 위한 데이터 로더.

################################################################ 2025.02.19 ################################################################
# networkX 중심성계좌 함수 추가 1개
# train.py 에서 GAT 모델을 생성할 때, 변경된 입력 채널 수를 반영해야 함.
# dataset.py를 수정한 후 train.py 를 처음 실행하면, dataset.process() 함수가 실행되어 data.pt 파일이 다시 생성되고, 
# data.num_features 값이 업데이트 됨. 이 업데이트된 data.num_features 값을 GAT 모델 생성 시 in_channels 로 사용해야 함.
############################################################################################################################################
# 다운샘플링 추가 - 중심성 계좌 함수 추가 후 끝나지 않음.
# 
############################################################################################################################################
# 새로 실행 할때, data.pt 삭제 하고 실행 할 것 : 수정된 dataset.py 의 process() 함수가 다시 실행되어 새로운 중심성 Feature가 포함된 data.pt 파일이 생성 됨.
############################################################################################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # GPU or CPU 선택
dataset = AMLtoGraph('../../../../archive') # AMLtoGraph 클래스를 사용하여 데이터셋을 로드.
data = dataset[0] # 데이터셋의 첫 번째 그래프 데이터를 로드.
epoch = 100

# 2025.02.19 networkX 여기 수정 하라는데, 똑같다 ? 
model = GAT(in_channels=data.num_features, hidden_channels=16, out_channels=1, heads=8) 
# Graph Attention Network(GAT) 모델을 초기화.
## in_channels: 입력 특성의 차원 (노드 특성 수).
## hidden_channels: 은닉층의 차원.
## out_channels: 출력 차원 (이진 분류 문제이므로 1).
## heads: 멀티헤드 어텐션의 헤드 수

model = model.to(device) # 모델을 GPU 또는 CPU로 이동.

criterion = torch.nn.BCELoss() # 이진 분류를 위한 Binary Cross Entropy Loss(BCELoss).
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001) # Stochastic Gradient Descent(SGD) 옵티마이저.


# 학습데이터 분할
split = T.RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0)
data = split(data)
# RandomNodeSplit: 노드를 무작위로 학습, 검증, 테스트 세트로 분할.
# 	split='train_rest': 학습 세트와 검증 세트로 분할.
# 	num_val=0.1: 검증 세트로 10%의 노드를 사용.
# 	num_test=0: 테스트 세트는 사용하지 않음.

train_loader = NeighborLoader(
    data,
    num_neighbors=[30] * 2,
    batch_size=256,
    input_nodes=data.train_mask,
)

test_loader = NeighborLoader(
    data,
    num_neighbors=[30] * 2,
    batch_size=256,
    input_nodes=data.val_mask,
)
# NeighborLoader: 그래프 데이터를 미니배치로 로드.
# 	num_neighbors=[30] * 2: 각 노드의 이웃을 2-hop까지 30개씩 샘플링.
# 	batch_size=256: 배치 크기를 256으로 설정.
# 	input_nodes: 학습 또는 검증에 사용할 노드 마스크.

# 학습루프
## epoch: 전체 데이터셋을 몇 번 반복할지 설정.
for i in range(epoch):
    total_loss = 0
    model.train() # 모델을 학습 모드로 설정.
    for data in train_loader:
        optimizer.zero_grad() # 그래디언트를 초기화.
        data.to(device) # 데이터를 GPU 또는 CPU로 이동.
        pred = model(data.x, data.edge_index, data.edge_attr) # 모델을 통해 예측값을 계산.
        ground_truth = data.y
        loss = criterion(pred, ground_truth.unsqueeze(1)) # 예측값과 실제값 사이의 손실을 계산.
        loss.backward() #역전파를 통해 그래디언트 계산.
        optimizer.step() # 옵티마이저를 통해 모델 파라미터 업데이트
        total_loss += float(loss) # 에포크별 총 손실을 누적.

    # 검증 및 정확도 계산    
    if epoch%10 == 0:
        print(f"Epoch: {i:03d}, Loss: {total_loss:.4f}")
        model.eval() # 모델을 평가 모드로 설정.
        acc = 0
        total = 0

        # torch.no_grad() : 그래디언트 계산을 비활성화.
        with torch.no_grad():
            for test_data in test_loader:
                test_data.to(device)
                pred = model(test_data.x, test_data.edge_index, test_data.edge_attr) # 검증 데이터에 대해 예측값을 계산.
                ground_truth = test_data.y
                correct = (pred == ground_truth.unsqueeze(1)).sum().item() # 예측값과 실제값이 일치하는지 확인.
                total += len(ground_truth)
                acc += correct # 정확도를 계산하고 출력.
            acc = acc/total
            print('accuracy:', acc)