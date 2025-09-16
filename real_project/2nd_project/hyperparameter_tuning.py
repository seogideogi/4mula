import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from model import EdgeGATOptimized, EdgeGATLightweight
# EdgeGAT는 EdgeGATOptimized의 alias로 정의됨
from dataset import AMLtoGraph
import numpy as np
import pandas as pd
from itertools import product
import json
from datetime import datetime
import traceback
import warnings
warnings.filterwarnings('ignore')

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.95, gamma=1, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# --------- Top-K 평가 함수 ----------
def precision_at_k(y_true, y_pred_proba, k):
    if len(y_true) < k:
        k = len(y_true)
    if k == 0:
        return 0.0
    top_k_idx = np.argsort(y_pred_proba)[::-1][:k]
    return np.sum(y_true[top_k_idx]) / k

def recall_at_k(y_true, y_pred_proba, k):
    if len(y_true) < k:
        k = len(y_true)
    if k == 0:
        return 0.0
    top_k_idx = np.argsort(y_pred_proba)[::-1][:k]
    total_positive = np.sum(y_true)
    if total_positive == 0:
        return 0.0
    return np.sum(y_true[top_k_idx]) / total_positive

def f1_score_at_k(y_true, y_pred_proba, k):
    precision_k = precision_at_k(y_true, y_pred_proba, k)
    recall_k = recall_at_k(y_true, y_pred_proba, k)
    
    if precision_k + recall_k == 0:
        return 0.0
    
    f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
    return f1_k

def cleanup_gpu_memory():
    """GPU 메모리 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def train_and_evaluate_safe(params, dataset, device, k=150, max_epochs=20, patience=3):
    """
    완전 자동화된 안전한 학습 및 평가 함수
    """
    combination_id = f"h{params['hidden_channels']}_head{params['heads']}_lr{params['lr']}"
    
    try:
        # GPU 메모리 정리
        cleanup_gpu_memory()
        
        # 데이터 로더
        train_loader = DataLoader([dataset.data_train], batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader([dataset.data_val], batch_size=params['batch_size'])
        
        # 모델 초기화
        model_class = params.get('model_class', EdgeGATOptimized)
        
        model = model_class(
            node_feat_dim=dataset.data_train.x.shape[1],
            edge_feat_dim=dataset.data_train.edge_attr.shape[1],
            hidden_channels=params['hidden_channels'],
            heads=params['heads']
        ).to(device)
        
        # 학습 설정
        focal_alpha = params.get('focal_alpha', 0.95)
        focal_gamma = params.get('focal_gamma', 1)
        weight_decay = params.get('weight_decay', 1e-4)
        
        loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean').to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
        
        best_f1 = 0.0
        best_metrics = {}
        patience_counter = 0
        
        print(f"  [{combination_id}] 학습 시작...")
        
        # 학습 루프
        for epoch in range(max_epochs):
            # Training
            model.train()
            total_loss = 0.0
            
            try:
                for batch_idx, data in enumerate(train_loader):
                    data = data.to(device)
                    optimizer.zero_grad()
                    
                    edge_logits = model(data.x, data.edge_index, data.edge_attr)
                    loss = loss_fn(edge_logits, data.y)
                    
                    loss.backward()
                    
                    # 그래디언트 클리핑으로 안정성 향상
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    total_loss += loss.item()
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  [{combination_id}] GPU 메모리 부족, 조합 스킵")
                    cleanup_gpu_memory()
                    return None
                else:
                    raise e
            
            # Validation
            model.eval()
            val_preds, val_labels = [], []
            val_loss = 0.0
            
            try:
                with torch.no_grad():
                    for data in val_loader:
                        data = data.to(device)
                        edge_logits = model(data.x, data.edge_index, data.edge_attr)
                        
                        loss = loss_fn(edge_logits, data.y)
                        val_loss += loss.item()
                        
                        val_preds.extend(torch.sigmoid(edge_logits).cpu().numpy())
                        val_labels.extend(data.y.cpu().numpy())
                
                # 배열 변환
                val_preds = np.array(val_preds)
                val_labels = np.array(val_labels)
                val_bin_preds = (val_preds > 0.5).astype(int)
                
                # 메트릭 계산
                f1 = f1_score(val_labels, val_bin_preds, zero_division=0)
                precision = precision_score(val_labels, val_bin_preds, zero_division=0)
                recall = recall_score(val_labels, val_bin_preds, zero_division=0)
                
                # ROC-AUC 안전 계산
                if len(np.unique(val_labels)) > 1 and len(val_preds) > 0:
                    roc = roc_auc_score(val_labels, val_preds)
                else:
                    roc = 0.0
                
                # Top-K 메트릭
                precision_k = precision_at_k(val_labels, val_preds, k)
                recall_k = recall_at_k(val_labels, val_preds, k)
                f1_k = f1_score_at_k(val_labels, val_preds, k)
                
                scheduler.step(f1)
                
                # Best model 업데이트
                if f1 > best_f1:
                    best_f1 = f1
                    patience_counter = 0
                    best_metrics = {
                        'epoch': epoch,
                        'train_loss': float(total_loss),
                        'val_loss': float(val_loss),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1': float(f1),
                        'roc_auc': float(roc),
                        'precision_at_k': float(precision_k),
                        'recall_at_k': float(recall_k),
                        'f1_at_k': float(f1_k),
                        'fraud_ratio': float(val_labels.mean()) if len(val_labels) > 0 else 0.0
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  [{combination_id}] Early stopping at epoch {epoch}")
                        break
                        
            except Exception as val_error:
                print(f"  [{combination_id}] Validation 오류: {str(val_error)}")
                break
        
        # 메모리 정리
        del model, optimizer, scheduler, loss_fn
        cleanup_gpu_memory()
        
        if best_metrics:
            print(f"  [{combination_id}] 완료 - F1: {best_metrics['f1']:.4f}, F1@{k}: {best_metrics['f1_at_k']:.4f}")
            return best_metrics
        else:
            print(f"  [{combination_id}] 실패 - 유효한 결과 없음")
            return None
            
    except Exception as e:
        print(f"  [{combination_id}] 심각한 오류: {str(e)}")
        cleanup_gpu_memory()
        return None

def overnight_grid_search(dataset, param_grid, device, k=150, max_epochs=25, patience=5, checkpoint_freq=5):
    """
    야간 실행용 완전 자동화된 Grid Search
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"overnight_search_log_{timestamp}.txt"
    results_file = f"overnight_search_results_{timestamp}.json"
    checkpoint_file = f"checkpoint_{timestamp}.json"
    
    # 로그 파일 초기화
    with open(log_file, 'w') as f:
        f.write(f"Overnight Grid Search 시작 - {datetime.now()}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Dataset: Train {dataset.data_train.x.shape[0]} nodes, {dataset.data_train.edge_index.shape[1]} edges\n")
        f.write(f"Dataset: Val {dataset.data_val.x.shape[0]} nodes, {dataset.data_val.edge_index.shape[1]} edges\n\n")
    
    print("🌙 야간 Grid Search 시작!")
    print(f"로그 파일: {log_file}")
    print(f"결과 파일: {results_file}")
    
    # 파라미터 조합 생성
    param_combinations = []
    param_names = list(param_grid.keys())
    
    for values in product(*param_grid.values()):
        param_dict = dict(zip(param_names, values))
        param_combinations.append(param_dict)
    
    total_combinations = len(param_combinations)
    print(f"총 조합 수: {total_combinations}")
    print(f"예상 시간: 약 {total_combinations * max_epochs * 2 / 60:.1f}시간")
    
    # 결과 저장용
    results = []
    best_score = 0.0
    best_params = None
    completed_count = 0
    
    # 체크포인트 로딩 (재시작 시)
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                results = checkpoint_data.get('results', [])
                best_score = checkpoint_data.get('best_score', 0.0)
                best_params = checkpoint_data.get('best_params', None)
                completed_count = len(results)
                print(f"체크포인트에서 재개: {completed_count}개 조합 완료됨")
        except:
            print("체크포인트 로딩 실패, 새로 시작")
    
    # Grid Search 실행
    for i, params in enumerate(param_combinations):
        # 이미 완료된 조합 스킵
        if i < completed_count:
            continue
            
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{current_time}] 조합 {i+1}/{total_combinations}")
        print(f"파라미터: {params}")
        
        # 로그 기록
        with open(log_file, 'a') as f:
            f.write(f"[{current_time}] 조합 {i+1}/{total_combinations}: {params}\n")
        
        # 학습 실행
        metrics = train_and_evaluate_safe(
            params=params,
            dataset=dataset,
            device=device,
            k=k,
            max_epochs=max_epochs,
            patience=patience
        )
        
        if metrics is not None:
            # 결과 저장
            result = {
                'combination_id': i+1,
                'params': params,
                'metrics': metrics,
                'f1_score': metrics['f1'],
                'f1_at_k': metrics['f1_at_k'],
                'roc_auc': metrics['roc_auc'],
                'completed_at': datetime.now().isoformat()
            }
            results.append(result)
            
            # 로그 기록
            with open(log_file, 'a') as f:
                f.write(f"  결과: F1={metrics['f1']:.4f}, F1@{k}={metrics['f1_at_k']:.4f}, ROC={metrics['roc_auc']:.4f}\n")
            
            # Best 업데이트
            if metrics['f1_at_k'] > best_score:
                best_score = metrics['f1_at_k']
                best_params = params.copy()
                
                with open(log_file, 'a') as f:
                    f.write(f"  ⭐ 새로운 최고 점수!\n")
                
                print(f"  ⭐ 새로운 최고 점수: F1@{k}={best_score:.4f}")
            
        else:
            with open(log_file, 'a') as f:
                f.write(f"  실패\n")
            print(f"  조합 실패")
        
        # 체크포인트 저장 (5개 조합마다)
        if (i + 1) % checkpoint_freq == 0:
            checkpoint_data = {
                'results': results,
                'best_score': float(best_score),
                'best_params': best_params,
                'completed_count': len(results),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            print(f"  체크포인트 저장됨 ({len(results)}개 완료)")
    
    # 최종 결과 정리 및 저장
    if results:
        results.sort(key=lambda x: x['f1_at_k'], reverse=True)
        
        final_results = {
            'search_completed_at': datetime.now().isoformat(),
            'total_combinations': total_combinations,
            'successful_combinations': len(results),
            'best_params': best_params,
            'best_score': float(best_score),
            'top_10_results': results[:10],
            'all_results': results,
            'dataset_info': {
                'train_nodes': int(dataset.data_train.x.shape[0]),
                'train_edges': int(dataset.data_train.edge_index.shape[1]),
                'val_nodes': int(dataset.data_val.x.shape[0]),
                'val_edges': int(dataset.data_val.edge_index.shape[1]),
                'node_features': int(dataset.data_train.x.shape[1]),
                'edge_features': int(dataset.data_train.edge_attr.shape[1])
            },
            'search_config': {
                'param_grid': param_grid,
                'k': k,
                'max_epochs': max_epochs,
                'patience': patience
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # 최종 로그 기록
        with open(log_file, 'a') as f:
            f.write(f"\n=== 최종 결과 ===\n")
            f.write(f"완료 시간: {datetime.now()}\n")
            f.write(f"성공한 조합: {len(results)}/{total_combinations}\n")
            f.write(f"최고 성능: F1@{k} = {best_score:.4f}\n")
            f.write(f"최적 파라미터: {best_params}\n")
            
            if len(results) >= 10:
                f.write(f"\nTop 10 결과:\n")
                for i, result in enumerate(results[:10]):
                    f.write(f"{i+1:2d}. F1@{k}: {result['f1_at_k']:.4f} | {result['params']}\n")
        
        print(f"\n🎉 야간 Grid Search 완료!")
        print(f"성공한 조합: {len(results)}/{total_combinations}")
        print(f"최고 성능: F1@{k} = {best_score:.4f}")
        print(f"최적 파라미터: {best_params}")
        print(f"결과 파일: {results_file}")
        
        return results, best_params
    
    else:
        print("❌ 모든 조합이 실패했습니다.")
        return [], None

# ================== 실행 부분 ==================
if __name__ == "__main__":
    print("🌙 야간 실행용 Grid Search")
    print("=" * 50)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    
    # 데이터셋 로딩
    print("\n데이터셋 로딩 중...")
    try:
        dataset = AMLtoGraph("./lej_dataset_002")
        print("데이터셋 로딩 완료!")
    except Exception as e:
        print(f"데이터셋 로딩 실패: {e}")
        exit(1)
    
    # 야간 실행용 확장 파라미터 그리드
    param_grid = {
        'model_class': [EdgeGATOptimized, EdgeGATLightweight],
        'hidden_channels': [64, 128, 256],
        'heads': [2, 4, 8],
        'batch_size': [1024, 2048],
        'lr': [0.005, 0.01, 0.015, 0.02],
        'weight_decay': [1e-4, 1e-5],
        'focal_alpha': [0.9, 0.95, 0.99],
        'focal_gamma': [1, 2]
    }
    
    total_combinations = len(list(product(*param_grid.values())))
    estimated_hours = total_combinations * 25 * 2 / 3600  # 대략적인 시간 계산
    
    print(f"\n실행 정보:")
    print(f"- 총 파라미터 조합: {total_combinations}개")
    print(f"- 예상 실행 시간: 약 {estimated_hours:.1f}시간")
    print(f"- 각 조합당 최대 epoch: 25")
    print(f"- Early stopping patience: 5")
    
    input("\n준비되면 Enter를 눌러 시작하세요...")
    
    # 야간 Grid Search 실행
    results, best_params = overnight_grid_search(
        dataset=dataset,
        param_grid=param_grid,
        device=device,
        k=150,
        max_epochs=25,
        patience=5,
        checkpoint_freq=5
    )
    
    if best_params:
        print(f"\n✨ 최종 최적 파라미터:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
    
    print("\n모든 작업 완료! 내일 아침에 결과를 확인하세요.")
    
    