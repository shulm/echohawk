"""Train DroneCNN on the DroneAudioDataset.

Workflow
--------
1. Load dataset clips (groups = source recordings).
2. Compute log-mel spectrogram for each clip, pad/crop to 64 frames.
3. Grouped split into train (55%), val (15%), test (30%).
4. Standardise per-mel-bin using train-split statistics.
5. Train DroneCNN (PyTorch) with Adam and weighted CrossEntropyLoss.
6. Evaluate on the held-out test split (accuracy, AUC, Pd@1/5/10%FA).
7. Save learning curves and confusion matrix to `results/`.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from echohawk import features, models, evaluate
from echohawk.data_io import load_drone_audio_dataset

RESULTS = os.path.join(_REPO, "results")
FS = 16_000
CLIP_S = 1.0
SEED = 0

# Random seeds
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class DroneCNNDataset(Dataset):
    def __init__(self, X_list, y, fs, fit_scaler=False, scaler_mean=None, scaler_std=None):
        self.y = y
        self.features = []
        for x in X_list:
            mel = features.log_mel_spectrogram(x, fs)
            n_mels, T = mel.shape
            target_T = 64
            if T < target_T:
                mel = np.pad(mel, ((0, 0), (0, target_T - T)), mode='constant')
            else:
                mel = mel[:, :target_T]
            self.features.append(mel)
        self.features = np.stack(self.features)
        
        if fit_scaler:
            self.scaler_mean = self.features.mean(axis=(0, 2), keepdims=True)
            self.scaler_std = self.features.std(axis=(0, 2), keepdims=True) + 1e-8
        else:
            self.scaler_mean = scaler_mean
            self.scaler_std = scaler_std
            
        self.features = (self.features - self.scaler_mean) / self.scaler_std
        self.features = np.expand_dims(self.features, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(RESULTS, exist_ok=True)
    
    # 1. Load data
    _DAD_GITHUB_ROOT = os.path.join(_REPO, "data", "drone_audio_dataset")
    print(f"Loading DroneAudioDataset from local path: {_DAD_GITHUB_ROOT} ...")
    t0 = time.time()
    X_list, y, groups, fs = load_drone_audio_dataset(_DAD_GITHUB_ROOT, fs=FS, clip_s=CLIP_S)
    print(f"Loaded in {time.time()-t0:.1f}s")
    
    # 2. Split data (Test 30%, Train+Val 70%)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_val_idx, test_idx = next(gss.split(X_list, y, groups))
    
    X_train_val = [X_list[i] for i in train_val_idx]
    y_train_val = y[train_val_idx]
    groups_train_val = groups[train_val_idx]
    
    X_test = [X_list[i] for i in test_idx]
    y_test = y[test_idx]
    
    # Split Train+Val into Train and Val (Val is 15% of total, so ~21.4% of Train+Val)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.214, random_state=SEED)
    train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups_train_val))
    
    X_train = [X_train_val[i] for i in train_idx]
    y_train = y_train_val[train_idx]
    
    X_val = [X_train_val[i] for i in val_idx]
    y_val = y_train_val[val_idx]
    
    print(f"Splits - Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    
    # 3. Features & Datasets
    print("Extracting features and standardising...")
    t0 = time.time()
    ds_train = DroneCNNDataset(X_train, y_train, fs, fit_scaler=True)
    ds_val = DroneCNNDataset(X_val, y_val, fs, fit_scaler=False, scaler_mean=ds_train.scaler_mean, scaler_std=ds_train.scaler_std)
    ds_test = DroneCNNDataset(X_test, y_test, fs, fit_scaler=False, scaler_mean=ds_train.scaler_mean, scaler_std=ds_train.scaler_std)
    print(f"Done in {time.time()-t0:.1f}s")
    
    train_loader = DataLoader(ds_train, batch_size=32, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=32, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=32, shuffle=False)
    
    # 4. Model, Loss, Optimizer
    model = models.build_cnn(n_mels=40, n_classes=2).to(device)
    
    counts = np.bincount(y_train)
    weights = 1.0 / counts
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 5. Training Loop
    epochs = 40
    patience = 6
    best_val_auc = -1
    epochs_no_improve = 0
    
    train_losses = []
    val_losses = []
    val_aucs = []
    
    print("Training CNN...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bx.size(0)
        
        train_loss = total_loss / len(ds_train)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        val_probs = []
        val_targets = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                total_val_loss += loss.item() * bx.size(0)
                probs = torch.softmax(logits, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_targets.extend(by.cpu().numpy())
                
        val_loss = total_val_loss / len(ds_val)
        val_losses.append(val_loss)
        
        try:
            val_auc = roc_auc_score(val_targets, val_probs)
        except ValueError:
            val_auc = 0.5
        val_aucs.append(val_auc)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(RESULTS, "cnn.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
    # 6. Evaluation
    print(f"\nEvaluating on Test Split (Best Val AUC: {best_val_auc:.4f})")
    model.load_state_dict(torch.load(os.path.join(RESULTS, "cnn.pt")))
    model.eval()
    
    test_probs = []
    test_targets = []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            probs = torch.softmax(logits, dim=1)[:, 1]
            test_probs.extend(probs.cpu().numpy())
            test_targets.extend(by.cpu().numpy())
            
    test_probs = np.array(test_probs)
    test_targets = np.array(test_targets)
    y_hat = (test_probs >= 0.5).astype(int)
    
    acc = accuracy_score(test_targets, y_hat)
    auc = evaluate.plot_roc(test_targets, test_probs,
                            os.path.join(RESULTS, "roc_cnn_real.png"),
                            title="Drone detection ROC (CNN)")
    evaluate.plot_confusion(test_targets, y_hat,
                            os.path.join(RESULTS, "confusion_cnn.png"))
    
    pd01 = evaluate.pd_at_pfa(test_targets, test_probs, pfa=0.01)
    pd05 = evaluate.pd_at_pfa(test_targets, test_probs, pfa=0.05)
    pd10 = evaluate.pd_at_pfa(test_targets, test_probs, pfa=0.10)
    
    print("=" * 58)
    print("  Real-data results (CNN — held-out test split)")
    print("=" * 58)
    print(f"  Accuracy   : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  ROC-AUC    : {auc:.4f}")
    print(f"  Pd@1%FA    : {pd01:.4f}")
    print(f"  Pd@5%FA    : {pd05:.4f}")
    print(f"  Pd@10%FA   : {pd10:.4f}")
    print("=" * 58)
    
    # 7. Training Curve Figure
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropyLoss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(val_aucs, label="Val AUC", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("ROC AUC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "cnn_training.png"))
    plt.close()

if __name__ == "__main__":
    main()
