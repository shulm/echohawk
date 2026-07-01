"""Train/evaluate DroneCNN with corrected grouped cross-validation."""
from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from echohawk import features, models, evaluate, eval_utils  # noqa: E402
from echohawk.data_io import load_drone_audio_dataset  # noqa: E402

RESULTS = os.path.join(_REPO, "results")
MANIFESTS = os.path.join(RESULTS, "manifests")
FS = 16_000
CLIP_S = 1.0
SEED = 0
DATA_ROOT = os.path.join(_REPO, "data", "drone_audio_dataset")


def _group_keys_for_mode(mode: str, corrected_keys, file_keys, n_samples: int):
    if mode == "corrected":
        return corrected_keys
    if mode == "file":
        return file_keys
    if mode == "clip":
        return np.asarray([f"clip:{i}" for i in range(n_samples)], dtype=object)
    raise ValueError(mode)


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DroneCNNDataset(Dataset):
    def __init__(self, X_list, y, fs, fit_scaler=False, scaler_mean=None, scaler_std=None):
        self.y = np.asarray(y, dtype=np.int64)
        feats = []
        for x in X_list:
            mel = features.log_mel_spectrogram(x, fs)
            n_mels, frames = mel.shape
            target_frames = 64
            if frames < target_frames:
                mel = np.pad(mel, ((0, 0), (0, target_frames - frames)), mode="constant")
            else:
                mel = mel[:, :target_frames]
            feats.append(mel.astype(np.float32))
        self.features = np.stack(feats)

        if fit_scaler:
            self.scaler_mean = self.features.mean(axis=(0, 2), keepdims=True)
            self.scaler_std = self.features.std(axis=(0, 2), keepdims=True) + 1e-8
        else:
            if scaler_mean is None or scaler_std is None:
                raise ValueError("scaler_mean/scaler_std required when fit_scaler=False")
            self.scaler_mean = scaler_mean
            self.scaler_std = scaler_std

        self.features = (self.features - self.scaler_mean) / self.scaler_std
        self.features = np.expand_dims(self.features, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


def _load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def train_one_fold(
    fold,
    X_list,
    y,
    group_keys,
    file_keys,
    group_mode,
    train_idx,
    test_idx,
    args,
    device,
):
    set_seeds(SEED + fold)
    train_idx, val_idx = eval_utils.make_validation_split(
        train_idx,
        y,
        group_keys,
        seed=SEED + fold,
        val_size=args.val_size,
    )
    assertions = eval_utils.assert_group_disjoint(
        y,
        group_keys,
        {"train": train_idx, "val": val_idx, "test": test_idx},
    )
    for msg in assertions:
        print(f"  fold {fold} {msg}")

    X_train = [X_list[i] for i in train_idx]
    X_val = [X_list[i] for i in val_idx]
    X_test = [X_list[i] for i in test_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    print(
        f"  fold {fold} sizes: "
        f"train={len(y_train)} val={len(y_val)} test={len(y_test)}"
    )

    t0 = time.time()
    ds_train = DroneCNNDataset(X_train, y_train, FS, fit_scaler=True)
    ds_val = DroneCNNDataset(
        X_val,
        y_val,
        FS,
        fit_scaler=False,
        scaler_mean=ds_train.scaler_mean,
        scaler_std=ds_train.scaler_std,
    )
    ds_test = DroneCNNDataset(
        X_test,
        y_test,
        FS,
        fit_scaler=False,
        scaler_mean=ds_train.scaler_mean,
        scaler_std=ds_train.scaler_std,
    )
    print(f"  fold {fold} feature/scaler time: {time.time() - t0:.1f}s")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)

    model = models.build_cnn(n_mels=40, n_classes=2).to(device)
    counts = np.bincount(y_train, minlength=2)
    if np.any(counts == 0):
        raise RuntimeError(f"fold {fold}: train split missing a class: {counts}")
    weights = 1.0 / counts
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_auc = -1.0
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    val_aucs = []
    ckpt_path = os.path.join(RESULTS, f"cnn_fold{fold}.pt")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bx.size(0)
        train_loss = total_loss / len(ds_train)
        train_losses.append(train_loss)

        model.eval()
        total_val_loss = 0.0
        val_probs = []
        val_targets = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                total_val_loss += loss.item() * bx.size(0)
                val_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                val_targets.extend(by.cpu().numpy())
        val_loss = total_val_loss / len(ds_val)
        val_losses.append(val_loss)
        try:
            from sklearn.metrics import roc_auc_score

            val_auc = float(roc_auc_score(val_targets, val_probs))
        except ValueError:
            val_auc = 0.5
        val_aucs.append(val_auc)

        print(
            f"    fold {fold} epoch {epoch + 1:02d}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f}"
        )
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"    fold {fold}: early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(_load_state_dict(ckpt_path, device))
    model.eval()
    test_probs = []
    test_targets = []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            test_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            test_targets.extend(by.cpu().numpy())

    test_probs = np.asarray(test_probs)
    test_targets = np.asarray(test_targets)
    metrics = evaluate.binary_detection_metrics(test_targets, test_probs)
    pred_rows = []
    for sample_idx, label, score in zip(test_idx, test_targets, test_probs):
        pred_rows.append(
            {
                "model": "cnn",
                "group_mode": group_mode,
                "fold": fold,
                "sample_index": int(sample_idx),
                "label": int(label),
                "score": float(score),
                "group_key": str(group_keys[sample_idx]),
                "file_key": str(file_keys[sample_idx]),
            }
        )

    return {
        "metrics": metrics,
        "pred_rows": pred_rows,
        "assertions": [f"fold={fold} {msg}" for msg in assertions],
        "val_split": (train_idx, val_idx),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_aucs": val_aucs,
        "best_val_auc": best_val_auc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--group-mode",
        default="corrected",
        choices=["clip", "file", "corrected"],
    )
    args = parser.parse_args()

    os.makedirs(MANIFESTS, exist_ok=True)
    set_seeds(SEED)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    print(f"Loading DroneAudioDataset from local path: {DATA_ROOT} ...")
    t0 = time.time()
    X_list, y, groups, fs, metadata = load_drone_audio_dataset(
        DATA_ROOT,
        fs=FS,
        clip_s=CLIP_S,
        return_metadata=True,
    )
    if fs != FS:
        raise RuntimeError(f"unexpected fs={fs}")
    corrected_keys = metadata["group_keys"]
    file_keys = metadata["file_keys"]
    group_keys = _group_keys_for_mode(args.group_mode, corrected_keys, file_keys, len(y))
    print(f"Loaded {len(y)} clips in {time.time() - t0:.1f}s")
    print(f"Group mode: {args.group_mode}")
    print("Group summary:", eval_utils.group_summary(y, group_keys))

    folds = eval_utils.make_group_kfold(y, group_keys, n_splits=args.folds)
    fold_metrics = []
    pred_rows = []
    assertions = []
    val_splits = {}
    histories = {}

    for fold, (train_idx, test_idx) in enumerate(folds):
        print(f"\nCNN fold {fold}/{args.folds - 1}")
        result = train_one_fold(
            fold,
            X_list,
            y,
            group_keys,
            file_keys,
            args.group_mode,
            train_idx,
            test_idx,
            args,
            device,
        )
        fold_metrics.append(result["metrics"])
        pred_rows.extend(result["pred_rows"])
        assertions.extend(result["assertions"])
        val_splits[fold] = result["val_split"]
        histories[str(fold)] = {
            "train_losses": result["train_losses"],
            "val_losses": result["val_losses"],
            "val_aucs": result["val_aucs"],
            "best_val_auc": result["best_val_auc"],
        }
        m = result["metrics"]
        print(
            f"  fold {fold}: AUC={m['auc']:.4f} acc={m['accuracy']:.4f} "
            f"Pd@1%FA={m['pd_at_1pct_fa']:.4f} "
            f"Pd@5%FA={m['pd_at_5pct_fa']:.4f} "
            f"Pd@10%FA={m['pd_at_10pct_fa']:.4f}"
        )

    summary = eval_utils.metrics_summary(fold_metrics)
    print(f"\nCNN {args.group_mode} grouped-CV summary")
    for key, item in summary.items():
        print(f"  {key}: {item['mean']:.4f} ± {item['ci95']:.4f}")

    prefix = os.path.join(MANIFESTS, f"real_cnn_{args.group_mode}_{args.folds}fold")
    eval_utils.write_predictions_csv(prefix + "_predictions.csv", pred_rows)
    eval_utils.save_json(
        prefix + "_summary.json",
        {
            "model": "cnn",
            "seed": SEED,
            "folds": args.folds,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "device": str(device),
            "group_mode": args.group_mode,
            "pd_at_pfa_method": "interp",
            "group_summary": eval_utils.group_summary(y, group_keys),
            "fold_metrics": fold_metrics,
            "summary": summary,
            "assertions": assertions,
            "histories": histories,
        },
    )
    eval_utils.save_json(
        prefix + "_split_manifest.json",
        {
            "model": "cnn",
            "group_mode": args.group_mode,
            "folds": eval_utils.split_manifest(y, group_keys, file_keys, folds, val_splits),
        },
    )
    eval_utils.save_pip_freeze(os.path.join(MANIFESTS, "pip_freeze_cnn.txt"))

    # Keep familiar figures from the final fold.
    last_rows = [row for row in pred_rows if row["fold"] == args.folds - 1]
    y_last = np.asarray([row["label"] for row in last_rows])
    scores_last = np.asarray([row["score"] for row in last_rows])
    evaluate.plot_roc(y_last, scores_last, os.path.join(RESULTS, "roc_cnn_real.png"))
    evaluate.plot_confusion(
        y_last,
        (scores_last >= 0.5).astype(int),
        os.path.join(RESULTS, "confusion_cnn.png"),
    )
    if histories:
        fig, ax = plt.subplots(figsize=(6, 4))
        for fold, hist in histories.items():
            ax.plot(hist["val_aucs"], label=f"fold {fold}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation ROC AUC")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "cnn_training.png"), dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
