# train_anomaly.py
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from src.dataset import get_dataloader, TelemetryDataset
from src.model import LSTMAutoencoder

DATA_DIR = "data"
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

SEQ_LEN = 100     # 2 seconds @ 50Hz
BATCH_SIZE = 16
EPOCHS = 15       # bump to 50+ for real
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FALLBACK_FEATURE_NAMES = [
    "feature_0",
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
    "feature_5",
    "feature_6",
    "feature_7",
]


def _resolve_feature_names(input_dim: int) -> List[str]:
    order_file = Path(DATA_DIR) / "feature_order.txt"
    names: List[str] = []
    if order_file.exists():
        with order_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                clean = line.strip()
                if clean:
                    names.append(clean)
                if len(names) >= input_dim:
                    break
    if len(names) < input_dim:
        names.extend(FALLBACK_FEATURE_NAMES[len(names):input_dim])
    return names[:input_dim]


def _compute_feature_stats(dataset: TelemetryDataset, feature_names: List[str]) -> Dict[str, List[float]]:
    arrays: List[np.ndarray] = []
    lengths: List[int] = []
    for path in dataset.files:
        arr = dataset._load_file(path)
        if arr.ndim == 1:
            arr = arr[:, None]
        arrays.append(arr)
        lengths.append(arr.shape[0])

    if not arrays:
        raise RuntimeError(f"No telemetry files found in {DATA_DIR!r} to compute stats.")

    merged = np.vstack(arrays)  # (total_steps, features)
    if merged.shape[1] != len(feature_names):
        raise RuntimeError(f"Feature count mismatch: merged has {merged.shape[1]}, expected {len(feature_names)}")

    stats: Dict[str, List[float]] = {
        "feature_names": feature_names,
        "mean": merged.mean(axis=0).tolist(),
        "std": (merged.std(axis=0) + 1e-6).tolist(),
        "min": merged.min(axis=0).tolist(),
        "max": merged.max(axis=0).tolist(),
        "quantiles_p50": np.quantile(merged, 0.5, axis=0).tolist(),
        "quantiles_p80": np.quantile(merged, 0.8, axis=0).tolist(),
        "quantiles_p90": np.quantile(merged, 0.9, axis=0).tolist(),
        "quantiles_p95": np.quantile(merged, 0.95, axis=0).tolist(),
        "quantiles_p99": np.quantile(merged, 0.99, axis=0).tolist(),
        "dataset_size": len(arrays),
        "total_timesteps": int(merged.shape[0]),
    }

    abs_quantiles: Dict[str, Dict[str, float]] = {}
    neg_quantiles: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(feature_names):
        abs_vals = np.abs(merged[:, idx])
        abs_quantiles[name] = {
            "p80": float(np.quantile(abs_vals, 0.8)),
            "p90": float(np.quantile(abs_vals, 0.9)),
            "p95": float(np.quantile(abs_vals, 0.95)),
            "p99": float(np.quantile(abs_vals, 0.99)),
        }
        neg_component = -np.minimum(merged[:, idx], 0.0)
        if np.count_nonzero(neg_component) > 0:
            neg_quantiles[name] = {
                "p80": float(np.quantile(neg_component, 0.8)),
                "p90": float(np.quantile(neg_component, 0.9)),
                "p95": float(np.quantile(neg_component, 0.95)),
                "p99": float(np.quantile(neg_component, 0.99)),
            }

    stats["abs_quantiles"] = abs_quantiles
    stats["neg_quantiles"] = neg_quantiles
    stats["sequence_length"] = {
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "mean": float(np.mean(lengths)),
    }
    return stats


def main():
    # 1. dataloader
    dataset = TelemetryDataset(DATA_DIR, seq_len=SEQ_LEN, normalize=False)
    if len(dataset) == 0:
        raise RuntimeError(f"No telemetry sequences found under {DATA_DIR!r}")

    dl = get_dataloader(DATA_DIR, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, shuffle=True, normalize=False)

    # infer feature size from first batch
    sample_batch = next(iter(dl))
    _, seq_len, input_dim = sample_batch.shape

    feature_names = _resolve_feature_names(input_dim)
    feature_stats = _compute_feature_stats(dataset, feature_names)
    mean = torch.tensor(feature_stats["mean"], dtype=torch.float32, device=DEVICE).view(1, 1, input_dim)
    std = torch.tensor(feature_stats["std"], dtype=torch.float32, device=DEVICE).view(1, 1, input_dim)

    # 2. model
    model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=128, latent_dim=32, num_layers=2).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for batch in dl:
            batch = batch.to(DEVICE)  # (B, T, F)
            norm_batch = (batch - mean) / std
            recon, _ = model(norm_batch)
            loss = criterion(recon, norm_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * batch.size(0)

        epoch_loss = running / len(dl.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                "model_state": model.state_dict(),
                "input_dim": input_dim,
                "seq_len": SEQ_LEN,
                "feature_stats": feature_stats,
            }, os.path.join(SAVE_DIR, "best_model.pth"))
            print("✅ saved better model")

    # compute reconstruction error calibration on best model
    ckpt_path = Path(SAVE_DIR) / "best_model.pth"
    if not ckpt_path.exists():
        raise RuntimeError("Expected best model checkpoint to exist but none was saved.")
    best_ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()
    recon_errors: List[float] = []
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(DEVICE)
            norm_batch = (batch - mean) / std
            recon, _ = model(norm_batch)
            mse = torch.mean((recon - norm_batch) ** 2, dim=(1, 2))
            recon_errors.extend(mse.cpu().numpy().tolist())

    if recon_errors:
        recon_errors_np = np.array(recon_errors)
        calibration = {
            "mse_mean": float(np.mean(recon_errors_np)),
            "mse_p90": float(np.quantile(recon_errors_np, 0.90)),
            "mse_p95": float(np.quantile(recon_errors_np, 0.95)),
            "mse_p99": float(np.quantile(recon_errors_np, 0.99)),
            "mse_max": float(np.max(recon_errors_np)),
            "sample_count": int(len(recon_errors_np)),
        }
        ckpt_path = Path(SAVE_DIR) / "best_model.pth"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt["calibration"] = {
            "reconstruction_error": calibration,
        }
        torch.save(ckpt, ckpt_path)

if __name__ == "__main__":
    main()
