# train_anomaly.py
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from src.dataset import get_dataloader
from src.model import LSTMAutoencoder

DATA_DIR = "data"
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

SEQ_LEN = 100     # 2 seconds @ 50Hz
BATCH_SIZE = 16
EPOCHS = 15       # bump to 50+ for real
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # 1. dataloader
    dl = get_dataloader(DATA_DIR, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)

    # infer feature size from first batch
    sample_batch = next(iter(dl))
    _, seq_len, input_dim = sample_batch.shape

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
            recon, _ = model(batch)
            loss = criterion(recon, batch)

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
                "seq_len": SEQ_LEN
            }, os.path.join(SAVE_DIR, "best_model.pth"))
            print("✅ saved better model")

if __name__ == "__main__":
    main()
