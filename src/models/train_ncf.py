# src/models/train_ncf.py
from pathlib import Path
import json
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.ncf_dataset import RatingsDataset
from src.models.ncf_model import NeuralCF

DATA_GOLD = Path("data/gold")
MODELS = Path("models")

def loss_loss(pred, target):
    return torch.sqrt(nn.functional.mse_loss(pred, target))

def train_one_epoch(model, criterion, loader, optim, device, clip_grad=None):
    model.train()
    total_loss = 0.0
    for users, items, ratings in tqdm(loader, leave=False):
        users = users.to(device)
        items = items.to(device)
        ratings = ratings.to(device)

        optim.zero_grad()
        preds = model(users, items)
        
        loss = criterion(preds, ratings)
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optim.step()
        total_loss += loss.item() * users.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for users, items, ratings in loader:
        users = users.to(device)
        items = items.to(device)
        ratings = ratings.to(device)
        preds = model(users, items)
        loss = loss_loss(preds, ratings)
        total_loss += loss.item() * users.size(0)
    return total_loss / len(loader.dataset)

def count_cardinalities():
    # infer n_users / n_items from the splits
    import pandas as pd
    tr = pd.read_parquet(DATA_GOLD / "train.parquet")
    n_users = tr["user_idx"].max() + 1
    n_items = tr["movie_idx"].max() + 1
    return int(n_users), int(n_items)

def main(args):
    MODELS.mkdir(parents=True, exist_ok=True)
    criterion = nn.SmoothL1Loss()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    n_users, n_items = count_cardinalities()

    train_ds = RatingsDataset(DATA_GOLD / "train.parquet")
    val_ds   = RatingsDataset(DATA_GOLD / "val.parquet")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = NeuralCF(n_users, n_items, emb_dim=args.emb_dim, hidden=(args.h1, args.h2), dropout=args.dropout).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = math.inf
    patience = args.patience
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, criterion, train_loader, optim, device, clip_grad=args.clip_grad)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | train loss: {tr_loss:.4f} | val loss: {val_loss:.4f}")

        # checkpoint
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            epochs_no_improve = 0
            ckpt_path = MODELS / "ncf.pt"
            torch.save({
                "model_state": model.state_dict(),
                "n_users": n_users,
                "n_items": n_items,
                "emb_dim": args.emb_dim,
                "hidden": (args.h1, args.h2),
                "dropout": args.dropout,
                "val_loss": best_val,
            }, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break

    # save training summary
    (MODELS / "ncf_meta.json").write_text(json.dumps({
        "best_val_loss": best_val,
        "epochs_ran": epoch,
        "params": vars(args),
    }, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--h1", type=int, default=128)
    p.add_argument("--h2", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--cpu", action="store_true", help="force CPU")
    args = p.parse_args()
    main(args)
