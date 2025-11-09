import math
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from dataset import prepare_dataset, collate_fn
from model import Transformer, make_src_mask, make_tgt_mask


# ----------------------------
# Label smoothing loss
# ----------------------------
class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing: float, tgt_vocab_size: int, ignore_index: int = 0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smoothing = label_smoothing
        self.confidence = 1.0 - label_smoothing
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        true_dist = pred.clone()
        true_dist.fill_(self.smoothing / (self.tgt_vocab_size - 2))
        mask = (target != self.ignore_index)
        target = target[mask]
        if target.numel() == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred.device)
        true_dist[mask, :] = self.smoothing / (self.tgt_vocab_size - 2)
        true_dist[mask, target] = self.confidence
        loss = torch.sum(-true_dist[mask] * pred[mask]) / mask.sum()
        return loss


# ----------------------------
# Accuracy computation
# ----------------------------
def compute_accuracy(logits, targets, pad_idx):
    preds = logits.argmax(dim=-1)
    mask = (targets != pad_idx)
    correct = (preds == targets) & mask
    acc = correct.sum().item() / mask.sum().item()
    return acc


# ----------------------------
# Training and evaluation
# ----------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, pad_idx, device):
    model.train()
    total_loss, total_acc, total_tokens = 0, 0, 0

    for src, tgt in tqdm(dataloader, desc="Training", leave=False):
        src, tgt = src.to(device), tgt.to(device)
        src_mask = make_src_mask(src, pad_idx)
        tgt_mask = make_tgt_mask(tgt, pad_idx)

        tgt_inp = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        optimizer.zero_grad()
        logits = model(src, tgt_inp, src_mask=src_mask, tgt_mask=tgt_mask[:, :-1, :-1])
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc += compute_accuracy(logits, tgt_out, pad_idx)
        total_tokens += 1

    avg_loss = total_loss / total_tokens
    avg_acc = total_acc / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, avg_acc, ppl


def evaluate(model, dataloader, criterion, pad_idx, device):
    model.eval()
    if len(dataloader) == 0:
        print("Warning: empty dataloader in evaluation.")
        return float('inf'), 0.0, float('inf')
    total_loss, total_acc, total_tokens = 0, 0, 0
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating", leave=False):
            src, tgt = src.to(device), tgt.to(device)
            src_mask = make_src_mask(src, pad_idx)
            tgt_mask = make_tgt_mask(tgt, pad_idx)
            tgt_inp = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            logits = model(src, tgt_inp, src_mask=src_mask, tgt_mask=tgt_mask[:, :-1, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            total_loss += loss.item()
            total_acc += compute_accuracy(logits, tgt_out, pad_idx)
            total_tokens += 1
    avg_loss = total_loss / total_tokens
    avg_acc = total_acc / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, avg_acc, ppl


# ----------------------------
# Greedy decoding for sample test
# ----------------------------
def greedy_decode(model, src, src_mask, max_len, start_symbol, pad_idx, device):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).long().to(device)
    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, pad_idx)
        out = model.decode(ys, memory, src_mask, tgt_mask)
        prob = model.generator(out[:, -1])
        next_word = torch.argmax(prob, dim=1).item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src).fill_(next_word)], dim=1)
        if next_word == model.tgt_embed.num_embeddings - 1:
            break
    return ys


def sample_translation(model, src_vocab, tgt_vocab, dataset, pad_idx, device):
    model.eval()
    idx = random.randint(0, len(dataset) - 1)
    src_tensor, tgt_tensor = dataset[idx]
    src = src_tensor.unsqueeze(0).to(device)
    src_mask = make_src_mask(src, pad_idx)

    decoded = greedy_decode(model, src, src_mask, max_len=50, start_symbol=tgt_vocab.stoi['<bos>'], pad_idx=pad_idx, device=device)

    src_text = ' '.join(src_vocab.decode(src_tensor.tolist()))
    ref_text = ' '.join(tgt_vocab.decode(tgt_tensor.tolist()))
    pred_text = ' '.join(tgt_vocab.decode(decoded[0].tolist()))

    print("\n[SRC]", src_text)
    print("[REF]", ref_text)
    print("[PRED]", pred_text)


def set_seed(seed: int = 42):
    import random, os, numpy as np, torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 保证每次结果一致（会稍微降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[Info] Random seed set to {seed}")


# ----------------------------
# Main
# ----------------------------
def main():
    set_seed(42)
    
    SRC_FILE = "en-de/train.tags.en-de.en"
    TGT_FILE = "en-de/train.tags.en-de.de"
    DEV_SRC = "en-de/IWSLT17.TED.dev2010.en-de.en.xml"
    DEV_TGT = "en-de/IWSLT17.TED.dev2010.en-de.de.xml"

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    train_data, dev_data, src_vocab, tgt_vocab = prepare_dataset(SRC_FILE, TGT_FILE, DEV_SRC, DEV_TGT)

    BATCH_SIZE = 64
    EPOCHS = 30
    LR = 3e-4
    PATIENCE = 5
    PAD_IDX = src_vocab.stoi['<pad>']

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = Transformer(len(src_vocab), len(tgt_vocab), d_model=256, N=4, heads=8, d_ff=1024, dropout=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    criterion = LabelSmoothingLoss(0.1, len(tgt_vocab), ignore_index=PAD_IDX)

    best_val_loss = float('inf')
    no_improve_epochs = 0
    history = []

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_dir = os.path.join('results', f'train_{timestamp}')
    os.makedirs(result_dir, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_acc, train_ppl = train_one_epoch(model, train_loader, optimizer, criterion, PAD_IDX, device)
        val_loss, val_acc, val_ppl = evaluate(model, dev_loader, criterion, PAD_IDX, device)

        print(f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")
        print(f"Train PPL={train_ppl:.2f} | Val PPL={val_ppl:.2f}")
        print(f"Train Acc={train_acc*100:.2f}% | Val Acc={val_acc*100:.2f}%")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss, 'val_loss': val_loss,
            'train_ppl': train_ppl, 'val_ppl': val_ppl,
            'train_acc': train_acc, 'val_acc': val_acc
        })

        sample_translation(model, src_vocab, tgt_vocab, dev_data or train_data, PAD_IDX, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'src_vocab': src_vocab.__dict__,
                'tgt_vocab': tgt_vocab.__dict__,
            }, os.path.join(result_dir, 'best_transformer.pt'))
            print("✅ Saved new best model.")
        else:
            no_improve_epochs += 1
            print(f"⚠️ No improvement for {no_improve_epochs} epochs.")
            if no_improve_epochs >= PATIENCE:
                print(f"⏹ Early stopping triggered after {epoch} epochs.")
                break

        # Log saving
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(result_dir, 'training_log.csv'), index=False)

        # Plot curves
        plt.figure()
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curve')
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(result_dir, 'loss_curve.png')); plt.close()

        plt.figure()
        plt.plot(df['epoch'], df['train_ppl'], label='Train PPL')
        plt.plot(df['epoch'], df['val_ppl'], label='Val PPL')
        plt.xlabel('Epoch'); plt.ylabel('Perplexity'); plt.title('Perplexity Curve')
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(result_dir, 'ppl_curve.png')); plt.close()

        plt.figure()
        plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy')
        plt.plot(df['epoch'], df['val_acc'], label='Val Accuracy')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy Curve')
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(result_dir, 'accuracy_curve.png')); plt.close()

    print(f"\nTraining complete. Logs and figures saved in {result_dir}")


if __name__ == "__main__":
    main()
