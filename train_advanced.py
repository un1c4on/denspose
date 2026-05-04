#!/usr/bin/env python3
"""
Advanced CSI Training Script
============================
Features:
  - Loads data from multiple directories (dataset, dataset_2)
  - 3-way split: Train, Validation, Test
  - CNN + Fusion architecture for multi-node CSI
  - Confidence (probability) output logic
"""

import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix

# ── Configuration ─────────────────────────────────────────────────────────────
# Directories to load data from
DATA_DIRS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_2')
]
MODEL_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_advanced.pt')
LABELS       = ['empty', 'present', 'walking', 'fall']
LABEL2IDX    = {l: i for i, l in enumerate(LABELS)}
N_SC         = 64
WINDOW_FRAMES = 60   # 3s @ 20Hz
N_NODES      = 2     # Based on recent setup
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-3

# Styling
def green(s):  return f'\033[92m{s}\033[0m'
def yellow(s): return f'\033[93m{s}\033[0m'
def cyan(s):   return f'\033[96m{s}\033[0m'
def bold(s):   return f'\033[1m{s}\033[0m'

# ── Dataset ───────────────────────────────────────────────────────────────────
class MultiFolderCSIDataset(Dataset):
    def __init__(self, data_dirs):
        self.samples = []
        self.labels  = []
        self.data_dirs = data_dirs
        self._load()

    def _load(self):
        print(bold('\n── Veri yükleniyor (Multi-Folder) ────────────────'))
        total_counts = {lbl: 0 for lbl in LABELS}
        
        for data_dir in self.data_dirs:
            if not os.path.exists(data_dir):
                print(yellow(f'  Klasör bulunamadı: {data_dir}'))
                continue
            
            print(f'  Kaynaktan okunuyor: {cyan(os.path.basename(data_dir))}')
            for lbl in LABELS:
                path = os.path.join(data_dir, f'{lbl}.jsonl')
                if not os.path.exists(path):
                    continue
                
                count = 0
                with open(path) as f:
                    for line in f:
                        try:
                            s = json.loads(line)
                        except:
                            continue

                        nodes = s.get('nodes', {})
                        node_keys = sorted(nodes.keys())
                        if not node_keys:
                            continue

                        # Prepare tensors for each node
                        node_tensors = []
                        # Take up to N_NODES
                        for i in range(N_NODES):
                            if i < len(node_keys):
                                nid = node_keys[i]
                                frames = np.array(nodes[nid], dtype=np.float32)
                                if frames.shape != (WINDOW_FRAMES, N_SC):
                                    # Pad or skip if corrupted
                                    continue
                                node_tensors.append(frames)
                            else:
                                # If missing a node, repeat the last one or zero pad
                                # Repeating the last node is usually better for CNNs than zeros
                                node_tensors.append(node_tensors[-1] if node_tensors else np.zeros((WINDOW_FRAMES, N_SC), dtype=np.float32))

                        if not node_tensors or len(node_tensors) < N_NODES:
                            continue

                        tensor = np.stack(node_tensors) # (N_NODES, 60, 64)
                        
                        # Local Z-score normalization
                        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8)

                        self.samples.append(tensor)
                        self.labels.append(LABEL2IDX[lbl])
                        count += 1
                
                total_counts[lbl] += count
                print(f'    {lbl:<12} +{count} örnek')

        print(bold('\n── Dataset Özeti ──────────────────────────'))
        for lbl, count in total_counts.items():
            bar = '█' * min(count // 50, 20)
            print(f'  {lbl:<12} {bar} {green(str(count))} toplam')
        print(f'  {"TOPLAM":<12} {bold(green(str(len(self.samples))))} örnek\n')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx],  dtype=torch.long)
        return x, y

# ── Model ─────────────────────────────────────────────────────────────────────
class AdvancedCSINet(nn.Module):
    def __init__(self, n_nodes=N_NODES, n_classes=len(LABELS)):
        super().__init__()
        
        # Shared CNN Backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # (30, 32)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # (15, 16)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) # (128, 4, 4)
        )
        
        feat_dim = 128 * 4 * 4
        
        # Fusion and Classifier
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * n_nodes, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: (B, N_NODES, 60, 64)
        batch_size = x.shape[0]
        node_features = []
        
        for i in range(x.shape[1]):
            node_x = x[:, i, :, :].unsqueeze(1) # (B, 1, 60, 64)
            feat = self.cnn(node_x)
            node_features.append(feat.view(batch_size, -1))
            
        combined = torch.cat(node_features, dim=1) # (B, feat_dim * N_NODES)
        fused = self.fusion(combined)
        logits = self.classifier(fused)
        return logits

# ── Training Loop ─────────────────────────────────────────────────────────────
def run_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{bold("Cihaz:")} {cyan(str(device))}')

    # Load Full Dataset
    full_dataset = MultiFolderCSIDataset(DATA_DIRS)
    if len(full_dataset) == 0:
        print(yellow("Veri bulunamadı. Lütfen dataset klasörlerini kontrol edin."))
        return

    # Split: 70% Train, 15% Val, 15% Test
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size   = int(0.15 * total_size)
    test_size  = total_size - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    model = AdvancedCSINet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    print(bold('── Eğitim Başlıyor ────────────────────────'))
    print(f'  Train: {train_size} | Val: {val_size} | Test: {test_size}')
    
    best_val_acc = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
            train_correct += (outputs.argmax(1) == y).sum().item()
            train_total += x.size(0)
            
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                val_correct += (outputs.argmax(1) == y).sum().item()
                val_total += x.size(0)
        
        val_acc = val_correct / val_total
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state': model.state_dict(),
                'labels': LABELS,
                'n_nodes': N_NODES
            }, MODEL_PATH)
            status = green(f'val_acc={val_acc*100:.1f}% ★')
        else:
            status = f'val_acc={val_acc*100:.1f}%'

        print(f'  Epoch {epoch:2d}/{EPOCHS} | loss: {train_loss/train_total:.4f} | train_acc: {train_correct/train_total*100:.1f}% | {status}')

    print(f'\n{bold(green("Eğitim Tamamlandı!"))} En iyi val_acc: {best_val_acc*100:.1f}%')

    # ── Final Testing with Confidence Values ──────────────────────────────────
    print(bold('\n── Test Seti Değerlendirmesi (Confidence) ──────'))
    model.load_state_dict(torch.load(MODEL_PATH)['model_state'])
    model.eval()
    
    all_preds = []
    all_probs = []
    all_trues = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1) # Get confidence values
            
            conf, preds = torch.max(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(conf.cpu().numpy())
            all_trues.extend(y.numpy())

    # Classification Report
    print(classification_report(all_trues, all_preds, target_names=LABELS))
    
    # Show some sample confidence values from test set
    print(bold('Sample Predictions with Confidence:'))
    for i in range(min(10, len(all_trues))):
        gt = LABELS[all_trues[i]]
        pd = LABELS[all_preds[i]]
        cf = all_probs[i] * 100
        result = green("CORRECT") if gt == pd else yellow("WRONG")
        print(f'  Target: {gt:<10} | Pred: {pd:<10} | Confidence: {cf:>5.1f}% | {result}')

    print(f'\nAdvanced model saved to: {cyan(MODEL_PATH)}')

if __name__ == '__main__':
    run_training()
