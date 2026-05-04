#!/usr/bin/env python3
"""
CSI CNN Eğitim Scripti

Kullanım:
  python3 train.py

Çıktı:
  model.pt  — eğitilmiş model ağırlıkları
"""

import json, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix

# ── Sabitler ──────────────────────────────────────────────────────────────────
SAVE_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
MODEL_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pt')
LABELS       = ['empty', 'present', 'walking', 'fall']
LABEL2IDX    = {l: i for i, l in enumerate(LABELS)}
N_SC         = 64
WINDOW_FRAMES = 60   # 3s × 20Hz
N_NODES      = 3
EPOCHS       = 40
BATCH_SIZE   = 32
LR           = 1e-3

# Renk
def green(s):  return f'\033[92m{s}\033[0m'
def yellow(s): return f'\033[93m{s}\033[0m'
def cyan(s):   return f'\033[96m{s}\033[0m'
def bold(s):   return f'\033[1m{s}\033[0m'

# ── Dataset ───────────────────────────────────────────────────────────────────
class CSIDataset(Dataset):
    def __init__(self):
        self.samples = []
        self.labels  = []
        self._load()

    def _load(self):
        print(bold('\n── Veri yükleniyor ─────────────────────────'))
        for lbl in LABELS:
            path = os.path.join(SAVE_DIR, f'{lbl}.jsonl')
            if not os.path.exists(path):
                print(yellow(f'  {lbl:<12} — dosya yok, atlandı'))
                continue
            count = 0
            with open(path) as f:
                for line in f:
                    try:
                        s = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    nodes = s.get('nodes', {})
                    # Her node'dan WINDOW_FRAMES×N_SC matris al
                    node_tensors = []
                    for nid in sorted(nodes.keys())[:N_NODES]:
                        frames = np.array(nodes[nid], dtype=np.float32)  # (60, 64)
                        if frames.shape != (WINDOW_FRAMES, N_SC):
                            continue
                        node_tensors.append(frames)

                    if len(node_tensors) == 0:
                        continue

                    # Birden fazla node varsa ortala, yoksa tekrar et
                    if len(node_tensors) >= N_NODES:
                        tensor = np.stack(node_tensors[:N_NODES])  # (N_NODES, 60, 64)
                    else:
                        tensor = np.stack([node_tensors[0]] * N_NODES)

                    # Z-score normalizasyon
                    mean = tensor.mean()
                    std  = tensor.std() + 1e-8
                    tensor = (tensor - mean) / std

                    self.samples.append(tensor)
                    self.labels.append(LABEL2IDX[lbl])
                    count += 1

            bar = '█' * min(count // 10, 30)
            print(f'  {lbl:<12} {bar} {green(str(count))} örnek')

        print(f'  {"TOPLAM":<12} {green(str(len(self.samples)))} örnek\n')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx],  dtype=torch.long)
        return x, y


# ── Model: CNN + LSTM ─────────────────────────────────────────────────────────
class CSINet(nn.Module):
    """
    Giriş: (batch, N_NODES, 60, 64)
    CNN ile uzamsal özellik → LSTM ile zamansal özellik → 4 sınıf
    """
    def __init__(self, n_nodes=N_NODES, n_classes=4):
        super().__init__()

        # Her node için paylaşılan CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),          # (32, 30, 32)

            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),          # (64, 15, 16)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), # (128, 4, 4) = 2048
        )

        cnn_out = 128 * 4 * 4  # 2048

        # Node özelliklerini birleştir
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out * n_nodes, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x):
        # x: (B, N_NODES, 60, 64)
        batch = x.shape[0]
        node_feats = []
        for i in range(x.shape[1]):
            xi = x[:, i, :, :].unsqueeze(1)  # (B, 1, 60, 64)
            fi = self.cnn(xi).view(batch, -1)  # (B, 2048)
            node_feats.append(fi)

        combined = torch.cat(node_feats, dim=1)  # (B, 2048*N_NODES)
        out = self.fusion(combined)
        return self.classifier(out)


# ── Eğitim ────────────────────────────────────────────────────────────────────
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{bold("Cihaz:")} {cyan(str(device))}')

    dataset = CSIDataset()
    if len(dataset) == 0:
        print(yellow('Hiç veri bulunamadı. Önce collect_data.py çalıştır.'))
        return

    # %80 eğitim, %20 test
    n_train = int(0.8 * len(dataset))
    n_test  = len(dataset) - n_train
    train_ds, test_ds = random_split(dataset, [n_train, n_test],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = CSINet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    print(bold('── Eğitim başlıyor ────────────────────────'))
    print(f'  Eğitim: {n_train}  |  Test: {n_test}  |  Epoch: {EPOCHS}\n')

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        # Eğitim
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct    += (out.argmax(1) == y).sum().item()
            total      += x.size(0)
        scheduler.step()

        train_acc  = correct / total * 100
        train_loss = total_loss / total

        # Değerlendirme
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total   += x.size(0)
        val_acc = correct / total * 100

        # En iyi modeli kaydet
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model_state': model.state_dict(),
                        'labels': LABELS,
                        'n_nodes': N_NODES}, MODEL_PATH)
            marker = green(' ✓ kaydedildi')
        else:
            marker = ''

        print(f'  Epoch {epoch:>3}/{EPOCHS}  '
              f'loss={train_loss:.4f}  '
              f'train={train_acc:.1f}%  '
              f'val={cyan(f"{val_acc:.1f}%")}'
              f'{marker}')

    print(f'\n{green("✓ Eğitim tamamlandı!")} En iyi doğruluk: {green(f"{best_acc:.1f}%")}')
    print(f'  Model: {MODEL_PATH}\n')

    # Detaylı rapor
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            all_preds.extend(model(x).argmax(1).cpu().numpy())
            all_true.extend(y.numpy())

    print(bold('── Test Sonuçları ──────────────────────────'))
    print(classification_report(all_true, all_preds, target_names=LABELS))

    cm = confusion_matrix(all_true, all_preds)
    print(bold('Karışıklık Matrisi:'))
    header = ''.join(f'{l:>10}' for l in LABELS)
    print(f'{"":>10}{header}')
    for i, row in enumerate(cm):
        cells = ''.join(f'{v:>10}' for v in row)
        print(f'{LABELS[i]:>10}{cells}')

    print(f'\n  Çıkarım için: {cyan("python3 predict.py")}')


if __name__ == '__main__':
    train()
