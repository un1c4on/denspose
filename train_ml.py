#!/usr/bin/env python3
"""
FFT-Enhanced Machine Learning CSI Classifier
============================================
Adds Frequency Domain features (FFT) to better distinguish 
rhythmic walking from sudden bursts (Falls).
"""

import json
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from scipy.stats import kurtosis, skew

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIRS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_2')
]
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_model.joblib')
LABELS     = ['empty', 'present', 'walking', 'fall']
LABEL2IDX  = {l: i for i, l in enumerate(LABELS)}
N_NODES    = 2
WINDOW_FRAMES = 60
N_SC       = 64

# ── Feature Extraction ────────────────────────────────────────────────────────
def extract_features(node_data):
    all_features = []
    for n in range(len(node_data)):
        node_window = node_data[n] # (60, 64)
        
        # 1. Time Domain Stats
        means = np.mean(node_window, axis=0)
        stds  = np.std(node_window, axis=0)
        diffs = np.mean(np.abs(np.diff(node_window, axis=0)), axis=0)
        
        # 2. Global Time Domain Stats
        glob_std  = np.std(node_window)
        glob_kurt = kurtosis(node_window.flatten())
        glob_max  = np.max(node_window)
        
        # 3. Frequency Domain Stats (FFT)
        # Compute FFT across time for each subcarrier
        fft_vals = np.abs(np.fft.rfft(node_window, axis=0)) # (31, 64)
        # Energy in bands
        walking_band_energy = np.mean(fft_vals[1:6, :]) # ~0.3Hz to 2Hz
        high_freq_energy    = np.mean(fft_vals[10:, :]) # High frequency noise
        
        node_feats = np.concatenate([
            means, stds, diffs, 
            [glob_std, glob_kurt, glob_max, walking_band_energy, high_freq_energy]
        ])
        all_features.append(node_feats)
        
    return np.concatenate(all_features)

# ── Data Loading ──────────────────────────────────────────────────────────────
def load_data():
    X, y = [], []
    print(bold('\n── Veri yükleniyor (FFT Features) ───────────────'))
    
    for ddir in DATA_DIRS:
        if not os.path.exists(ddir): continue
        print(f'  Kaynaktan okunuyor: {cyan(os.path.basename(ddir))}')
        
        for lbl in LABELS:
            path = os.path.join(ddir, f'{lbl}.jsonl')
            if not os.path.exists(path): continue
            
            count = 0
            with open(path) as f:
                for line in f:
                    try:
                        s = json.loads(line)
                        nodes = s.get('nodes', {})
                        node_keys = sorted(nodes.keys())
                        if not node_keys: continue
                        node_data = []
                        for i in range(N_NODES):
                            if i < len(node_keys):
                                frames = np.array(nodes[node_keys[i]], dtype=np.float32)
                                if frames.shape == (WINDOW_FRAMES, N_SC): node_data.append(frames)
                            else:
                                node_data.append(node_data[-1] if node_data else np.zeros((WINDOW_FRAMES, N_SC)))
                        if len(node_data) < N_NODES: continue
                        node_data = np.array(node_data)
                        node_data = (node_data - node_data.mean()) / (node_data.std() + 1e-8)
                        X.append(extract_features(node_data))
                        y.append(LABEL2IDX[lbl])
                        count += 1
                    except: continue
            print(f'    {lbl:<12} +{count} örnek')
    return np.array(X), np.array(y)

def bold(s): return f'\033[1m{s}\033[0m'
def cyan(s): return f'\033[96m{s}\033[0m'
def green(s): return f'\033[92m{s}\033[0m'

def main():
    X, y = load_data()
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Neutral Weights: Let the features (FFT) do the work
    custom_weights = {i: 1.0 for i in range(len(LABELS))}
    
    print(f"  Kullanılan Ağırlıklar (Nötr): { {LABELS[i]: w for i, w in custom_weights.items()} }")

    # Using Random Forest with neutral weights for stability
    model = RandomForestClassifier(n_estimators=300, max_depth=15, 
                                   min_samples_leaf=5, # Higher regularization
                                   n_jobs=-1, random_state=42, verbose=1, 
                                   class_weight=custom_weights)
    model.fit(X_train, y_train)

    print(f'\n  Validation Acc: {model.score(X_val, y_val)*100:.2f}%')
    print(classification_report(y_test, model.predict(X_test), target_names=LABELS))
    joblib.dump(model, MODEL_PATH)

if __name__ == '__main__':
    main()
