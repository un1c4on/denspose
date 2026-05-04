#!/usr/bin/env python3
"""
Real-time CSI Predictor (FFT-Enhanced)
======================================
Uses Frequency Domain features to filter out Walking rhythms
from Fall detections.
"""

import socket
import struct
import math
import collections
import numpy as np
import joblib
import os
import time
import sys
from scipy.stats import kurtosis, skew

# ── Configuration ─────────────────────────────────────────────────────────────
UDP_PORT      = 5005
MAGIC         = 0xC5110001
HEADER_FMT    = '<IBBHIIBB2x'
HEADER_SIZE   = 20
WINDOW_FRAMES = 60
N_SC          = 64
N_NODES       = 2
MODEL_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_model.joblib')
LABELS        = ['empty', 'present', 'walking', 'fall']

# Styling
def green(s):  return f'\033[92m{s}\033[0m'
def yellow(s): return f'\033[93m{s}\033[0m'
def cyan(s):   return f'\033[96m{s}\033[0m'
def red(s):    return f'\033[91m{s}\033[0m'
def bold(s):   return f'\033[1m{s}\033[0m'

# ── Feature Extraction ────────────────────────────────────────────────────────
def extract_features(node_data):
    all_features = []
    for n in range(len(node_data)):
        node_window = node_data[n]
        means = np.mean(node_window, axis=0)
        stds  = np.std(node_window, axis=0)
        diffs = np.mean(np.abs(np.diff(node_window, axis=0)), axis=0)
        
        glob_std  = np.std(node_window)
        glob_kurt = kurtosis(node_window.flatten())
        glob_max  = np.max(node_window)
        
        # FFT Features
        fft_vals = np.abs(np.fft.rfft(node_window, axis=0))
        walking_band_energy = np.mean(fft_vals[1:6, :]) 
        high_freq_energy    = np.mean(fft_vals[10:, :])
        
        node_feats = np.concatenate([means, stds, diffs, [glob_std, glob_kurt, glob_max, walking_band_energy, high_freq_energy]])
        all_features.append(node_feats)
    return np.concatenate(all_features)

def parse_packet(data: bytes):
    if len(data) < HEADER_SIZE: return None
    magic, node_id, n_ant, n_sc, freq, seq, rssi_u8, _ = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
    if magic != MAGIC: return None
    iq = data[HEADER_SIZE:]
    amps = []
    for i in range(0, min(len(iq) - 1, n_ant * n_sc * 2), 2):
        I, Q = struct.unpack('b', bytes([iq[i]]))[0], struct.unpack('b', bytes([iq[i+1]]))[0]
        amps.append(math.sqrt(I*I + Q*Q))
    if len(amps) < N_SC: amps += [0.0] * (N_SC - len(amps))
    return {'amps': amps[:N_SC]}

def main():
    model = joblib.load(MODEL_PATH)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', UDP_PORT))
    sock.settimeout(1.0)
    buffers = {}
    last_predict_time = 0
    
    print(bold('\n  FFT-Enhanced Multi-Class Monitor'))

    try:
        while True:
            try:
                data, addr = sock.recvfrom(8192)
            except socket.timeout: continue
            f = parse_packet(data)
            if not f: continue
            ip = addr[0]
            if ip not in buffers: buffers[ip] = collections.deque(maxlen=WINDOW_FRAMES)
            buffers[ip].append(f['amps'])
            ready_nodes = [ip for ip, b in buffers.items() if len(b) == WINDOW_FRAMES]
            
            if len(ready_nodes) >= N_NODES:
                now = time.time()
                if now - last_predict_time > 0.3:
                    last_predict_time = now
                    node_keys = sorted(ready_nodes)[:N_NODES]
                    node_data = np.array([list(buffers[k]) for k in node_keys])
                    node_data = (node_data - node_data.mean()) / (node_data.std() + 1e-8)
                    feat = extract_features(node_data).reshape(1, -1)
                    
                    probs = model.predict_proba(feat)[0]
                    if hasattr(model, 'verbose'): model.verbose = 0
                    
                    # --- Rhythm Suppression Logic ---
                    # If there is a strong walking rhythm (high FFT in walking band), 
                    # we drastically reduce the chance of a false 'fall' alarm.
                    # This uses the feature we just added.
                    
                    # Extract the walking energy from the feature vector (index depends on extract_features)
                    # For Node 0: glob_std is idx 192+0, kurt 193, max 194, walking_energy 195
                    # Let's use a simpler way: recalculate walking energy for suppression
                    fft_vals = np.abs(np.fft.rfft(node_data, axis=1)) # (N_NODES, 31, 64)
                    avg_walking_rhythm = np.mean(fft_vals[:, 1:6, :]) 
                    
                    if avg_walking_rhythm > 0.4: # Threshold for 'distinct walking rhythm'
                        # Suppress FALL by 80% if we are clearly walking
                        probs[3] *= 0.2 
                        # Re-normalize
                        probs /= np.sum(probs)

                    pred_idx = np.argmax(probs)
                    chosen_label = LABELS[pred_idx]
                    chosen_color = green if chosen_label == 'empty' else yellow if chosen_label == 'present' else cyan if chosen_label == 'walking' else red
                    
                    out = "\r"
                    for idx, lbl in enumerate(LABELS):
                        p = probs[idx] * 100
                        base_clr = green if lbl == 'empty' else yellow if lbl == 'present' else cyan if lbl == 'walking' else red
                        val_str = f"{p:>4.1f}%"
                        styled_val = bold(base_clr(val_str)) if p > 50 else base_clr(val_str)
                        out += f" {lbl.upper()}: {styled_val} |"
                    
                    # Highlight the CHOSEN one at the end
                    out += f"  {bold('==>')} [{bold(chosen_color(chosen_label.upper()))}]      "
                    print(out, end='', flush=True)

    except KeyboardInterrupt: pass
    finally: sock.close()

if __name__ == '__main__':
    main()
