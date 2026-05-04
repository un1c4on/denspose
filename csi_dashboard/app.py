#!/usr/bin/env python3
"""CSI Dashboard — Flask + SocketIO + CNN çıkarım"""

import socket, struct, time, threading, collections, math, os
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO

# ── PyTorch + Model ───────────────────────────────────────────────────────────
import torch
import torch.nn as nn

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model.pt')
LABELS     = ['empty', 'present', 'walking', 'fall']
LABEL_TR   = {'empty': 'BOŞ ODA', 'present': 'VARLIK', 'walking': 'HAREKET', 'fall': 'DÜŞME'}

class CSINet(nn.Module):
    def __init__(self, n_nodes=3, n_classes=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 * 4 * 4 * n_nodes, 256),
            nn.ReLU(), nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x):
        batch = x.shape[0]
        feats = [self.cnn(x[:, i].unsqueeze(1)).view(batch, -1) for i in range(x.shape[1])]
        return self.classifier(self.fusion(torch.cat(feats, dim=1)))

# Model yükle
cnn_model = None
try:
    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    cnn_model = CSINet()
    cnn_model.load_state_dict(ckpt['model_state'])
    cnn_model.eval()
    print(f"✓ Model yüklendi: {MODEL_PATH}")
except Exception as e:
    print(f"⚠ Model yüklenemedi: {e} — kural tabanlı mod aktif")

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'csi_secret'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

# ── Sabitler ──────────────────────────────────────────────────────────────────
UDP_PORT      = 5005
MAGIC         = 0xC5110001
HEADER_FMT    = '<IBBHIIBB2x'
HEADER_SIZE   = 20
FS            = 20
BUF_SIZE      = 300
WINDOW_FRAMES = 60
N_SC          = 64
INFER_INTERVAL= 1.5   # saniyede bir çıkarım

# ── Eşikler ───────────────────────────────────────────────────────────────────
thresholds = {
    'motion_var':   3.0,
    'presence_diff':0.8,
    'fall_spike':   25.0,
    'fall_drop':    8.0,
    'fall_window':  1.5,
    'debounce':     2.5,
}

# ── Paylaşılan durum ──────────────────────────────────────────────────────────
nodes       = {}
events      = collections.deque(maxlen=50)
lock        = threading.Lock()
infer_bufs  = {}   # node_id → deque(maxlen=WINDOW_FRAMES) — ham amp vektörleri
last_infer  = 0.0
cnn_result  = {'label': '—', 'label_tr': '—', 'probs': [0.25]*4, 'conf': 0.0}

# ── Parser ────────────────────────────────────────────────────────────────────
def parse(data):
    if len(data) < HEADER_SIZE:
        return None
    magic, node_id, n_ant, n_sc, freq, seq, rssi_u8, noise_u8 = struct.unpack(
        HEADER_FMT, data[:HEADER_SIZE])
    if magic != MAGIC:
        return None
    rssi  = rssi_u8  - 256 if rssi_u8  > 127 else rssi_u8
    noise = noise_u8 - 256 if noise_u8 > 127 else noise_u8
    iq = data[HEADER_SIZE:]
    amps = []
    for i in range(0, min(len(iq)-1, n_ant * n_sc * 2), 2):
        I = struct.unpack('b', bytes([iq[i]]))[0]
        Q = struct.unpack('b', bytes([iq[i+1]]))[0]
        amps.append(math.sqrt(I*I + Q*Q))
    mean = float(np.mean(amps)) if amps else 0.0
    return {'node': node_id, 'seq': seq, 'rssi': rssi,
            'noise': noise, 'freq': freq, 'n_sc': n_sc,
            'mean': mean, 'amps': amps[:N_SC], 'ts': time.time()}

# ── CNN çıkarım ───────────────────────────────────────────────────────────────
def run_inference(node_buffers):
    global cnn_result
    if cnn_model is None:
        return

    # En az 2 node ve her birinde WINDOW_FRAMES frame olmalı
    ready = sorted([n for n, b in node_buffers.items() if len(b) == WINDOW_FRAMES])
    if len(ready) < 2:
        return

    node_tensors = []
    for nid in ready[:2]:
        frames = np.array(list(node_buffers[nid]), dtype=np.float32)  # (60, 64)
        node_tensors.append(frames)

    tensor = np.stack(node_tensors)           # (2, 60, 64)
    mean   = tensor.mean()
    std    = tensor.std() + 1e-8
    tensor = (tensor - mean) / std

    x = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)  # (1, 2, 60, 64)

    with torch.no_grad():
        logits = cnn_model(x)
        probs  = torch.softmax(logits, dim=1)[0].numpy().tolist()

    idx   = int(np.argmax(probs))
    label = LABELS[idx]
    conf  = probs[idx]

    cnn_result = {
        'label':    label,
        'label_tr': LABEL_TR[label],
        'probs':    [round(p, 3) for p in probs],
        'conf':     round(conf, 3),
    }

    # Düşme olayı CNN'den geldi
    if label == 'fall' and conf > 0.8:
        events.appendleft({'type': 'fall', 'source': 'CNN',
                           'ts': time.strftime('%H:%M:%S'),
                           'amp': round(conf * 100, 1)})

    socketio.emit('cnn_result', cnn_result)

# ── Durum makinesi ─────────────────────────────────────────────────────────────
class NodeState:
    def __init__(self, nid):
        self.nid = nid
        self.buf = collections.deque(maxlen=BUF_SIZE)
        self.ts_buf = collections.deque(maxlen=BUF_SIZE)
        self.baseline = None
        self.cal_buf = []
        self.status = 'KALİBRASYON'
        self.last_mean = 0
        self.peak_ts = 0
        self.peak_val = 0
        self.rssi = 0
        self.seq = 0
        self.n_sc = 0
        self.pkt_count = 0
        self.rate = 0.0
        self.rate_ts = time.time()
        self.rate_cnt = 0
        self.candidate_status = 'BOŞ ODA'
        self.candidate_since = time.time()
        self.confirmed_status = 'BOŞ ODA'
        self.ema = None
        self.EMA_A = 0.15

    def update(self, f):
        self.seq = f['seq']
        self.rssi = f['rssi']
        self.n_sc = f['n_sc']
        self.last_mean = f['mean']
        self.pkt_count += 1
        self.rate_cnt += 1
        now = time.time()
        if now - self.rate_ts >= 1.0:
            self.rate = self.rate_cnt / (now - self.rate_ts)
            self.rate_cnt = 0
            self.rate_ts = now

        if self.ema is None:
            self.ema = f['mean']
        else:
            self.ema = self.EMA_A * f['mean'] + (1 - self.EMA_A) * self.ema

        self.buf.append(self.ema)
        self.ts_buf.append(f['ts'])

        if self.baseline is None:
            self.cal_buf.append(self.ema)
            if len(self.cal_buf) >= 80:
                self.baseline = float(np.mean(self.cal_buf))
                self.candidate_status = 'BOŞ ODA'
                self.confirmed_status = 'BOŞ ODA'
                self.status = 'BOŞ ODA'
            return 'KALİBRASYON', None
        return self._detect(f)

    def _detect(self, f):
        buf = list(self.buf)
        thr = thresholds
        now = f['ts']
        event = None

        win  = buf[-60:] if len(buf) >= 60 else buf
        var  = float(np.var(win))
        diff = abs(self.ema - self.baseline)

        if f['mean'] > self.peak_val:
            self.peak_val = f['mean']
            self.peak_ts = now

        if (self.peak_val > thr['fall_spike'] and
                f['mean'] < self.peak_val - thr['fall_drop'] and
                now - self.peak_ts < thr['fall_window']):
            self.peak_val = 0
            self.status = 'DÜŞME TESPİT'
            self.confirmed_status = 'DÜŞME TESPİT'
            self.candidate_status = 'DÜŞME TESPİT'
            event = {'type': 'fall', 'source': 'kural', 'node': self.nid,
                     'ts': time.strftime('%H:%M:%S'), 'amp': round(f['mean'], 1)}
            return self.status, event

        if var > thr['motion_var']:
            raw = 'HAREKET VAR'
        elif diff > thr['presence_diff']:
            raw = 'VARLIK TESPİT'
        else:
            raw = 'BOŞ ODA'

        DEBOUNCE_SEC = thresholds.get('debounce', 2.5)
        if raw == self.candidate_status:
            if now - self.candidate_since >= DEBOUNCE_SEC:
                if self.confirmed_status != raw:
                    self.confirmed_status = raw
        else:
            self.candidate_status = raw
            self.candidate_since = now

        self.status = self.confirmed_status
        return self.status, event

# ── UDP dinleyici ─────────────────────────────────────────────────────────────
def udp_thread():
    global last_infer
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', UDP_PORT))
    sock.settimeout(0.5)
    print(f"UDP :{UDP_PORT} dinleniyor...")

    while True:
        try:
            data, addr = sock.recvfrom(8192)
        except socket.timeout:
            continue
        f = parse(data)
        if f is None:
            continue

        with lock:
            nid = f['node']
            if nid not in nodes:
                nodes[nid] = NodeState(nid)
            if nid not in infer_bufs:
                infer_bufs[nid] = collections.deque(maxlen=WINDOW_FRAMES)

            ns = nodes[nid]
            status, event = ns.update(f)

            # CNN için ham amp buffer
            if len(f['amps']) == N_SC:
                infer_bufs[nid].append(f['amps'])

            if event:
                events.appendleft(event)

            payload = {
                'node':   nid,
                'seq':    f['seq'],
                'rssi':   f['rssi'],
                'mean':   round(f['mean'], 2),
                'status': ns.status,
                'var':    round(float(np.var(list(ns.buf)[-20:])) if len(ns.buf) >= 20 else 0, 2),
                'buf':    [round(x, 1) for x in list(ns.buf)[-100:]],
                'rate':   round(ns.rate, 1),
                'pkt':    ns.pkt_count,
                'event':  event,
                'cnn':    cnn_result,
            }

        socketio.emit('frame', payload)

        # CNN çıkarımı — her INFER_INTERVAL saniyede bir
        now = time.time()
        if now - last_infer >= INFER_INTERVAL:
            last_infer = now
            with lock:
                bufs_snapshot = {n: collections.deque(b) for n, b in infer_bufs.items()}
            threading.Thread(target=run_inference, args=(bufs_snapshot,), daemon=True).start()

# ── SocketIO events ───────────────────────────────────────────────────────────
@socketio.on('set_threshold')
def on_threshold(data):
    key = data.get('key')
    val = data.get('val')
    if key in thresholds:
        thresholds[key] = float(val)
        socketio.emit('threshold_ack', {'key': key, 'val': val})

@socketio.on('recalibrate')
def on_recalibrate(data):
    nid = int(data.get('node', 0))
    with lock:
        if nid in nodes:
            nodes[nid].baseline = None
            nodes[nid].cal_buf  = []
            nodes[nid].status   = 'KALİBRASYON'
    socketio.emit('recal_ack', {'node': nid})

@socketio.on('get_events')
def on_get_events():
    socketio.emit('events', list(events))

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    t = threading.Thread(target=udp_thread, daemon=True)
    t.start()
    print("Dashboard: http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
