#!/usr/bin/env python3
"""CSI + Sensör Dashboard — Flask + SocketIO + AdvancedCSINet"""

import socket, struct, time, threading, collections, math, os, re
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO
import torch
import torch.nn as nn

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model_advanced.pt')
LABELS   = ['empty', 'present', 'walking', 'fall']
LABEL_TR = {'empty': 'BOŞ ODA', 'present': 'VARLIK', 'walking': 'YÜRÜYOR', 'fall': 'DÜŞME'}

class AdvancedCSINet(nn.Module):
    def __init__(self, n_nodes=2, n_classes=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        feat_dim = 128 * 4 * 4
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * n_nodes, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.ReLU()
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        b = x.shape[0]
        feats = [self.cnn(x[:, i].unsqueeze(1)).view(b, -1) for i in range(x.shape[1])]
        return self.classifier(self.fusion(torch.cat(feats, dim=1)))

cnn_model   = None
cnn_n_nodes = 2
try:
    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    cnn_n_nodes = ckpt.get('n_nodes', 2)
    cnn_model = AdvancedCSINet(n_nodes=cnn_n_nodes)
    cnn_model.load_state_dict(ckpt['model_state'])
    cnn_model.eval()
    print(f"✓ model_advanced.pt yüklendi (n_nodes={cnn_n_nodes})")
except Exception as e:
    print(f"⚠ Model yüklenemedi: {e}")

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'csi_secret'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

# ── Sabitler ──────────────────────────────────────────────────────────────────
UDP_PORT      = 5005
MAGIC         = 0xC5110001
HEADER_FMT    = '<IBBHIIBB2x'
HEADER_SIZE   = 20
BUF_SIZE      = 300
WINDOW_FRAMES = 60
N_SC          = 64
INFER_INTERVAL= 1.5
SERIAL_PORT   = '/dev/ttyACM1'
SERIAL_BAUD   = 115200
HIST_LEN      = 60

# ── Paylaşılan durum ──────────────────────────────────────────────────────────
nodes      = {}
events     = collections.deque(maxlen=50)
lock       = threading.Lock()
infer_bufs = {}
last_infer = 0.0

cnn_result = {'label': '—', 'label_tr': '—', 'probs': [0.25]*4, 'conf': 0.0}

sensor_state = {
    'connected': False,
    'ax': 0.0, 'ay': 0.0, 'az': 0.0,
    'gx': 0.0, 'gy': 0.0, 'gz': 0.0,
    'mag': 0.0,
    'hr': 0.0,
    'spo2': 0,
    'ax_hist':  collections.deque([0.0]*HIST_LEN, maxlen=HIST_LEN),
    'ay_hist':  collections.deque([0.0]*HIST_LEN, maxlen=HIST_LEN),
    'az_hist':  collections.deque([0.0]*HIST_LEN, maxlen=HIST_LEN),
    'mag_hist': collections.deque([1.0]*HIST_LEN, maxlen=HIST_LEN),
    'gx_hist':  collections.deque([0.0]*HIST_LEN, maxlen=HIST_LEN),
    'gy_hist':  collections.deque([0.0]*HIST_LEN, maxlen=HIST_LEN),
    'gz_hist':  collections.deque([0.0]*HIST_LEN, maxlen=HIST_LEN),
    'hr_hist':  collections.deque([0.0]*HIST_LEN, maxlen=HIST_LEN),
}

combined = {'label': '—', 'label_tr': '—', 'conf': 0.0, 'level': '—', 'sources': []}
_fall_clear_timer = None

# ── 5 saniye sonra düşme alarmını kaldır ──────────────────────────────────────
def _schedule_fall_clear():
    global _fall_clear_timer, combined
    def _do_clear():
        global combined
        lbl = cnn_result['label']
        if lbl and lbl != 'fall':
            combined = {'label': lbl, 'label_tr': LABEL_TR.get(lbl, ''),
                        'conf': cnn_result['conf'], 'level': 'ORTA', 'sources': ['CSI']}
        else:
            combined = {'label': 'present', 'label_tr': 'VARLIK',
                        'conf': 0.5, 'level': 'ORTA', 'sources': ['CSI']}
        socketio.emit('fall_clear', {})
        socketio.emit('combined', combined)
    if _fall_clear_timer is not None:
        try: _fall_clear_timer.cancel()
        except: pass
    t = threading.Timer(5.0, _do_clear)
    t.daemon = True
    t.start()
    _fall_clear_timer = t

# ── Birleşik karar ────────────────────────────────────────────────────────────
def update_combined():
    global combined
    lbl  = cnn_result['label']
    conf = cnn_result['conf']
    mag  = sensor_state['mag']
    sens = sensor_state['connected']

    if lbl == 'fall' and sens and mag > 1.8:
        combined = {'label': 'fall', 'label_tr': 'DÜŞME', 'conf': conf,
                    'level': 'YÜKSEK', 'sources': ['CSI', 'İvme']}
        _add_event('fall', 'Birleşik', round(mag, 2))
        _schedule_fall_clear()
    elif lbl == 'fall':
        combined = {'label': 'fall', 'label_tr': 'DÜŞME', 'conf': conf,
                    'level': 'ORTA', 'sources': ['CSI']}
        _schedule_fall_clear()
    elif sens and mag > 2.5:
        combined = {'label': 'fall', 'label_tr': 'DÜŞME', 'conf': 0.7,
                    'level': 'ORTA', 'sources': ['İvme']}
        _add_event('fall', 'İvme', round(mag, 2))
        _schedule_fall_clear()
    elif lbl == 'walking':
        if sens and mag > 1.05:
            combined = {'label': 'walking', 'label_tr': 'YÜRÜYOR', 'conf': conf,
                        'level': 'YÜKSEK', 'sources': ['CSI', 'İvme']}
        else:
            combined = {'label': 'walking', 'label_tr': 'YÜRÜYOR', 'conf': conf,
                        'level': 'ORTA', 'sources': ['CSI']}
    elif lbl == 'present':
        combined = {'label': 'present', 'label_tr': 'VARLIK', 'conf': conf,
                    'level': 'YÜKSEK', 'sources': ['CSI']}
    elif lbl == 'empty':
        combined = {'label': 'empty', 'label_tr': 'BOŞ ODA', 'conf': conf,
                    'level': 'YÜKSEK', 'sources': ['CSI']}
    else:
        combined = {'label': '—', 'label_tr': '—', 'conf': 0.0, 'level': '—', 'sources': []}

def _add_event(ev_type, source, amp):
    events.appendleft({'type': ev_type, 'source': source,
                       'ts': time.strftime('%H:%M:%S'), 'amp': amp})

# ── UDP Parser ────────────────────────────────────────────────────────────────
def parse(data):
    if len(data) < HEADER_SIZE:
        return None
    magic, node_id, n_ant, n_sc, freq, seq, rssi_u8, noise_u8 = struct.unpack(
        HEADER_FMT, data[:HEADER_SIZE])
    if magic != MAGIC:
        return None
    rssi = rssi_u8 - 256 if rssi_u8 > 127 else rssi_u8
    iq = data[HEADER_SIZE:]
    amps = []
    for i in range(0, min(len(iq)-1, n_ant * n_sc * 2), 2):
        I = struct.unpack('b', bytes([iq[i]]))[0]
        Q = struct.unpack('b', bytes([iq[i+1]]))[0]
        amps.append(math.sqrt(I*I + Q*Q))
    mean = float(np.mean(amps)) if amps else 0.0
    return {'node': node_id, 'seq': seq, 'rssi': rssi,
            'n_sc': n_sc, 'mean': mean, 'amps': amps[:N_SC], 'ts': time.time()}

# ── CNN çıkarım ───────────────────────────────────────────────────────────────
def run_inference(snap):
    global cnn_result
    if cnn_model is None:
        return
    ready = sorted([n for n, b in snap.items() if len(b) == WINDOW_FRAMES])
    if len(ready) < 2:
        return
    tensors = [np.array(list(snap[nid]), dtype=np.float32) for nid in ready[:2]]
    t = np.stack(tensors)
    t = (t - t.mean()) / (t.std() + 1e-8)
    x = torch.tensor(t, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(cnn_model(x), dim=1)[0].numpy().tolist()
    idx = int(np.argmax(probs))
    cnn_result = {
        'label':    LABELS[idx],
        'label_tr': LABEL_TR[LABELS[idx]],
        'probs':    [round(p, 3) for p in probs],
        'conf':     round(probs[idx], 3),
    }
    update_combined()
    socketio.emit('cnn_result', cnn_result)
    socketio.emit('combined', combined)
    socketio.emit('events_update', list(events))
    if LABELS[idx] == 'fall' and probs[idx] > 0.8:
        _add_event('fall', 'CNN', round(probs[idx]*100, 1))

# ── Node state ────────────────────────────────────────────────────────────────
class NodeState:
    def __init__(self, nid):
        self.nid = nid
        self.buf = collections.deque(maxlen=BUF_SIZE)
        self.baseline = None
        self.cal_buf = []
        self.status = 'KALİBRASYON'
        self.ema = None
        self.EMA_A = 0.15
        self.rssi = 0
        self.seq = 0
        self.n_sc = 0
        self.pkt_count = 0
        self.rate = 0.0
        self.rate_ts = time.time()
        self.rate_cnt = 0

    def update(self, f):
        self.seq = f['seq']
        self.rssi = f['rssi']
        self.n_sc = f['n_sc']
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
        if self.baseline is None:
            self.cal_buf.append(self.ema)
            if len(self.cal_buf) >= 80:
                self.baseline = float(np.mean(self.cal_buf))
                self.status = 'BOŞ ODA'
            else:
                self.status = 'KALİBRASYON'

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
            ns.update(f)
            if len(f['amps']) == N_SC:
                infer_bufs[nid].append(f['amps'])
            payload = {
                'node':  nid, 'seq': f['seq'], 'rssi': f['rssi'],
                'mean':  round(f['mean'], 2), 'status': ns.status,
                'var':   round(float(np.var(list(ns.buf)[-20:])) if len(ns.buf) >= 20 else 0, 2),
                'buf':   [round(x, 1) for x in list(ns.buf)[-100:]],
                'rate':  round(ns.rate, 1), 'pkt': ns.pkt_count,
            }
        socketio.emit('frame', payload)
        now = time.time()
        if now - last_infer >= INFER_INTERVAL:
            last_infer = now
            with lock:
                snap = {n: collections.deque(b) for n, b in infer_bufs.items()}
            threading.Thread(target=run_inference, args=(snap,), daemon=True).start()

# ── Serial dinleyici (MPU6050 + MAX30100) ─────────────────────────────────────
def serial_thread():
    import serial as pyserial
    while True:
        try:
            s = pyserial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
            sensor_state['connected'] = True
            print(f"✓ Sensör bağlandı: {SERIAL_PORT}")
            socketio.emit('sensor_status', {'connected': True})
            while True:
                line = s.readline().decode('utf-8', errors='replace').strip()
                if not line:
                    continue

                # Akselerometre
                m = re.match(r'Ax:\s*([-\d.]+)\s+Ay:\s*([-\d.]+)\s+Az:\s*([-\d.]+)', line)
                if m:
                    ax, ay, az = float(m[1]), float(m[2]), float(m[3])
                    mag = math.sqrt(ax*ax + ay*ay + az*az)
                    with lock:
                        sensor_state['ax'] = ax
                        sensor_state['ay'] = ay
                        sensor_state['az'] = az
                        sensor_state['mag'] = mag
                        sensor_state['ax_hist'].append(ax)
                        sensor_state['ay_hist'].append(ay)
                        sensor_state['az_hist'].append(az)
                        sensor_state['mag_hist'].append(mag)
                        payload = {
                            'ax': ax, 'ay': ay, 'az': az, 'mag': round(mag, 3),
                            'ax_hist':  list(sensor_state['ax_hist']),
                            'ay_hist':  list(sensor_state['ay_hist']),
                            'az_hist':  list(sensor_state['az_hist']),
                            'mag_hist': list(sensor_state['mag_hist']),
                        }
                    print(f"[SENSOR] accel ax={ax:.2f} mag={mag:.2f}", flush=True)
                    socketio.emit('accel', payload, namespace='/')
                    if mag > 2.5:
                        update_combined()
                        socketio.emit('combined', combined)
                        socketio.emit('events_update', list(events))
                    continue

                # Jiroskop
                m = re.match(r'Gx:\s*([-\d.]+)\s+Gy:\s*([-\d.]+)\s+Gz:\s*([-\d.]+)', line)
                if m:
                    gx, gy, gz = float(m[1]), float(m[2]), float(m[3])
                    with lock:
                        sensor_state['gx'] = gx
                        sensor_state['gy'] = gy
                        sensor_state['gz'] = gz
                        sensor_state['gx_hist'].append(gx)
                        sensor_state['gy_hist'].append(gy)
                        sensor_state['gz_hist'].append(gz)
                        payload = {
                            'gx': gx, 'gy': gy, 'gz': gz,
                            'gx_hist': list(sensor_state['gx_hist']),
                            'gy_hist': list(sensor_state['gy_hist']),
                            'gz_hist': list(sensor_state['gz_hist']),
                        }
                    socketio.emit('gyro', payload, namespace='/')
                    continue

                # Kalp atışı + SpO2
                m = re.match(r'Kalp Atisi:\s*([\d.]+)\s+bpm.*SpO2:\s*(\d+)', line)
                if m:
                    hr, spo2 = float(m[1]), int(m[2])
                    with lock:
                        sensor_state['hr'] = hr
                        sensor_state['spo2'] = spo2
                        sensor_state['hr_hist'].append(hr)
                        payload = {
                            'hr': hr, 'spo2': spo2,
                            'hr_hist': list(sensor_state['hr_hist']),
                        }
                    socketio.emit('vitals', payload, namespace='/')
                    if 0 < spo2 < 95:
                        _add_event('spo2', 'MAX30100', spo2)
                        socketio.emit('events_update', list(events))
                    continue

        except Exception as e:
            sensor_state['connected'] = False
            socketio.emit('sensor_status', {'connected': False})
            print(f"Sensör bağlantısı kesildi: {e}, 3s sonra tekrar...")
            time.sleep(3)

# ── SocketIO events ───────────────────────────────────────────────────────────
@socketio.on('recalibrate')
def on_recalibrate(data):
    nid = int(data.get('node', 0))
    with lock:
        if nid in nodes:
            nodes[nid].baseline = None
            nodes[nid].cal_buf  = []
            nodes[nid].status   = 'KALİBRASYON'
    socketio.emit('recal_ack', {'node': nid})

@socketio.on('connect')
def on_connect():
    print(f"[WS] Client bağlandı", flush=True)
    from flask_socketio import emit as ws_emit
    ws_emit('sensor_status', {'connected': sensor_state['connected']})
    ws_emit('cnn_result', cnn_result)
    ws_emit('combined', combined)
    ws_emit('events', list(events))

@socketio.on('get_state')
def on_get_state():
    from flask_socketio import emit as ws_emit
    ws_emit('sensor_status', {'connected': sensor_state['connected']})
    ws_emit('cnn_result', cnn_result)
    ws_emit('combined', combined)
    ws_emit('events', list(events))

@app.route('/debug')
def debug():
    return {
        'sensor_connected': sensor_state['connected'],
        'ax': sensor_state['ax'], 'ay': sensor_state['ay'], 'az': sensor_state['az'],
        'hr': sensor_state['hr'], 'spo2': sensor_state['spo2'],
        'cnn': cnn_result['label'],
    }

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    threading.Thread(target=udp_thread,    daemon=True).start()
    threading.Thread(target=serial_thread, daemon=True).start()
    print("Dashboard: http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
