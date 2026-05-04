#!/usr/bin/env python3
"""
CSI Monitor — ESP32-S3 UDP stream parser + gerçek zamanlı görselleştirme
ADR-018 binary format: magic(4) + node_id(1) + antennas(1) + subcarriers(2) +
                       freq(4) + seq(4) + rssi(1) + noise(1) + pad(2) + IQ data
"""

import socket
import struct
import time
import collections
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

# ── Sabitler ──────────────────────────────────────────────────────────────────
UDP_PORT     = 5005
MAGIC        = 0xC5110001
HEADER_FMT   = '<IBBHIIBB2x'   # 20 byte
HEADER_SIZE  = 20
FS           = 20               # ~20 Hz örnekleme (firmware default)
WINDOW_SEC   = 30               # analiz penceresi
WINDOW_SIZE  = FS * WINDOW_SEC  # 600 frame

# ── Butter bandpass filtre ────────────────────────────────────────────────────
def bandpass(data, lo, hi, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [lo / nyq, hi / nyq], btype='band')
    return filtfilt(b, a, data)

# ── ADR-018 paketi parse et ───────────────────────────────────────────────────
def parse_packet(data: bytes):
    if len(data) < HEADER_SIZE:
        return None
    magic, node_id, n_ant, n_sc, freq, seq, rssi_u8, noise_u8 = struct.unpack(
        HEADER_FMT, data[:HEADER_SIZE]
    )
    if magic != MAGIC:
        return None

    rssi  = rssi_u8  if rssi_u8  < 128 else rssi_u8  - 256
    noise = noise_u8 if noise_u8 < 128 else noise_u8 - 256

    iq = data[HEADER_SIZE:]
    n_pairs = n_ant * n_sc
    if len(iq) < n_pairs * 2:
        return None

    amps   = np.zeros((n_ant, n_sc))
    phases = np.zeros((n_ant, n_sc))
    for a in range(n_ant):
        for s in range(n_sc):
            idx = (a * n_sc + s) * 2
            I = struct.unpack('b', bytes([iq[idx]]))[0]
            Q = struct.unpack('b', bytes([iq[idx+1]]))[0]
            amps[a, s]   = np.sqrt(I*I + Q*Q)
            phases[a, s] = np.arctan2(Q, I)

    return {
        'node_id': node_id,
        'seq':     seq,
        'rssi':    rssi,
        'noise':   noise,
        'freq_mhz': freq,
        'n_ant':   n_ant,
        'n_sc':    n_sc,
        'amp':     amps,
        'phase':   phases,
        'mean_amp': float(np.mean(amps)),
        'ts':      time.time(),
    }

# ── Varlık / hareket tespiti (CUSUM) ─────────────────────────────────────────
class MotionDetector:
    def __init__(self, window=50, thresh_motion=2.0, thresh_presence=0.3):
        self.buf       = collections.deque(maxlen=window)
        self.baseline  = None
        self.t_motion  = thresh_motion
        self.t_presence= thresh_presence

    def update(self, mean_amp):
        self.buf.append(mean_amp)
        if len(self.buf) < 20:
            return "KALIBRASYON"
        if self.baseline is None:
            self.baseline = np.mean(list(self.buf))
        var = float(np.var(list(self.buf)))
        diff = abs(mean_amp - self.baseline)
        if var > self.t_motion:
            return "HAREKET VAR"
        elif diff > self.t_presence:
            return "HAREKETSIZ VARLIK"
        else:
            return "BOŞ ODA"

# ── Vital sign tahmini ────────────────────────────────────────────────────────
def estimate_vitals(amp_series: list, fs=FS):
    if len(amp_series) < FS * 10:
        return None, None
    sig = np.array(amp_series) - np.mean(amp_series)

    # Nefes: 0.1-0.5 Hz
    try:
        breath_sig  = bandpass(sig, 0.1, 0.5, fs)
        freqs       = np.fft.rfftfreq(len(breath_sig), 1/fs)
        psd         = np.abs(np.fft.rfft(breath_sig))**2
        mask        = (freqs >= 0.1) & (freqs <= 0.5)
        peak_freq   = freqs[mask][np.argmax(psd[mask])]
        breath_bpm  = round(peak_freq * 60, 1)
    except Exception:
        breath_bpm = None

    # Kalp atışı: 0.8-2.0 Hz
    try:
        hr_sig   = bandpass(sig, 0.8, 2.0, fs)
        freqs    = np.fft.rfftfreq(len(hr_sig), 1/fs)
        psd      = np.abs(np.fft.rfft(hr_sig))**2
        mask     = (freqs >= 0.8) & (freqs <= 2.0)
        peak_freq= freqs[mask][np.argmax(psd[mask])]
        hr_bpm   = round(peak_freq * 60, 1)
    except Exception:
        hr_bpm = None

    return breath_bpm, hr_bpm

# ── Ana döngü ────────────────────────────────────────────────────────────────
def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', UDP_PORT))
    sock.settimeout(2)

    detector   = MotionDetector()
    amp_window = collections.deque(maxlen=WINDOW_SIZE)

    print(f"{'='*55}")
    print(f"  CSI Monitor — UDP :{UDP_PORT}")
    print(f"  Pencere: {WINDOW_SEC}s | FS: {FS} Hz")
    print(f"{'='*55}")

    pkt_count  = 0
    last_print = time.time()
    last_vital = time.time()

    try:
        while True:
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                print("  [bekleniyor...]")
                continue

            frame = parse_packet(data)
            if frame is None:
                continue

            pkt_count += 1
            amp_window.append(frame['mean_amp'])
            status = detector.update(frame['mean_amp'])

            # Her saniye durum satırı
            now = time.time()
            if now - last_print >= 1.0:
                bar_len  = int(min(frame['mean_amp'] / 2, 30))
                bar      = '█' * bar_len + '░' * (30 - bar_len)
                print(
                    f"  #{frame['seq']:>6} | "
                    f"node:{frame['node_id']} | "
                    f"RSSI:{frame['rssi']:>4}dBm | "
                    f"amp:{frame['mean_amp']:>5.1f} [{bar}] | "
                    f"{status}"
                )
                last_print = now

            # Her 15 saniyede vital tahmin
            if now - last_vital >= 15.0 and len(amp_window) >= FS * 10:
                br, hr = estimate_vitals(list(amp_window))
                print(f"\n  ┌─ VİTAL TAHMİN ({'%d' % len(amp_window)} frame) ─────────────")
                print(f"  │  Nefes hızı : {br if br else '?':>6} nefes/dak")
                print(f"  │  Kalp atışı : {hr if hr else '?':>6} bpm  (deneysel)")
                print(f"  └{'─'*42}\n")
                last_vital = now

    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        print(f"\nToplam {pkt_count} paket işlendi.")

if __name__ == '__main__':
    main()
