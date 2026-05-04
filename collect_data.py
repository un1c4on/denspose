#!/usr/bin/env python3
"""
CSI Veri Toplama — Tek sınıf, tek çalıştırma

Kullanım:
  python3 collect_data.py --label empty   --duration 180
  python3 collect_data.py --label present --duration 180
  python3 collect_data.py --label walking --duration 180
  python3 collect_data.py --label fall    --count 40

Parametreler:
  --label     : empty | present | walking | fall
  --duration  : saniye cinsinden kayıt süresi (sürekli sınıflar)
  --count     : kaydedilecek örnek sayısı (fall için önerilir)
  --countdown : kayıt öncesi geri sayım süresi (varsayılan 5s)
"""

import argparse, socket, struct, time, json, os, math
import collections, sys, signal
import numpy as np

# ── Sabitler ──────────────────────────────────────────────────────────────────
UDP_PORT      = 5005
MAGIC         = 0xC5110001
HEADER_FMT    = '<IBBHIIBB2x'
HEADER_SIZE   = 20
FS            = 20
WINDOW_SEC    = 3
WINDOW_FRAMES = FS * WINDOW_SEC   # 60 frame = 1 örnek
N_SC          = 64
SAVE_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
VALID_LABELS  = ['empty', 'present', 'walking', 'fall']

os.makedirs(SAVE_DIR, exist_ok=True)

# ── Parser ────────────────────────────────────────────────────────────────────
def parse_packet(data: bytes):
    if len(data) < HEADER_SIZE:
        return None
    magic, node_id, n_ant, n_sc, freq, seq, rssi_u8, _ = struct.unpack(
        HEADER_FMT, data[:HEADER_SIZE])
    if magic != MAGIC:
        return None
    iq   = data[HEADER_SIZE:]
    amps = []
    for i in range(0, min(len(iq) - 1, n_ant * n_sc * 2), 2):
        I = struct.unpack('b', bytes([iq[i]]))[0]
        Q = struct.unpack('b', bytes([iq[i+1]]))[0]
        amps.append(math.sqrt(I*I + Q*Q))
    if len(amps) < N_SC:
        amps += [0.0] * (N_SC - len(amps))
    return {'node': node_id, 'seq': seq, 'amps': amps[:N_SC]}

# ── Yardımcı: renkli print ────────────────────────────────────────────────────
def pr(msg): print(msg, flush=True)
def green(s):  return f'\033[92m{s}\033[0m'
def yellow(s): return f'\033[93m{s}\033[0m'
def red(s):    return f'\033[91m{s}\033[0m'
def cyan(s):   return f'\033[96m{s}\033[0m'
def bold(s):   return f'\033[1m{s}\033[0m'

# ── Geri sayım ────────────────────────────────────────────────────────────────
def countdown(seconds, message):
    pr(f'\n{yellow(message)}')
    for i in range(seconds, 0, -1):
        print(f'\r  {i} saniye...   ', end='', flush=True)
        time.sleep(1)
    print('\r  ' + green('BAŞLADI!') + '           ')

# ── Node bağlantısını doğrula ─────────────────────────────────────────────────
def wait_for_nodes(sock, min_nodes=2, timeout=15):
    pr(f'\n{cyan("Node bağlantısı bekleniyor...")} (en az {min_nodes} node)')
    seen = set()
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            data, _ = sock.recvfrom(8192)
        except socket.timeout:
            continue
        f = parse_packet(data)
        if f:
            seen.add(f['node'])
            print(f'\r  Bağlı node\'lar: {sorted(seen)}   ', end='', flush=True)
            if len(seen) >= min_nodes:
                pr(f'\n{green(f"{len(seen)} node bağlandı ✓")}')
                return True
    pr(f'\n{red("Yeterli node bulunamadı! Firmware ve WiFi bağlantısını kontrol edin.")}')
    return False

# ── Ana kayıt döngüsü ─────────────────────────────────────────────────────────
def record(sock, label, duration=None, target_count=None, min_nodes=2):
    """
    duration    → N saniye boyunca sürekli kayıt (empty, present, walking)
    target_count → N örnek toplanınca dur (fall)
    """
    buffers   = {}   # node_id → deque
    saved     = 0
    last_save = 0
    start     = time.time()
    fname     = os.path.join(SAVE_DIR, f'{label}.jsonl')

    deadline = (start + duration) if duration else None

    pr(f'\n{bold("Kayıt başladı:")} {cyan(label)}')
    if duration:
        pr(f'  Süre: {duration} saniye')
    else:
        pr(f'  Hedef: {target_count} örnek')
    pr(f'  Dosya: {fname}')
    pr(f'  Durdurmak için: Ctrl+C\n')

    with open(fname, 'a') as out:
        while True:
            # Süre veya sayı kontrolü
            if deadline and time.time() > deadline:
                break
            if target_count and saved >= target_count:
                break

            try:
                data, _ = sock.recvfrom(8192)
            except socket.timeout:
                continue

            f = parse_packet(data)
            if f is None:
                continue

            nid = f['node']
            if nid not in buffers:
                buffers[nid] = collections.deque(maxlen=WINDOW_FRAMES)
            buffers[nid].append(f['amps'])

            # Pencere dolu node'ları bul
            ready = [n for n, b in buffers.items() if len(b) == WINDOW_FRAMES]
            if len(ready) < min_nodes:
                continue

            # 0.5s aralıkla kaydet (örnekler birbiriyle çok örtüşmesin)
            now = time.time()
            if now - last_save < 0.5:
                continue
            last_save = now

            # Örneği yaz
            sample = {
                'ts':    now,
                'label': label,
                'nodes': {str(n): list(buffers[n]) for n in ready},
            }
            out.write(json.dumps(sample) + '\n')
            out.flush()
            saved += 1

            # İlerleme
            if duration:
                elapsed  = now - start
                remain   = max(0, deadline - now)
                pct      = int(elapsed / duration * 30)
                bar      = '█' * pct + '░' * (30 - pct)
                print(f'\r  [{bar}] {remain:.0f}s kaldı  |  '
                      f'{green(str(saved))} örnek kaydedildi   ',
                      end='', flush=True)
            else:
                pct  = int(saved / target_count * 30)
                bar  = '█' * pct + '░' * (30 - pct)
                print(f'\r  [{bar}] {green(str(saved))}/{target_count} örnek   ',
                      end='', flush=True)

    print()
    return saved

# ── Dataset özeti ─────────────────────────────────────────────────────────────
def print_summary():
    pr(f'\n{bold("── Dataset Durumu ──────────────────────────")}')
    total = 0
    for lbl in VALID_LABELS:
        fname = os.path.join(SAVE_DIR, f'{lbl}.jsonl')
        count = 0
        if os.path.exists(fname):
            with open(fname) as f:
                count = sum(1 for _ in f)
        total += count
        status = green('✓ Yeterli') if count >= 30 else yellow('⚠ Az') if count > 0 else red('✗ Boş')
        bar    = '█' * min(count, 40) + '░' * max(0, 40 - count)
        pr(f'  {lbl:<12} {bar} {count:>4}  {status}')
    pr(f'  {"TOPLAM":<12} {"─"*40} {total:>4}')
    pr(f'\n  Veriler: {SAVE_DIR}/')
    if total > 0:
        pr(f'\n  Eğitmek için: {cyan("python3 train.py")}')

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='CSI veri toplama aracı')
    ap.add_argument('--label',     required=True, choices=VALID_LABELS,
                    help='Sınıf etiketi')
    ap.add_argument('--duration',  type=int, default=None,
                    help='Kayıt süresi (saniye) — empty/present/walking için')
    ap.add_argument('--count',     type=int, default=None,
                    help='Hedef örnek sayısı — fall için önerilir')
    ap.add_argument('--countdown', type=int, default=5,
                    help='Başlamadan önceki geri sayım (varsayılan: 5s)')
    ap.add_argument('--min-nodes', type=int, default=2,
                    help='Minimum bağlı node sayısı (varsayılan: 2)')
    args = ap.parse_args()

    # Parametre kontrolü
    if args.duration is None and args.count is None:
        if args.label == 'fall':
            args.count = 40
        else:
            args.duration = 180
        pr(f'{yellow("Süre/sayı belirtilmedi, varsayılan kullanılıyor:")} '
           f'{args.count or args.duration}')

    # Socket aç
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', UDP_PORT))
    sock.settimeout(1.0)

    # Ctrl+C ile temiz çıkış
    def handle_sigint(sig, frame):
        pr(f'\n\n{yellow("Kayıt kullanıcı tarafından durduruldu.")}')
        print_summary()
        sock.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

    pr(bold('\n╔══════════════════════════════════════╗'))
    pr(bold( '║      CSI VERİ TOPLAMA ARACI          ║'))
    pr(bold( '╚══════════════════════════════════════╝'))
    pr(f'  Etiket  : {cyan(args.label)}')
    pr(f'  Süre    : {args.duration}s' if args.duration else
       f'  Hedef   : {args.count} örnek')

    # Senaryo talimatı
    tips = {
        'empty':   '🚪 Odadan çıkın veya hiç hareket etmeyin.',
        'present': '🪑 Bir yere oturun, hareketsiz kalın.',
        'walking': '🚶 Odada rahatça yürüyün, dönün, durun, tekrar yürüyün.',
        'fall':    '⬇️  Yavaşça yere oturun/uzanın, kalkın. Tekrarlayın.',
    }
    pr(f'\n  {bold("Senaryo:")} {tips[args.label]}')

    # Node bağlantısı
    if not wait_for_nodes(sock, min_nodes=args.min_nodes):
        sock.close()
        sys.exit(1)

    # Geri sayım
    countdown(args.countdown,
              f'Hazır olun! {args.countdown} saniye sonra kayıt başlıyor...')

    # Kayıt
    saved = record(sock, args.label,
                   duration=args.duration,
                   target_count=args.count,
                   min_nodes=args.min_nodes)

    pr(f'\n{green("✓ Tamamlandı!")} {saved} örnek kaydedildi.')
    print_summary()
    sock.close()

if __name__ == '__main__':
    main()
