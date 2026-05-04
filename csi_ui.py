#!/usr/bin/env python3
"""
CSI Live UI — UDP :5005 paketlerini terminalde canlı gösterir
Çıkmak için: q veya Ctrl+C
"""

import socket, struct, time, curses, collections, threading
import numpy as np

UDP_PORT    = 5005
MAGIC       = 0xC5110001
HEADER_FMT  = '<IBBHIIBB2x'
HEADER_SIZE = 20

# ── Parser ────────────────────────────────────────────────────────────────────
def parse(data: bytes):
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
        amps.append((I*I + Q*Q)**0.5)

    return {
        'node':    node_id,
        'seq':     seq,
        'rssi':    rssi,
        'noise':   noise,
        'freq':    freq,
        'n_ant':   n_ant,
        'n_sc':    n_sc,
        'size':    len(data),
        'amps':    amps,
        'mean':    float(np.mean(amps)) if amps else 0,
        'ts':      time.time(),
        'raw_hex': data[:32].hex(),
    }

# ── Paylaşılan durum ──────────────────────────────────────────────────────────
state = {
    'packets':    collections.deque(maxlen=200),   # son 200 paket logu
    'nodes':      {},                               # node_id → son frame
    'total':      0,
    'running':    True,
    'rate':       0.0,
    'count_buf':  collections.deque(maxlen=10),
}

def udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', UDP_PORT))
    sock.settimeout(0.5)
    last_ts = time.time()
    count   = 0
    while state['running']:
        try:
            data, addr = sock.recvfrom(8192)
        except socket.timeout:
            continue
        f = parse(data)
        if f is None:
            ts = time.strftime('%H:%M:%S')
            state['packets'].append(
                f"[{ts}] {addr[0]}:{addr[1]}  {len(data)}B  YANLIŞ MAGIC: {data[:4].hex()}")
            continue
        f['addr'] = addr[0]
        state['total'] += 1
        count += 1
        state['nodes'][f['node']] = f

        # paket logu satırı
        ts   = time.strftime('%H:%M:%S')
        bar  = '█' * min(int(f['mean'] / 3), 20)
        line = (f"[{ts}] node={f['node']}  seq={f['seq']:>7}  "
                f"rssi={f['rssi']:>4}dBm  sc={f['n_sc']:>3}  "
                f"amp={f['mean']:>5.1f} {bar}")
        state['packets'].append(line)

        # saniyede paket hızı
        now = time.time()
        if now - last_ts >= 1.0:
            state['rate'] = count / (now - last_ts)
            count  = 0
            last_ts = now
    sock.close()

# ── Curses UI ─────────────────────────────────────────────────────────────────
def draw(stdscr):
    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_CYAN,    curses.COLOR_BLACK)  # başlık
    curses.init_pair(2, curses.COLOR_GREEN,   curses.COLOR_BLACK)  # iyi değer
    curses.init_pair(3, curses.COLOR_YELLOW,  curses.COLOR_BLACK)  # uyarı
    curses.init_pair(4, curses.COLOR_RED,     curses.COLOR_BLACK)  # kötü
    curses.init_pair(5, curses.COLOR_WHITE,   curses.COLOR_BLACK)  # normal
    curses.init_pair(6, curses.COLOR_BLACK,   curses.COLOR_CYAN)   # header bg

    stdscr.nodelay(True)

    while state['running']:
        stdscr.erase()
        H, W = stdscr.getmaxyx()

        # ── Header ──────────────────────────────────────────────────────────
        header = f"  CSI Monitor  │  UDP :{UDP_PORT}  │  Toplam: {state['total']:>6}  │  {state['rate']:>5.1f} pkt/s  │  [q] çıkış"
        header = header[:W-1].ljust(W-1)
        stdscr.addstr(0, 0, header, curses.color_pair(6) | curses.A_BOLD)

        # ── Node kutuları ────────────────────────────────────────────────────
        row = 2
        nodes = sorted(state['nodes'].items())
        box_w = max(38, (W - 4) // max(len(nodes), 1))

        for idx, (nid, f) in enumerate(nodes):
            col = idx * (box_w + 2) + 1
            age = time.time() - f['ts']
            age_color = curses.color_pair(2) if age < 2 else curses.color_pair(3)

            # Kutu başlığı
            stdscr.addstr(row,   col, f"┌{'─'*(box_w-2)}┐", curses.color_pair(1))
            title = f" NODE {nid}  {f['addr']} "
            stdscr.addstr(row+1, col, f"│{title.center(box_w-2)}│", curses.color_pair(1) | curses.A_BOLD)
            stdscr.addstr(row+2, col, f"├{'─'*(box_w-2)}┤", curses.color_pair(1))

            # Değerler
            def row_add(r, label, val, color=5):
                line = f"│ {label:<14} {str(val):<{box_w-18}} │"
                stdscr.addstr(row+r, col, line[:box_w], curses.color_pair(color))

            rssi_color = 2 if f['rssi'] > -60 else (3 if f['rssi'] > -75 else 4)
            row_add(3, "Seq",        f"{f['seq']:,}")
            row_add(4, "RSSI",       f"{f['rssi']} dBm",   rssi_color)
            row_add(5, "Noise",      f"{f['noise']} dBm")
            row_add(6, "Frekans",    f"{f['freq']} MHz")
            row_add(7, "Anten",      f"{f['n_ant']}")
            row_add(8, "Subcarrier", f"{f['n_sc']}")
            row_add(9, "Boyut",      f"{f['size']} byte")
            row_add(10,"Ort. Amp",   f"{f['mean']:.2f}")

            # Amplitüd bar
            bar_max  = box_w - 6
            bar_fill = min(int(f['mean'] / 2), bar_max)
            bar      = '█' * bar_fill + '░' * (bar_max - bar_fill)
            amp_color = 2 if f['mean'] > 10 else (3 if f['mean'] > 4 else 4)
            stdscr.addstr(row+11, col, f"│ [{bar}] │"[:box_w], curses.color_pair(amp_color))

            # Son güncelleme
            row_add(12, "Son güncell.", f"{age:.1f}s önce", age_color)

            # Raw hex
            hex_line = f"│ {f['raw_hex'][:box_w-4]}"
            stdscr.addstr(row+13, col, (hex_line + ' '*(box_w-len(hex_line)-1) + '│')[:box_w])
            stdscr.addstr(row+14, col, f"└{'─'*(box_w-2)}┘", curses.color_pair(1))

        # ── Paket logu ───────────────────────────────────────────────────────
        log_start = row + 17
        log_lines = H - log_start - 2
        if log_lines > 0:
            stdscr.addstr(log_start - 1, 1, "─── Paket Logu " + "─"*(W-18), curses.color_pair(1))
            pkts = list(state['packets'])
            for i, line in enumerate(pkts[-log_lines:]):
                if log_start + i >= H - 1:
                    break
                node_color = 2 if ' node=1 ' in line else (3 if ' node=2 ' in line else 5)
                stdscr.addstr(log_start + i, 1, line[:W-2], curses.color_pair(node_color))

        # ── Footer ───────────────────────────────────────────────────────────
        if H > 2:
            footer = f" Node sayısı: {len(state['nodes'])}  │  Bağlı IP'ler: {', '.join(f['addr'] for f in state['nodes'].values())}"
            stdscr.addstr(H-1, 0, footer[:W-1], curses.color_pair(3))

        stdscr.refresh()

        # Klavye
        key = stdscr.getch()
        if key in (ord('q'), ord('Q'), 27):
            state['running'] = False
            break

        time.sleep(0.1)

def main():
    t = threading.Thread(target=udp_listener, daemon=True)
    t.start()
    try:
        curses.wrapper(draw)
    except KeyboardInterrupt:
        pass
    finally:
        state['running'] = False
        t.join(timeout=2)
        print(f"Toplam {state['total']} paket alındı.")

if __name__ == '__main__':
    main()
