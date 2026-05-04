#!/usr/bin/env python3
"""
Quick diagnostic: listen on UDP 5005 for 20 seconds and report
which nodes / IPs are sending CSI packets.
"""
import socket, struct, time, math

UDP_PORT    = 5005
MAGIC       = 0xC5110001
HEADER_FMT  = '<IBBHIIBB2x'
HEADER_SIZE = 20
LISTEN_SEC  = 20

def parse_header(data):
    if len(data) < HEADER_SIZE:
        return None
    magic, node_id, n_ant, n_sc, freq, seq, rssi_u8, _ = struct.unpack(
        HEADER_FMT, data[:HEADER_SIZE])
    if magic != MAGIC:
        return None
    return {'node': node_id, 'seq': seq, 'rssi': rssi_u8, 'n_ant': n_ant,
            'n_sc': n_sc, 'size': len(data)}

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('0.0.0.0', UDP_PORT))
sock.settimeout(1.0)

print(f"\n  Listening on UDP :{UDP_PORT} for {LISTEN_SEC}s ...")
print(f"  (Power on both ESP32 nodes and make sure they're on the same WiFi)\n")

stats = {}          # node_id -> {count, first_seq, last_seq, ips, last_time}
bad_packets = 0
start = time.time()

while time.time() - start < LISTEN_SEC:
    try:
        data, addr = sock.recvfrom(8192)
    except socket.timeout:
        elapsed = time.time() - start
        print(f"\r  [{int(elapsed)}/{LISTEN_SEC}s] nodes seen: {list(stats.keys())}  "
              f"packets: {sum(s['count'] for s in stats.values())}  "
              f"bad: {bad_packets}   ", end='', flush=True)
        continue

    h = parse_header(data)
    if h is None:
        bad_packets += 1
        continue

    nid = addr[0]   # IP adresine göre node ayır (firmware hepsi node_id=1)
    if nid not in stats:
        stats[nid] = {'count': 0, 'first_seq': h['seq'], 'last_seq': h['seq'],
                      'ips': set(), 'first_time': time.time(), 'rssi': []}
    s = stats[nid]
    s['count'] += 1
    s['last_seq'] = h['seq']
    s['ips'].add(addr[0])
    s['last_time'] = time.time()
    s['rssi'].append(h['rssi'])

    elapsed = time.time() - start
    print(f"\r  [{int(elapsed)}/{LISTEN_SEC}s] nodes seen: {list(stats.keys())}  "
          f"packets: {sum(s['count'] for s in stats.values())}  "
          f"bad: {bad_packets}   ", end='', flush=True)

sock.close()

# ── Report ────────────────────────────────────────────────────────────────────
print("\n")
print("=" * 60)
print("  DIAGNOSTIC REPORT")
print("=" * 60)

if not stats:
    print("\n  !! NO PACKETS RECEIVED !!")
    print("  Possible causes:")
    print("    - ESP32 nodes not connected to the same WiFi network")
    print("    - ESP32 firmware not running / crashed")
    print("    - Firewall blocking UDP port 5005")
    print("    - Wrong IP — nodes might be sending to a different PC IP")
else:
    for nid, s in sorted(stats.items()):
        duration = s['last_time'] - s['first_time']
        fps = s['count'] / duration if duration > 0 else 0
        avg_rssi = sum(s['rssi']) / len(s['rssi']) if s['rssi'] else 0
        seq_range = s['last_seq'] - s['first_seq']
        lost = max(0, seq_range - s['count'] + 1)
        print(f"\n  Node {nid}:")
        print(f"    Source IP(s) : {', '.join(s['ips'])}")
        print(f"    Packets      : {s['count']}")
        print(f"    Rate         : {fps:.1f} pkt/s")
        print(f"    Seq range    : {s['first_seq']} -> {s['last_seq']}  (lost ~{lost})")
        print(f"    Avg RSSI     : {avg_rssi:.0f}")

    if len(stats) < 2:
        print("\n  !! ONLY 1 NODE DETECTED !!")
        print("  Troubleshooting for the missing node:")
        print("    1. Check power — is the LED blinking on the 2nd ESP32?")
        print("    2. Check WiFi  — is it connected to the same network?")
        print("    3. Check firmware — does it have the CSI sender flashed?")
        print("    4. Check target IP — firmware must send to THIS PC's IP")
        print("    5. Try rebooting the missing ESP32")
    else:
        print(f"\n  All {len(stats)} nodes detected OK!")

print(f"\n  Bad/unknown packets: {bad_packets}")
print("=" * 60 + "\n")
