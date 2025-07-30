# thing

import re

import re

def parse_chronyc_sources(output):
    lines = output.strip().splitlines()
    data = []
    
    for line in lines:
        # Skip header lines
        if not line or line.startswith('=') or line.startswith('MS'):
            continue
        
        # Match everything before the offset block (last 3 numbers)
        parts = re.split(r'\s{2,}', line.strip())
        if len(parts) < 6:
            continue  # Skip malformed lines

        try:
            entry = {
                'mode': parts[0][0],        # ^ or =
                'state': parts[0][1],       # *, +, -, ?
                'name': parts[1],
                'stratum': int(parts[2]),
                'poll': int(parts[3]),
                'reach': int(parts[4], 8),  # octal -> decimal
                'last_rx': int(parts[5]),
                'offset': None,
                'raw_offset': None,
                'jitter': None
            }

            # Match offset patterns (e.g., -14us[+1247us] +/- 1234us or +0.123s[+0.223s] +/- 0.789 ms)
            offset_match = re.search(r'([+-]?\d+\.?\d*)([a-z]+)?\s*\[\s*([+-]?\d+\.?\d*)([a-z]+)?\]\s*\+/-\s*([+-]?\d+\.?\d*)([a-z]+)?', line)

            if offset_match:
                offset_val = float(offset_match.group(1))
                raw_offset_val = float(offset_match.group(3))
                jitter_val = float(offset_match.group(5))

                # Optionally, convert microseconds to seconds if unit is 'us'
                if offset_match.group(2) == 'us':
                    offset_val /= 1_000_000
                if offset_match.group(4) == 'us':
                    raw_offset_val /= 1_000_000
                if offset_match.group(6) == 'us':
                    jitter_val /= 1_000_000

                entry['offset'] = offset_val
                entry['raw_offset'] = raw_offset_val
                entry['jitter'] = jitter_val

            data.append(entry)
        except Exception as e:
            print(f"Error parsing line: {line}\n{e}")

    return data


def parse_chronyc_clients(output):
    lines = output.strip().splitlines()
    data = []
    i = 0
    # Skip header lines (typically 3)
    while i < len(lines):
        if lines[i].startswith('==='):
            i += 1
            break
        i += 1

    while i < len(lines) - 1:
        ntp_line = lines[i].strip()
        cmd_line = lines[i + 1].strip()

        ntp_parts = ntp_line.split()
        cmd_parts = cmd_line.split()

        if len(ntp_parts) >= 6 and len(cmd_parts) >= 5:
            entry = {
                "ip": ntp_parts[0],
                "ntp_score": int(ntp_parts[1]),
                "ntp_drop": int(ntp_parts[2]),
                "ntp_int": int(ntp_parts[3]),
                "ntp_intl": int(ntp_parts[4]),
                "ntp_last_rx": int(ntp_parts[5]),
                "cmd_score": int(cmd_parts[1]),
                "cmd_drop": int(cmd_parts[2]),
                "cmd_int": cmd_parts[3],
                "cmd_last_rx": int(cmd_parts[4]) if cmd_parts[4].isdigit() else None,
            }
            data.append(entry)
        i += 2

    return data
