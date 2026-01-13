#!/usr/bin/env python3
# filter_open_meteo.py
import subprocess
import sys
import json

def is_valid_json(line):
    """Check if a line is valid JSON"""
    line = line.strip()
    if not line:
        return False
    try:
        json.loads(line)
        return True
    except json.JSONDecodeError:
        return False

# Run the open-meteo server
proc = subprocess.Popen(
    ["npx", "open-meteo-mcp-server"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=sys.stderr,
    text=True,
    bufsize=1  # Line buffered
)

# Forward stdin in a separate thread
import threading
def forward_stdin():
    try:
        for line in sys.stdin:
            proc.stdin.write(line)
            proc.stdin.flush()
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        try:
            proc.stdin.close()
        except:
            pass

stdin_thread = threading.Thread(target=forward_stdin, daemon=True)
stdin_thread.start()

# Filter stdout - only pass valid JSON lines
try:
    for line in proc.stdout:
        if line.strip() and is_valid_json(line):
            sys.stdout.write(line)
            sys.stdout.flush()
except (BrokenPipeError, KeyboardInterrupt):
    pass
finally:
    try:
        proc.wait()
    except:
        pass
    sys.exit(proc.returncode if proc.returncode else 0)