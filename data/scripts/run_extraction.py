"""
Batch-run panel extraction over all JPGs in data/raw/.

Usage:
    python data/scripts/run_extraction.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data.scripts.panel_extractor import extract_panels
import config

inp = config.RAW_DIR
out = config.PROCESSED_DIR
os.makedirs(out, exist_ok=True)

total = 0
for f in sorted(os.listdir(inp)):
    if f.lower().endswith(".jpg"):
        n = extract_panels(os.path.join(inp, f), out)
        print(f"{f}: {n} panels extracted")
        total += n

print(f"\nTotal panels extracted: {total}")
