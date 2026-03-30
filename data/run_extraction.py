import os
from panel_extractor import extract_panels

inp, out = "data/raw/", "data/processed/"
os.makedirs(out, exist_ok=True)

for f in os.listdir(inp):
    if f.endswith(".jpg"):
        extract_panels(os.path.join(inp, f), out)
