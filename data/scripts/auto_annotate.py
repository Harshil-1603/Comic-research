"""
Fully automated annotation pipeline — plan §STEP 4.

For every panel in data/processed/:
  1. Run OCR (pytesseract) to extract dialogue text
  2. Run sentiment model to map text → emotion label
  3. Write data/annotations.csv

Usage:
    python data/scripts/auto_annotate.py
    python data/scripts/auto_annotate.py --input data/processed --output data/annotations.csv
    python data/scripts/auto_annotate.py --force   # overwrite existing annotations
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import cv2
import pandas as pd
from tqdm import tqdm

from features.ocr import extract_text
from features.sentiment import get_emotion
import config


def parse_args():
    p = argparse.ArgumentParser(description="Auto-annotate comic panels via OCR + sentiment")
    p.add_argument("--input",  type=str, default=config.PROCESSED_DIR)
    p.add_argument("--output", type=str, default=config.ANNOTATIONS_CSV)
    p.add_argument("--source", type=str, default="auto_ocr_sentiment")
    p.add_argument("--force",  action="store_true",
                   help="Re-annotate files that already have an entry in the CSV")
    p.add_argument("--skip-debug", action="store_true", default=True,
                   help="Skip _debug.jpg overlay images (default: True)")
    return p.parse_args()


def load_existing(output_path):
    """Return set of already-annotated filenames and their rows."""
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        if len(df) > 0:
            return set(df["image"].tolist()), df.to_dict("records")
    return set(), []


def get_panels(img_dir, skip_debug):
    files = sorted([
        f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")
    ])
    if skip_debug:
        files = [f for f in files if "_debug" not in f]
    return files


def main():
    args = parse_args()

    if not os.path.isdir(args.input):
        print(f"✘  Input directory not found: {args.input}")
        sys.exit(1)

    panels = get_panels(args.input, args.skip_debug)
    if not panels:
        print(f"✘  No panel images found in {args.input}")
        print("   Run:  python data/scripts/run_extraction.py  first.")
        sys.exit(1)

    done_set, rows = load_existing(args.output)
    already = len(done_set)
    to_process = [f for f in panels if args.force or f not in done_set]

    print(f"\nAuto-Annotation Pipeline")
    print(f"  Input   : {args.input}  ({len(panels)} panels)")
    print(f"  Output  : {args.output}")
    print(f"  Already : {already} annotated")
    print(f"  To do   : {len(to_process)}\n")

    if not to_process:
        print("✔  All panels already annotated.")
        return

    errors = 0
    for img_name in tqdm(to_process, desc="Annotating", unit="panel"):
        path = os.path.join(args.input, img_name)
        img = cv2.imread(path)

        if img is None:
            tqdm.write(f"  ⚠  Skipping unreadable image: {img_name}")
            errors += 1
            continue

        try:
            text    = extract_text(img)
            emotion = get_emotion(text)
        except Exception as e:
            tqdm.write(f"  ⚠  Error on {img_name}: {e}")
            text    = ""
            emotion = "neutral"
            errors += 1

        rows.append({
            "image":   img_name,
            "emotion": emotion,
            "text":    text,
            "source":  args.source,
        })

    # Save final CSV
    df = pd.DataFrame(rows, columns=["image", "emotion", "text", "source"])
    df.to_csv(args.output, index=False)

    print(f"\n✔  Saved {len(df)} annotations → {args.output}")
    if errors:
        print(f"  ⚠  {errors} panels had errors (labelled neutral).")

    print(f"\nEmotion distribution:")
    for emotion, count in df["emotion"].value_counts().items():
        bar = "█" * int(count / max(len(df), 1) * 40)
        print(f"  {emotion:10s}: {count:4d}  {bar}")


if __name__ == "__main__":
    main()
