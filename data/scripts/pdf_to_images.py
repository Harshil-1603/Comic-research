"""
Convert source PDFs to per-page JPEG images.

Usage:
    python data/scripts/pdf_to_images.py
    python data/scripts/pdf_to_images.py --pdf data/source/comic.pdf --out data/raw/
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def pdf_to_images(pdf_path, out_dir=config.RAW_DIR, dpi=300):
    from pdf2image import convert_from_path
    os.makedirs(out_dir, exist_ok=True)
    pages = convert_from_path(pdf_path, dpi=dpi)
    for i, p in enumerate(pages):
        out_name = f"{os.path.basename(pdf_path)}_p{i:04d}.jpg"
        p.save(os.path.join(out_dir, out_name), "JPEG")
    print(f"Converted {len(pages)} pages from {pdf_path} → {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PDF → JPEG pages")
    p.add_argument("--pdf", type=str, default=None,
                   help="PDF path (default: first PDF in data/source/)")
    p.add_argument("--out", type=str, default=config.RAW_DIR)
    args = p.parse_args()

    pdf = args.pdf
    if pdf is None:
        import glob
        pdfs = sorted(glob.glob(os.path.join(config.SOURCE_DIR, "*.pdf")))
        if not pdfs:
            print(f"No PDFs found in {config.SOURCE_DIR}")
            sys.exit(1)
        pdf = pdfs[0]

    pdf_to_images(pdf, args.out)
