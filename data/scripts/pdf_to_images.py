"""
Convert source PDFs to per-page JPEG images.

Processes one page at a time to avoid OOM kills on large comic volumes.

Usage:
    python data/scripts/pdf_to_images.py
    python data/scripts/pdf_to_images.py --pdf data/source/comic.pdf --out data/raw/
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def pdf_to_images(pdf_path, out_dir=config.RAW_DIR, dpi=150):
    """
    Convert every page in pdf_path to a JPEG in out_dir.

    Processes ONE PAGE AT A TIME to avoid OOM on large volumes.
    DPI=150 keeps file sizes manageable while retaining readability.
    """
    from pdf2image import convert_from_path
    import pdf2image.exceptions as pdf2image_exc

    os.makedirs(out_dir, exist_ok=True)

    # Get page count first (cheap call)
    try:
        info = convert_from_path(pdf_path, dpi=72,
                                  first_page=1, last_page=1)
        # Use pdfinfo to get total pages
        import subprocess
        result = subprocess.run(
            ["pdfinfo", pdf_path],
            capture_output=True, text=True
        )
        n_pages = 0
        for line in result.stdout.splitlines():
            if line.startswith("Pages:"):
                n_pages = int(line.split(":")[1].strip())
                break
        if n_pages == 0:
            n_pages = 999   # fallback: process until failure
    except Exception:
        n_pages = 999

    print(f"Converting '{os.path.basename(pdf_path)}' "
          f"({n_pages} pages, DPI={dpi}) → {out_dir}")

    saved = 0
    for page_num in range(1, n_pages + 1):
        out_name = (f"{os.path.basename(pdf_path)}_p{page_num - 1:04d}.jpg")
        out_path = os.path.join(out_dir, out_name)

        # Skip already-converted pages (resumable)
        if os.path.exists(out_path):
            saved += 1
            continue

        try:
            pages = convert_from_path(
                pdf_path, dpi=dpi,
                first_page=page_num, last_page=page_num,
            )
            if not pages:
                break
            pages[0].save(out_path, "JPEG")
            saved += 1
            print(f"  page {page_num}/{n_pages}", end="\r", flush=True)
        except Exception as e:
            print(f"\n  ⚠  Skipping page {page_num}: {e}")
            continue

    print(f"\n  ✔  Saved {saved} pages from '{os.path.basename(pdf_path)}'")
    return saved


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PDF → JPEG pages")
    p.add_argument("--pdf", type=str, default=None,
                   help="PDF path (default: all PDFs in data/source/)")
    p.add_argument("--out", type=str, default=config.RAW_DIR)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    if args.pdf:
        pdfs = [args.pdf]
    else:
        import glob
        pdfs = sorted(glob.glob(os.path.join(config.SOURCE_DIR, "*.pdf")))
        if not pdfs:
            print(f"No PDFs found in {config.SOURCE_DIR}")
            sys.exit(1)

    for pdf in pdfs:
        pdf_to_images(pdf, args.out, dpi=args.dpi)
