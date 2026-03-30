from pdf2image import convert_from_path
import os


def pdf_to_images(pdf_path, out_dir, dpi=300):
    os.makedirs(out_dir, exist_ok=True)
    pages = convert_from_path(pdf_path, dpi=dpi)
    for i, p in enumerate(pages):
        p.save(os.path.join(out_dir, f"{os.path.basename(pdf_path)}_p{i:04d}.jpg"), "JPEG")


if __name__ == "__main__":
    pdf_to_images("data/source/comic1.pdf", "data/raw/")
