# Save this as convert_pdfs_to_images.py
from pdf2image import convert_from_path
import os
from pathlib import Path

input_dir = Path("sample_plans")
output_dir = Path("data/images/train")
output_dir.mkdir(parents=True, exist_ok=True)

for pdf_file in input_dir.glob("*.pdf"):
    images = convert_from_path(str(pdf_file), dpi=300)
    for i, image in enumerate(images):
        out_path = output_dir / f"{pdf_file.stem}_page_{i+1}.png"
        image.save(out_path)
