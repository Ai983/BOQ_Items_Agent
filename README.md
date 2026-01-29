## BOQ Quantity Extraction Agent (CLI)

CLI tool for extracting BOQ quantities from architectural layout PDFs or image plans.
It uses OpenCV for contours, Tesseract OCR for room labels/scale text, and YOLOv8 for walls/partitions, then exports grouped quantities with audit trails.

### Requirements

- **Python**: 3.10+
- **Core libraries** (install in your own environment):
  - `opencv-python`
  - `pytesseract`
  - `pdf2image`
  - `openpyxl`
  - `ultralytics` (and its PyTorch dependencies)

Example installation:

```bash
pip install opencv-python pytesseract pdf2image openpyxl ultralytics
```

> Tesseract itself must be installed on your OS (and available on PATH) for OCR to work.

### Basic usage

Process a PDF plan and export `boq_output.xlsx`:

```bash
python3 boq_agent.py --input sample_plans/<test_plan>.pdf \
  --default-floor-material Tile \
  --default-ceiling-material POP \
  --wall-finish-material Paint \
  --partition-material Gypsum \
  --wall-height 9 \
  --partition-height 9 \
  --yolo-model models/boq_yolov8.pt \
  --output boq_output.xlsx \
  --debug-csv
```

Key options:

- `--input`: PDF or raster image plan.
- `--dpi`: Render DPI for PDFs (default 300).
- `--scale`: Override pixels-per-foot scale. If omitted, the agent tries to infer it from OCR:
  - `Scale 1:100` style notes.
  - Dimension text like `10'-0"`.
- `--yolo-model`: Path to YOLOv8 `.pt` weights (walls/partitions).
- `--partition-labels`, `--wall-labels`: Comma-separated YOLO class names to treat as partitions or walls.
- `--debug-csv`: Also write `rooms_debug.csv` and `walls_debug.csv` in the same output folder.

### Output

The main BOQ table (`boq_output.xlsx` or CSV) has:

- **Item Type**
- **Material**
- **Quantity**
- **Unit**
- **Audit**

Rooms are detected from contours; areas and perimeters are scaled using pixels-per-foot.  
Wall finish and partition quantities are computed from wall segments (Hough + YOLO) and YOLO partition runs respectively.

When `--debug-csv` is enabled:

- `rooms_debug.csv`: One row per detected room with polygon vertices and audit string.
- `walls_debug.csv`: One row per wall/partition segment with start/end coordinates and kind.
