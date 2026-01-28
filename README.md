# BOQ Quantity Extraction Agent (CLI)

This repository includes a lightweight CLI that follows the BOQ extraction workflow for DWG/DXF layout plans.
It detects closed room outlines, prompts for material assignments, and exports grouped quantities with
traceable audit notes.

## Requirements

- Python 3.10+
- Optional dependencies for DWG/DXF parsing and Excel export:
  - `ezdxf`
  - `openpyxl`

> **Note**: This environment does not ship with these libraries. Install them in your own environment
> with network access before running the tool.

## Usage

```bash
python boq_agent.py sample_plans/sample_layout.dwg \
  --scale 1 \
  --default-floor-material Tile \
  --default-ceiling-material POP \
  --partition-height 9 \
  --wall-height 9 \
  --output boq_output.xlsx
```

If the scale cannot be auto-detected, the CLI will prompt for a drawing-units-per-foot scale factor.
You can also pass `--scale` directly.

## Output

The CLI produces a grouped BOQ table with columns:

- Item Type
- Material Type
- Quantity
- Unit
- Audit

The `audit` column contains the traceable source (room name, layer, and scale) used for each quantity.

## Notes on DWG parsing

The sample plan is saved as AutoCAD 2018 DWG (`AC1032`). This format requires a DWG reader.
Install `ezdxf` or convert the file to DXF before running.
