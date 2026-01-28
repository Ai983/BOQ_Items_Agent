"""BOQ Quantity Extraction Agent CLI.

This tool parses DWG/DXF plans, detects rooms/walls/partitions, and
exports BOQ quantities with audit trails.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # optional dependency
    import ezdxf  # type: ignore
except ImportError:  # pragma: no cover - optional
    ezdxf = None

try:  # optional dependency
    import openpyxl  # type: ignore
except ImportError:  # pragma: no cover - optional
    openpyxl = None

Point = Tuple[float, float]


@dataclass
class Room:
    name: str
    polygon: List[Point]
    area_sqft: float
    audit: str


@dataclass
class QuantityItem:
    item_type: str
    material_type: str
    quantity: float
    unit: str
    audit: str


SCALE_PATTERN = re.compile(r"scale\s*[:]?\s*(\d+(?:\.\d+)?)\s*[:/]\s*(\d+(?:\.\d+)?)", re.I)
FEET_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:'|ft)")
INCH_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:\"|in)")


class ScaleDetectionError(Exception):
    pass


def shoelace_area(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1):
            inside = not inside
    return inside


def parse_text_value(text: str) -> Optional[float]:
    """Parse dimension text like 10'-0" into feet."""
    text = text.strip()
    feet_match = FEET_PATTERN.search(text)
    inch_match = INCH_PATTERN.search(text)
    if not feet_match and not inch_match:
        return None
    feet = float(feet_match.group(1)) if feet_match else 0.0
    inches = float(inch_match.group(1)) if inch_match else 0.0
    return feet + inches / 12.0


def detect_scale_from_text(texts: Iterable[str]) -> Optional[float]:
    for text in texts:
        match = SCALE_PATTERN.search(text)
        if match:
            numerator = float(match.group(1))
            denominator = float(match.group(2))
            if denominator == 0:
                continue
            return numerator / denominator
    return None


def prompt_for_scale() -> float:
    print("Scale could not be auto-detected.")
    raw = input("Enter scale factor as drawing_units_per_foot (e.g., 1 for feet units): ").strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise ScaleDetectionError("Invalid scale input.") from exc


def prompt_materials(room_names: Sequence[str], item_type: str, default: Optional[str]) -> Dict[str, str]:
    materials: Dict[str, str] = {}
    for name in room_names:
        if default:
            materials[name] = default
            continue
        material = input(f"Assign {item_type} material for room '{name}': ").strip()
        materials[name] = material or "Unspecified"
    return materials


def ensure_ezdxf_available() -> None:
    if ezdxf is None:
        raise RuntimeError(
            "ezdxf is required to read DWG/DXF files. "
            "Install it with 'pip install ezdxf' in an environment with network access."
        )


def extract_text_entities(msp) -> List[Tuple[str, Point]]:
    texts: List[Tuple[str, Point]] = []
    for entity in msp:
        if entity.dxftype() in {"TEXT", "MTEXT"}:
            content = entity.text if entity.dxftype() == "TEXT" else entity.plain_text()
            insert = entity.dxf.insert
            texts.append((content, (float(insert.x), float(insert.y))))
    return texts


def extract_closed_polylines(msp) -> List[Tuple[List[Point], str]]:
    polylines: List[Tuple[List[Point], str]] = []
    for entity in msp:
        if entity.dxftype() == "LWPOLYLINE":
            if not entity.closed:
                continue
            points = [(float(x), float(y)) for x, y, *_ in entity]
            polylines.append((points, entity.dxf.layer))
        elif entity.dxftype() == "POLYLINE":
            if not entity.is_closed:
                continue
            points = [(float(v.dxf.x), float(v.dxf.y)) for v in entity.vertices]
            polylines.append((points, entity.dxf.layer))
    return polylines


def extract_lines_by_layer(msp, layer_keywords: Sequence[str]) -> List[Tuple[Point, Point, str]]:
    lines: List[Tuple[Point, Point, str]] = []
    for entity in msp:
        if entity.dxftype() == "LINE":
            layer = entity.dxf.layer
            if any(keyword.lower() in layer.lower() for keyword in layer_keywords):
                start = entity.dxf.start
                end = entity.dxf.end
                lines.append(((float(start.x), float(start.y)), (float(end.x), float(end.y)), layer))
    return lines


def compute_length(lines: Sequence[Tuple[Point, Point, str]]) -> float:
    return sum(math.dist(start, end) for start, end, _ in lines)


def build_rooms(polylines: Sequence[Tuple[List[Point], str]], texts: Sequence[Tuple[str, Point]], scale: float) -> List[Room]:
    rooms: List[Room] = []
    for idx, (polygon, layer) in enumerate(polylines, start=1):
        area_units = shoelace_area(polygon)
        area_sqft = area_units / (scale ** 2)
        name = f"Room-{idx}"
        for text, pt in texts:
            if point_in_polygon(pt, polygon):
                name = text.strip() or name
                break
        audit = f"Area from closed polyline on layer '{layer}' with scale {scale:.4f}"
        rooms.append(Room(name=name, polygon=polygon, area_sqft=area_sqft, audit=audit))
    return rooms


def group_quantities(items: Sequence[QuantityItem]) -> List[QuantityItem]:
    grouped: Dict[Tuple[str, str, str], QuantityItem] = {}
    for item in items:
        key = (item.item_type, item.material_type, item.unit)
        if key not in grouped:
            grouped[key] = QuantityItem(
                item_type=item.item_type,
                material_type=item.material_type,
                quantity=0.0,
                unit=item.unit,
                audit="",
            )
        grouped[key].quantity += item.quantity
    return list(grouped.values())


def export_quantities(items: Sequence[QuantityItem], output_path: Path) -> None:
    rows = [asdict(item) for item in items]
    if output_path.suffix.lower() == ".xlsx":
        if openpyxl is None:
            raise RuntimeError("openpyxl is required to export .xlsx files.")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Item Type", "Material Type", "Quantity", "Unit", "Audit"])
        for item in items:
            ws.append([item.item_type, item.material_type, round(item.quantity, 2), item.unit, item.audit])
        wb.save(output_path)
    else:
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["item_type", "material_type", "quantity", "unit", "audit"])
            writer.writeheader()
            writer.writerows(rows)


def load_dwg(path: Path):
    ensure_ezdxf_available()
    try:
        return ezdxf.readfile(path)
    except Exception as exc:  # pragma: no cover - depends on ezdxf
        raise RuntimeError(f"Failed to read DWG/DXF: {exc}") from exc


def determine_scale(texts: Sequence[Tuple[str, Point]], dimension_texts: Sequence[str], scale_override: Optional[float]) -> float:
    if scale_override:
        return scale_override
    scale_from_title = detect_scale_from_text([text for text, _ in texts])
    if scale_from_title:
        return scale_from_title
    for text in dimension_texts:
        value = parse_text_value(text)
        if value:
            return 1.0
    return prompt_for_scale()


def main() -> int:
    parser = argparse.ArgumentParser(description="BOQ quantity extraction for DWG/DXF layouts.")
    parser.add_argument("input", type=Path, help="Input DWG/DXF file")
    parser.add_argument("--scale", type=float, help="Drawing units per foot (override autodetect)")
    parser.add_argument("--default-floor-material", type=str, default=None)
    parser.add_argument("--default-ceiling-material", type=str, default=None)
    parser.add_argument("--partition-material", type=str, default="Gypsum")
    parser.add_argument("--wall-finish-material", type=str, default="Paint")
    parser.add_argument("--partition-height", type=float, default=None)
    parser.add_argument("--wall-height", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("boq_output.csv"))
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        return 1

    doc = load_dwg(args.input)
    msp = doc.modelspace()

    texts = extract_text_entities(msp)
    dimension_texts = [text for text, _ in texts if parse_text_value(text)]
    scale = determine_scale(texts, dimension_texts, args.scale)

    polylines = extract_closed_polylines(msp)
    rooms = build_rooms(polylines, texts, scale)
    if not rooms:
        print("No closed polylines found. Unable to detect rooms.")

    floor_materials = prompt_materials([room.name for room in rooms], "flooring", args.default_floor_material)
    ceiling_materials = prompt_materials([room.name for room in rooms], "ceiling", args.default_ceiling_material)

    quantities: List[QuantityItem] = []
    for room in rooms:
        floor_type = floor_materials.get(room.name, "Unspecified")
        ceiling_type = ceiling_materials.get(room.name, "Unspecified")
        quantities.append(
            QuantityItem(
                item_type="Flooring",
                material_type=floor_type,
                quantity=room.area_sqft,
                unit="sq.ft",
                audit=f"{room.name} area {room.area_sqft:.2f} sq.ft from {room.audit}",
            )
        )
        quantities.append(
            QuantityItem(
                item_type="Ceiling",
                material_type=ceiling_type,
                quantity=room.area_sqft,
                unit="sq.ft",
                audit=f"{room.name} area {room.area_sqft:.2f} sq.ft from {room.audit}",
            )
        )

    partition_lines = extract_lines_by_layer(msp, ["partition", "ptn"])
    wall_lines = extract_lines_by_layer(msp, ["wall", "finish"])

    if partition_lines:
        if args.partition_height is None:
            height_raw = input("Enter partition height (ft): ").strip()
            args.partition_height = float(height_raw)
        partition_length = compute_length(partition_lines) / scale
        quantities.append(
            QuantityItem(
                item_type="Partition",
                material_type=args.partition_material,
                quantity=partition_length * args.partition_height,
                unit="sq.ft",
                audit=f"Length {partition_length:.2f} ft x height {args.partition_height:.2f} ft",
            )
        )

    if wall_lines:
        if args.wall_height is None:
            height_raw = input("Enter wall finish height (ft): ").strip()
            args.wall_height = float(height_raw)
        wall_length = compute_length(wall_lines) / scale
        quantities.append(
            QuantityItem(
                item_type="Wall Finish",
                material_type=args.wall_finish_material,
                quantity=wall_length * args.wall_height,
                unit="sq.ft",
                audit=f"Length {wall_length:.2f} ft x height {args.wall_height:.2f} ft",
            )
        )

    grouped = group_quantities(quantities)
    export_quantities(grouped, args.output)
    output_summary = {
        "scale": scale,
        "rooms_detected": len(rooms),
        "output": str(args.output),
    }
    print(json.dumps(output_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
