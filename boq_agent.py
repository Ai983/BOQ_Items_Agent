"""BOQ Quantity Extraction Agent CLI.

This tool parses PDF/image plans, detects rooms/walls/partitions with
OpenCV + OCR + YOLOv8, and exports BOQ quantities with audit trails.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]


@dataclass
class TextAnnotation:
    text: str
    center: Point
    bbox: Tuple[int, int, int, int]


@dataclass
class Room:
    name: str
    polygon: List[Point]
    area_sqft: float
    perimeter_ft: float
    audit: str


@dataclass
class WallSegment:
    start: Point
    end: Point
    kind: str


@dataclass
class DetectedObject:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]


@dataclass
class QuantityItem:
    item_type: str
    material_type: str
    quantity: float
    unit: str
    audit: str


SCALE_PATTERN = re.compile(r"scale\s*[:=]?\s*(\d+(?:\.\d+)?)\s*[:/]\s*(\d+(?:\.\d+)?)", re.I)
DIMENSION_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:'|ft)\s*(\d+(?:\.\d+)?)?\s*(?:\"|in)?",
    re.I,
)


def optional_import(module_name: str):
    if importlib.util.find_spec(module_name) is None:
        return None
    return importlib.import_module(module_name)


def shoelace_area(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def polygon_perimeter(points: Sequence[Point]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(math.dist(p1, p2) for p1, p2 in zip(points, points[1:] + points[:1]))


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

def detect_scale_from_texts(texts: Iterable[str]) -> Optional[float]:
    for text in texts:
        match = SCALE_PATTERN.search(text)
        if match:
            numerator = float(match.group(1))
            denominator = float(match.group(2))
            if denominator == 0:
                continue
            # Architectural notation like "Scale 1:100" means
            # 1 unit on paper = 100 units in reality.
            # We use the larger value as a pixels-per-foot proxy so that
            # higher denominators result in larger scale factors.
            return max(numerator, denominator)
    return None


def parse_dimension_text(text: str) -> Optional[float]:
    match = DIMENSION_PATTERN.search(text)
    if not match:
        return None
    feet = float(match.group(1))
    inches = float(match.group(2) or 0.0)
    return feet + inches / 12.0


def ensure_dependency(module, name: str) -> None:
    if module is None:
        raise RuntimeError(f"Missing dependency: {name}. Please install it and retry.")


def load_images_from_input(input_path: Path, dpi: int) -> List[Tuple[str, "numpy.ndarray"]]:
    pdf2image = optional_import("pdf2image")
    cv2 = optional_import("cv2")
    ensure_dependency(cv2, "opencv-python")

    if input_path.suffix.lower() in {".pdf"}:
        ensure_dependency(pdf2image, "pdf2image")
        images = pdf2image.convert_from_path(str(input_path), dpi=dpi)
        return [(f"page_{idx+1}", cv2.cvtColor(image, cv2.COLOR_RGB2BGR)) for idx, image in enumerate(images)]
    image = cv2.imread(str(input_path))
    if image is None:
        raise RuntimeError(f"Unable to read image: {input_path}")
    return [(input_path.stem, image)]


class RoomDetector:
    def __init__(self, ocr_lang: str = "eng") -> None:
        self.cv2 = optional_import("cv2")
        self.pytesseract = optional_import("pytesseract")
        ensure_dependency(self.cv2, "opencv-python")
        ensure_dependency(self.pytesseract, "pytesseract")
        self.ocr_lang = ocr_lang

    def extract_text(self, image) -> List[TextAnnotation]:
        rgb = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
        data = self.pytesseract.image_to_data(rgb, lang=self.ocr_lang, output_type=self.pytesseract.Output.DICT)
        annotations: List[TextAnnotation] = []
        for text, x, y, w, h in zip(
            data["text"],
            data["left"],
            data["top"],
            data["width"],
            data["height"],
        ):
            cleaned = text.strip()
            if not cleaned:
                continue
            center = (x + w / 2.0, y + h / 2.0)
            annotations.append(TextAnnotation(text=cleaned, center=center, bbox=(x, y, w, h)))
        return annotations

    def detect_room_polygons(self, image) -> List[List[Point]]:
        gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
        blurred = self.cv2.GaussianBlur(gray, (5, 5), 0)
        edges = self.cv2.Canny(blurred, 50, 150)
        contours, _ = self.cv2.findContours(edges, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)
        polygons: List[List[Point]] = []
        for contour in contours:
            if self.cv2.contourArea(contour) < 5000:
                continue
            approx = self.cv2.approxPolyDP(contour, 0.02 * self.cv2.arcLength(contour, True), True)
            polygon = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
            if len(polygon) >= 3:
                polygons.append(polygon)
        return polygons

    def detect_wall_segments(self, image) -> List[WallSegment]:
        gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
        edges = self.cv2.Canny(gray, 50, 150)
        lines = self.cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=100, minLineLength=40, maxLineGap=10)
        segments: List[WallSegment] = []
        if lines is None:
            return segments
        for x1, y1, x2, y2 in lines[:, 0]:
            segments.append(WallSegment(start=(float(x1), float(y1)), end=(float(x2), float(y2)), kind="wall"))
        return segments

    def detect_rooms(self, image, scale: float) -> Tuple[List[Room], List[WallSegment], List[TextAnnotation]]:
        text_annotations = self.extract_text(image)
        polygons = self.detect_room_polygons(image)
        rooms: List[Room] = []
        for idx, polygon in enumerate(polygons, start=1):
            area_px = shoelace_area(polygon)
            perimeter_px = polygon_perimeter(polygon)
            name = f"Room-{idx}"
            for annotation in text_annotations:
                if point_in_polygon(annotation.center, polygon):
                    name = annotation.text
                    break
            area_sqft = area_px / (scale**2)
            perimeter_ft = perimeter_px / scale
            audit = f"Contour area {area_px:.2f}px^2 / scale {scale:.2f}px/ft"
            rooms.append(
                Room(
                    name=name,
                    polygon=polygon,
                    area_sqft=area_sqft,
                    perimeter_ft=perimeter_ft,
                    audit=audit,
                )
            )
        wall_segments = self.detect_wall_segments(image)
        return rooms, wall_segments, text_annotations


class ObjectDetector:
    def __init__(self, model_path: Optional[str]) -> None:
        self.ultralytics = optional_import("ultralytics")
        self.model = None
        if model_path:
            ensure_dependency(self.ultralytics, "ultralytics")
            self.model = self.ultralytics.YOLO(model_path)

    def detect(self, image) -> List[DetectedObject]:
        if self.model is None:
            return []
        results = self.model.predict(source=image, verbose=False)
        detections: List[DetectedObject] = []
        for result in results:
            for box in result.boxes:
                label = result.names.get(int(box.cls[0]), "unknown")
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = [float(coord) for coord in box.xyxy[0]]
                detections.append(
                    DetectedObject(label=label, confidence=confidence, bbox=(int(x1), int(y1), int(x2), int(y2)))
                )
        return detections


class QuantityExtractor:
    def __init__(
        self,
        wall_height: float,
        partition_height: float,
        partition_labels: Sequence[str],
        wall_labels: Sequence[str],
    ) -> None:
        self.wall_height = wall_height
        self.partition_height = partition_height
        self.partition_labels = {label.lower() for label in partition_labels}
        self.wall_labels = {label.lower() for label in wall_labels}

    def compute_wall_finish(self, segments: Sequence[WallSegment], scale: float) -> float:
        total_length_px = sum(math.dist(seg.start, seg.end) for seg in segments)
        total_length_ft = total_length_px / scale
        return total_length_ft * self.wall_height

    def compute_partition_area(self, detections: Sequence[DetectedObject], scale: float) -> float:
        total_length_px = 0.0
        for detection in detections:
            if detection.label.lower() in self.partition_labels:
                x1, y1, x2, y2 = detection.bbox
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                # Treat the longer bbox side as the partition run length
                run_length_px = max(width, height)
                total_length_px += run_length_px
        length_ft = total_length_px / scale
        return length_ft * self.partition_height

    def compute_wall_segments_from_detections(self, detections: Sequence[DetectedObject]) -> List[WallSegment]:
        segments: List[WallSegment] = []
        for detection in detections:
            if detection.label.lower() in self.wall_labels:
                x1, y1, x2, y2 = detection.bbox
                segments.append(WallSegment(start=(x1, y1), end=(x2, y2), kind=detection.label))
        return segments


class BOQWriter:
    def __init__(self) -> None:
        self.openpyxl = optional_import("openpyxl")

    def export_quantities(self, items: Sequence[QuantityItem], output_path: Path) -> None:
        if output_path.suffix.lower() == ".xlsx":
            ensure_dependency(self.openpyxl, "openpyxl")
            workbook = self.openpyxl.Workbook()
            worksheet = workbook.active
            worksheet.append(["Item Type", "Material", "Quantity", "Unit", "Audit"])
            for item in items:
                worksheet.append([item.item_type, item.material_type, round(item.quantity, 2), item.unit, item.audit])
            workbook.save(output_path)
        else:
            with output_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["item_type", "material_type", "quantity", "unit", "audit"])
                writer.writeheader()
                writer.writerows(asdict(item) for item in items)

    def export_debug_csv(self, output_path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def build_quantities(
    rooms: Sequence[Room],
    detections: Sequence[DetectedObject],
    wall_segments: Sequence[WallSegment],
    scale: float,
    floor_material: str,
    ceiling_material: str,
    partition_material: str,
    wall_finish_material: str,
    extractor: QuantityExtractor,
) -> List[QuantityItem]:
    items: List[QuantityItem] = []
    for room in rooms:
        items.append(
            QuantityItem(
                item_type="Flooring",
                material_type=floor_material,
                quantity=room.area_sqft,
                unit="sq.ft",
                audit=f"{room.name}: {room.area_sqft:.2f} sq.ft from {room.audit}",
            )
        )
        items.append(
            QuantityItem(
                item_type="Ceiling",
                material_type=ceiling_material,
                quantity=room.area_sqft,
                unit="sq.ft",
                audit=f"{room.name}: {room.area_sqft:.2f} sq.ft from {room.audit}",
            )
        )

    detection_segments = extractor.compute_wall_segments_from_detections(detections)
    all_wall_segments = list(wall_segments) + detection_segments
    if all_wall_segments:
        wall_finish_area = extractor.compute_wall_finish(all_wall_segments, scale)
        items.append(
            QuantityItem(
                item_type="Wall Finish",
                material_type=wall_finish_material,
                quantity=wall_finish_area,
                unit="sq.ft",
                audit=f"Wall finish from {len(all_wall_segments)} segments @ scale {scale:.2f}px/ft",
            )
        )

    partition_area = extractor.compute_partition_area(detections, scale)
    if partition_area:
        items.append(
            QuantityItem(
                item_type="Partition",
                material_type=partition_material,
                quantity=partition_area,
                unit="sq.ft",
                audit=f"Partition area from YOLO detections @ scale {scale:.2f}px/ft",
            )
        )

    return items


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
                audit=item.audit,
            )
        grouped[key].quantity += item.quantity
    return list(grouped.values())


def ensure_output_path(output_path: Path) -> Path:
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path.is_absolute():
        if output_dir not in output_path.parents:
            return output_dir / output_path.name
        return output_path
    return output_dir / output_path


def determine_scale(texts: Sequence[TextAnnotation], scale_override: Optional[float]) -> float:
    if scale_override is not None:
        return scale_override
    scale = detect_scale_from_texts(annotation.text for annotation in texts)
    if scale:
        return scale
    for annotation in texts:
        if parse_dimension_text(annotation.text):
            return 1.0
    return 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description="BOQ quantity extraction for PDF/image layouts.")
    parser.add_argument("--input", type=Path, required=True, help="Input PDF/image file")
    parser.add_argument("--scale", type=float, default=None, help="Pixels per foot scale (override autodetect)")
    parser.add_argument("--default-floor-material", type=str, default="Tile")
    parser.add_argument("--default-ceiling-material", type=str, default="POP")
    parser.add_argument("--partition-material", type=str, default="Gypsum")
    parser.add_argument("--wall-finish-material", type=str, default="Paint")
    parser.add_argument("--partition-height", type=float, default=9.0)
    parser.add_argument("--wall-height", type=float, default=9.0)
    parser.add_argument("--output", type=Path, default=Path("boq_output.xlsx"))
    parser.add_argument("--debug-csv", action="store_true", help="Write debug CSVs for rooms and walls")
    parser.add_argument("--dpi", type=int, default=300, help="PDF render DPI")
    parser.add_argument("--ocr-lang", type=str, default="eng", help="Tesseract language code")
    parser.add_argument("--yolo-model", type=str, default=None, help="Path to YOLOv8 model weights")
    parser.add_argument("--partition-labels", type=str, default="partition", help="Comma-separated YOLO labels")
    parser.add_argument("--wall-labels", type=str, default="wall", help="Comma-separated YOLO labels")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        return 1

    images = load_images_from_input(args.input, args.dpi)
    room_detector = RoomDetector(ocr_lang=args.ocr_lang)
    object_detector = ObjectDetector(model_path=args.yolo_model)
    extractor = QuantityExtractor(
        wall_height=args.wall_height,
        partition_height=args.partition_height,
        partition_labels=[label.strip() for label in args.partition_labels.split(",") if label.strip()],
        wall_labels=[label.strip() for label in args.wall_labels.split(",") if label.strip()],
    )
    writer = BOQWriter()

    all_rooms: List[Room] = []
    all_wall_segments: List[WallSegment] = []
    all_detections: List[DetectedObject] = []
    text_annotations: List[TextAnnotation] = []

    for _, image in images:
        detections = object_detector.detect(image)
        rooms, wall_segments, annotations = room_detector.detect_rooms(image, scale=1.0)
        all_rooms.extend(rooms)
        all_wall_segments.extend(wall_segments)
        all_detections.extend(detections)
        text_annotations.extend(annotations)

    scale = determine_scale(text_annotations, args.scale)
    scaled_rooms: List[Room] = []
    for room in all_rooms:
        area_sqft = room.area_sqft / (scale**2)
        perimeter_ft = room.perimeter_ft / scale
        scaled_rooms.append(
            Room(
                name=room.name,
                polygon=room.polygon,
                area_sqft=area_sqft,
                perimeter_ft=perimeter_ft,
                audit=f"Scaled from {room.audit} with scale {scale:.2f}px/ft",
            )
        )

    quantities = build_quantities(
        rooms=scaled_rooms,
        detections=all_detections,
        wall_segments=all_wall_segments,
        scale=scale,
        floor_material=args.default_floor_material,
        ceiling_material=args.default_ceiling_material,
        partition_material=args.partition_material,
        wall_finish_material=args.wall_finish_material,
        extractor=extractor,
    )
    grouped = group_quantities(quantities)
    output_path = ensure_output_path(args.output)
    writer.export_quantities(grouped, output_path)

    if args.debug_csv:
        rooms_debug = [
            {
                "name": room.name,
                "area_sqft": round(room.area_sqft, 2),
                "perimeter_ft": round(room.perimeter_ft, 2),
                "polygon": json.dumps(room.polygon),
                "audit": room.audit,
            }
            for room in scaled_rooms
        ]
        detection_wall_segments = extractor.compute_wall_segments_from_detections(all_detections)
        all_debug_segments = list(all_wall_segments) + detection_wall_segments
        walls_debug = [
            {
                "start": json.dumps(segment.start),
                "end": json.dumps(segment.end),
                "kind": segment.kind,
            }
            for segment in all_debug_segments
        ]
        if rooms_debug:
            writer.export_debug_csv(output_path.with_name("rooms_debug.csv"), rooms_debug, rooms_debug[0].keys())
        if walls_debug:
            writer.export_debug_csv(output_path.with_name("walls_debug.csv"), walls_debug, walls_debug[0].keys())

    summary = {
        "images_processed": len(images),
        "rooms_detected": len(scaled_rooms),
        "detections": len(all_detections),
        "scale": scale,
        "output": str(output_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
