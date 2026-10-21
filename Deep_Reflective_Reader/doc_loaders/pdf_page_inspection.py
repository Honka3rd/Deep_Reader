"""Cheap native PDF evidence, without rendering, image decoding or OCR decisions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import groupby
import math
import unicodedata

from pypdf import PageObject, Transformation
from .pdf_native_visibility import proves_native_visibility


Rect = tuple[float, float, float, float]
MAX_IMAGE_RECTS = 512
MAX_TEXT_ORIGINS = 2048


@dataclass(frozen=True)
class PdfNativeTextQuality:
    character_count: int
    non_whitespace_count: int
    printable_ratio: float
    replacement_ratio: float
    control_ratio: float
    longest_repeat_ratio: float
    duplicate_line_ratio: float

    @classmethod
    def from_text(cls, text: str) -> PdfNativeTextQuality:
        size = len(text)
        visible = "".join(char for char in text if not char.isspace())
        # Per-glyph PDF extraction is not evidence of a duplicated text layer.
        lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 1]
        counts = Counter(lines)
        return cls(
            character_count=size,
            non_whitespace_count=len(visible),
            printable_ratio=sum(c.isprintable() or c in "\n\r\t" for c in text) / (size or 1),
            replacement_ratio=text.count("\ufffd") / (size or 1),
            control_ratio=sum(
                unicodedata.category(c) in {"Cc", "Cs"} and c not in "\n\r\t"
                for c in text
            ) / (size or 1),
            longest_repeat_ratio=max(
                (sum(1 for _ in group) for _, group in groupby(visible)), default=0,
            ) / (len(visible) or 1),
            duplicate_line_ratio=sum(count - 1 for count in counts.values()) / (len(lines) or 1),
        )

    def to_dict(self) -> dict[str, int | float]:
        return dict(vars(self))


@dataclass(frozen=True)
class PdfPageInspection:
    source_sha256: str
    page_index: int
    native_text: str | None = field(repr=False)
    text_quality: PdfNativeTextQuality | None
    crop_box: Rect | None
    rotation_degrees: int | None
    has_font_resources: bool | None
    image_draw_count: int | None
    image_rects: tuple[Rect, ...]
    image_coverage_ratio: float | None
    text_origins: tuple[tuple[float, float], ...]
    text_positions_complete: bool
    has_vector_paint: bool | None
    flags: tuple[str, ...]
    has_only_page_background: bool = False
    evidence_schema_version: int = 1

    def to_metadata(self) -> dict[str, object]:
        """Exclude source text; geometry is internal evidence, not task-layout data."""
        return {
            "source_sha256": self.source_sha256,
            "page_index": self.page_index,
            "text_quality": self.text_quality.to_dict() if self.text_quality else None,
            "crop_box": self.crop_box,
            "rotation_degrees": self.rotation_degrees,
            "has_font_resources": self.has_font_resources,
            "has_only_page_background": self.has_only_page_background,
            "image_draw_count": self.image_draw_count,
            "image_rects": list(self.image_rects),
            "image_coverage_ratio": self.image_coverage_ratio,
            "text_origins": list(self.text_origins),
            "text_positions_complete": self.text_positions_complete,
            "has_vector_paint": self.has_vector_paint,
            "flags": list(self.flags),
            "evidence_schema_version": self.evidence_schema_version,
        }


def _rect(values) -> Rect:
    result = tuple(float(value) for value in values)
    if len(result) != 4 or not all(math.isfinite(value) for value in result):
        raise ValueError("invalid rectangle")
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("empty rectangle")
    return result


def _matrix(values) -> Transformation:
    result = tuple(float(value) for value in values)
    if len(result) != 6 or not all(math.isfinite(value) for value in result):
        raise ValueError("invalid matrix")
    return Transformation(result)


def _image_rect(matrix: Transformation, crop: Rect) -> Rect | None:
    a, b, c, d, _, _ = matrix.ctm
    if not ((b == 0 and c == 0) or (a == 0 and d == 0)):
        raise ValueError("non-axis-aligned image")
    points = [matrix.apply_on(point) for point in ((0, 0), (1, 0), (0, 1), (1, 1))]
    if not all(math.isfinite(value) for point in points for value in point):
        raise ValueError("invalid image coordinates")
    left = max(crop[0], min(point[0] for point in points))
    bottom = max(crop[1], min(point[1] for point in points))
    right = min(crop[2], max(point[0] for point in points))
    top = min(crop[3], max(point[1] for point in points))
    return (left, bottom, right, top) if right > left and top > bottom else None


def _union_area(rects: list[Rect]) -> float:
    # Sweep vertical strips so overlapping or repeatedly drawn images count once.
    xs = sorted({x for rect in rects for x in (rect[0], rect[2])})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((b, t) for l, b, r, t in rects if l < right and r > left)
        end = -math.inf
        height = 0.0
        for bottom, top in intervals:
            height += max(0.0, top - max(bottom, end))
            end = max(end, top)
        area += (right - left) * height
    return area


def inspect_pdf_page(page: PageObject, *, source_sha256: str, page_index: int) -> PdfPageInspection:
    """Extract text once; unavailable evidence is None, never a false zero."""
    flags: set[str] = set()
    try:
        crop = _rect(page.cropbox)
    except Exception:
        crop = None
        flags.add("page_geometry_unavailable")
    try:
        raw_rotation = float(page.get("/Rotate", 0))
        if not math.isfinite(raw_rotation) or raw_rotation % 90:
            raise ValueError("invalid rotation")
        rotation = int(raw_rotation) % 360
    except Exception:
        rotation = None
        flags.add("page_rotation_unavailable")

    try:
        resources = page.get("/Resources")
        resources = resources.get_object() if resources is not None else {}
        has_fonts = bool(resources.get("/Font"))
    except Exception:
        resources = {}
        has_fonts = None
        flags.add("resources_unavailable")

    rects: list[Rect] = []
    image_count = 0
    inventory_complete = "resources_unavailable" not in flags
    geometry_complete = crop is not None and inventory_complete
    vector_paint = False
    background_only = True
    path_rect = None
    path_started = False
    text_operators_present = False
    try:
        matrix = Transformation()
        stack: list[Transformation] = []
        content = page.get_contents()
        for operands, operator in content.operations if content is not None else ():
            if operator == b"q":
                stack.append(matrix)
            elif operator == b"Q":
                matrix = stack.pop()
            elif operator == b"cm":
                matrix = _matrix(operands).transform(matrix)
            elif operator in {b"W", b"W*"}:
                geometry_complete = False
                flags.add("custom_clipping_unresolved")
            elif operator == b"Tr" and int(operands[0]) in {3, 7}:
                flags.add("invisible_text_mode")
            elif operator in {b"Tj", b"TJ", b"'", b'"'}:
                text_operators_present = True
            elif operator == b"re":
                x, y, width, height = (float(value) for value in operands)
                rect_matrix = Transformation().scale(width, height).translate(x, y).transform(matrix)
                try:
                    path_rect = _image_rect(rect_matrix, crop) if crop is not None and not path_started else None
                except ValueError:
                    path_rect = None
                path_started = True
            elif operator in {b"m", b"l", b"c", b"v", b"y", b"h"}:
                path_rect = None
                path_started = True
            elif operator == b"n":
                path_rect = None
                path_started = False
            elif operator in {b"S", b"s", b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*", b"sh"}:
                vector_paint = True
                background_only = background_only and (
                    operator in {b"f", b"F", b"f*"} and not text_operators_present
                    and path_rect is not None and crop is not None
                    and all(math.isclose(a, b) for a, b in zip(path_rect, crop))
                )
                path_rect = None
                path_started = False
            elif operator in {b"Do", b"INLINE IMAGE"}:
                if operator == b"Do":
                    obj = resources["/XObject"][operands[0]].get_object()
                    if obj.get("/Subtype") != "/Image":
                        # Form callbacks can be local to the form, not page coordinates.
                        flags.add("form_or_unknown_xobject_uninspected")
                        geometry_complete = inventory_complete = False
                        continue
                    if obj.get("/SMask") or obj.get("/Mask") or obj.get("/OC"):
                        flags.add("image_visibility_unresolved")
                else:
                    flags.add("inline_image_present")
                image_count += 1
                if image_count > MAX_IMAGE_RECTS:
                    geometry_complete = False
                    flags.add("image_geometry_limit")
                    continue
                if crop is not None:
                    try:
                        rect = _image_rect(matrix, crop)
                        if rect is not None:
                            rects.append(rect)
                    except ValueError:
                        geometry_complete = False
                        flags.add("image_transform_unresolved")
            elif operator == b"gs":
                try:
                    state = resources["/ExtGState"][operands[0]].get_object()
                    opaque_default = (
                        set(state) <= {"/Type", "/ca", "/CA", "/BM"}
                        and float(state.get("/ca", 1)) == 1
                        and float(state.get("/CA", 1)) == 1
                        and state.get("/BM", "/Normal") == "/Normal"
                    )
                except Exception:
                    opaque_default = False
                if not opaque_default:
                    flags.add("graphics_state_uninspected")
        if stack:
            raise ValueError("unbalanced graphics stack")
    except Exception:
        geometry_complete = inventory_complete = False
        flags.add("content_stream_inspection_failed")

    try:
        if page.get("/Annots"):
            flags.add("annotations_uninspected")
    except Exception:
        flags.add("annotations_unavailable")
    if not inventory_complete and not has_fonts:
        has_fonts = None

    origins: list[tuple[float, float]] = []
    positions_complete = inventory_complete and crop is not None

    def visit_text(text, cm, tm, font, size):
        nonlocal positions_complete
        if not text.strip():
            return
        if len(origins) >= MAX_TEXT_ORIGINS:
            positions_complete = False
            flags.add("text_position_limit")
            return
        try:
            point = _matrix(tm).transform(_matrix(cm)).apply_on((0, 0))
            if not all(math.isfinite(value) for value in point) or crop is None:
                raise ValueError("invalid text origin")
            if not (crop[0] <= point[0] <= crop[2] and crop[1] <= point[1] <= crop[3]):
                raise ValueError("text origin outside page")
            origins.append(tuple(point))
            if font is None:
                flags.add("text_font_mapping_unknown")
        except Exception:
            positions_complete = False
            flags.add("text_position_unavailable")

    try:
        native_text = page.extract_text(visitor_text=visit_text)
        if not isinstance(native_text, str):
            raise ValueError("native text unavailable")
    except Exception:
        native_text = None
        flags.add("native_text_extraction_failed")
        positions_complete = False
    if native_text and native_text.strip() and not origins:
        positions_complete = False
        flags.add("text_position_unavailable")
    if text_operators_present and native_text is not None and not native_text.strip():
        flags.add("text_operators_without_extracted_text")
        positions_complete = False
    if not positions_complete:
        origins = []

    if (image_count == 0 and inventory_complete
            and flags & {"custom_clipping_unresolved", "graphics_state_uninspected"}
            and proves_native_visibility(page, crop)):
        flags.difference_update({"custom_clipping_unresolved", "graphics_state_uninspected"})
        geometry_complete = True
        background_only = True

    coverage = None
    if geometry_complete and crop is not None:
        coverage = min(1.0, _union_area(rects) / ((crop[2] - crop[0]) * (crop[3] - crop[1])))
    return PdfPageInspection(
        source_sha256=source_sha256,
        page_index=page_index,
        native_text=native_text,
        text_quality=PdfNativeTextQuality.from_text(native_text) if native_text is not None else None,
        crop_box=crop,
        rotation_degrees=rotation,
        has_font_resources=has_fonts,
        image_draw_count=image_count if inventory_complete else None,
        image_rects=tuple(rects),
        image_coverage_ratio=coverage,
        text_origins=tuple(origins),
        text_positions_complete=positions_complete,
        has_vector_paint=vector_paint if inventory_complete else None,
        flags=tuple(sorted(flags)),
        has_only_page_background=bool(inventory_complete and vector_paint and background_only),
    )
