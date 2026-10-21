"""Conservative proof for simple clipped TrueType text; no rendering or OCR."""

import math
from io import BytesIO
from copy import copy
from dataclasses import dataclass, field

from pypdf import Transformation

VISIBILITY_VERSION = "native-visibility-v1"


def numbers(values):
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("nonfinite geometry")
    return result


def bounds(matrix, rect):
    a, b, c, d, _, _ = matrix.ctm
    if not ((b == 0 and c == 0) or (a == 0 and d == 0)):
        raise ValueError("unsupported transform")
    l, bottom, r, top = rect
    points = [matrix.apply_on(p) for p in ((l, bottom), (l, top), (r, bottom), (r, top))]
    return numbers((min(p[0] for p in points), min(p[1] for p in points),
                    max(p[0] for p in points), max(p[1] for p in points)))


def contains(outer, inner):
    return outer[0] <= inner[0] <= inner[2] <= outer[2] and outer[1] <= inner[1] <= inner[3] <= outer[3]


@dataclass
class State:
    clip: tuple
    matrix: Transformation = field(default_factory=Transformation)
    text_matrix: Transformation = field(default_factory=Transformation)
    font: object = None
    size: float = 0
    alpha: float = 1
    cursor: float = 0
    char_space: float = 0
    word_space: float = 0


def glyph_bounds(font, code, width, cache):
    # Only named simple encodings have an unambiguous Unicode-to-glyph mapping.
    codec = {"/MacRomanEncoding": "mac_roman", "/WinAnsiEncoding": "cp1252"}.get(font.get("/Encoding"))
    if codec is None:
        raise ValueError("unknown encoding")
    key = id(font)
    if key not in cache:
        from fontTools.ttLib import TTFont
        data = font["/FontDescriptor"].get_object()["/FontFile2"].get_object().get_data()
        if len(data) > 8_000_000:
            raise ValueError("font limit")
        with TTFont(BytesIO(data), lazy=False) as tt:
            units = tt["head"].unitsPerEm
            if tt["maxp"].numGlyphs > 4096 or units <= 0:
                raise ValueError("font complexity limit")
            cmap = tt.getBestCmap()
            if cmap is None and codec == "mac_roman":
                tables = [table for table in tt["cmap"].tables if (table.platformID, table.platEncID) == (1, 0)]
                if len(tables) != 1:
                    raise ValueError("ambiguous cmap")
                cmap = {ord(bytes([key]).decode(codec)): value for key, value in tables[0].cmap.items()}
            metrics = {}
            for char, name in cmap.items():
                glyph = tt["glyf"][name]
                if glyph.isComposite():
                    continue
                box = None if glyph.numberOfContours == 0 else tuple(
                    getattr(glyph, attr) * 1000 / units for attr in ("xMin", "yMin", "xMax", "yMax")
                )
                metrics[char] = (tt["hmtx"][name][0] * 1000 / units, box)
            cache[key] = metrics
    advance, box = cache[key][ord(bytes([code]).decode(codec))]
    if abs(advance - width) > 1:
        raise ValueError("font advance mismatch")
    return box


def proves_native_visibility(page, crop):
    """Return false on every unsupported construct, never infer unknown as empty."""
    try:
        return _prove(page, crop)
    except Exception:
        return False


def _prove(page, crop):
    if crop is None or page.get("/Annots"):
        return False
    resources = page["/Resources"].get_object()
    state = State(crop)
    stack = []
    path = []
    rectangle = None
    pending_clip = False
    text_seen = False
    in_text = False
    marked_depth = 0
    font_cache = {}
    glyph_count = 0
    operations = page.get_contents().operations
    if len(operations) > 100_000:
        return False
    for operands, op in operations:
        if op == b"q":
            if in_text or len(stack) >= 128:
                return False
            stack.append(copy(state))
        elif op == b"Q":
            if in_text:
                return False
            state = stack.pop()
        elif op == b"cm":
            state.matrix = Transformation(numbers(operands)).transform(state.matrix)
        elif op == b"gs":
            gs = resources["/ExtGState"][operands[0]].get_object()
            if set(gs) - {"/Type", "/ca", "/CA", "/BM"} or gs.get("/BM", "/Normal") != "/Normal":
                return False
            state.alpha = float(gs.get("/ca", state.alpha))
            if not 0.1 <= state.alpha <= 1 or not 0.1 <= float(gs.get("/CA", 1)) <= 1:
                return False
        elif op == b"re":
            if rectangle is not None or path:
                return False
            x, y, w, h = numbers(operands)
            if w <= 0 or h <= 0:
                return False
            rectangle = bounds(state.matrix, (x, y, x + w, y + h))
        elif op in (b"m", b"l"):
            if rectangle is not None or (op == b"m" and path):
                return False
            path.append(state.matrix.apply_on(numbers(operands)))
        elif op == b"h":
            if len(path) != 4:
                return False
            xs, ys = {p[0] for p in path}, {p[1] for p in path}
            if len(xs) != 2 or len(ys) != 2 or len(set(path)) != 4:
                return False
            if any(a[0] != b[0] and a[1] != b[1] for a, b in zip(path, path[1:] + path[:1])):
                return False
            rectangle = numbers((min(xs), min(ys), max(xs), max(ys)))
            path = []
        elif op in (b"W", b"W*"):
            if rectangle is None or pending_clip:
                return False
            pending_clip = True
        elif op in (b"n", b"f", b"f*", b"F"):
            if op != b"n" and (text_seen or rectangle is None or not contains(rectangle, crop) or state.alpha != 1):
                return False
            if pending_clip:
                l, b, r, t = state.clip
                x, y, right, top = rectangle
                state.clip = (max(l, x), max(b, y), min(r, right), min(t, top))
                if state.clip[0] >= state.clip[2] or state.clip[1] >= state.clip[3]:
                    return False
            path, rectangle, pending_clip = [], None, False
        elif op == b"BT":
            if in_text:
                return False
            in_text = True
            state.text_matrix, state.cursor = Transformation(), 0
        elif op == b"ET":
            if not in_text:
                return False
            in_text = False
        elif op == b"Tf":
            state.font = resources["/Font"][operands[0]].get_object()
            state.size = float(operands[1])
        elif op == b"Tm":
            state.text_matrix, state.cursor = Transformation(numbers(operands)), 0
        elif op == b"Tr":
            if numbers(operands) != (0.0,):
                return False
        elif op in (b"Tc", b"Tw"):
            value, = numbers(operands)
            if op == b"Tc":
                state.char_space = value
            else:
                state.word_space = value
        elif op in (b"Tj", b"TJ"):
            if not in_text or state.font is None or state.font.get("/Subtype") != "/TrueType" or not state.size > 0:
                return False
            font = state.font
            bbox = numbers(font["/FontDescriptor"].get_object()["/FontBBox"])
            if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                return False
            widths, first = font["/Widths"], int(font["/FirstChar"])
            transform = state.text_matrix.transform(state.matrix)
            for item in (operands[0] if op == b"TJ" else operands):
                if isinstance(item, (int, float)):
                    state.cursor -= float(item) / 1000 * state.size
                    continue
                data = item if isinstance(item, bytes) else item.original_bytes
                glyph_count += len(data)
                if glyph_count > 100_000:
                    return False
                for code in data:
                    width = float(widths[code - first]) if 0 <= code - first < len(widths) else math.nan
                    if not math.isfinite(width) or width < 0:
                        return False
                    l, b, r, t = (value / 1000 * state.size for value in bbox)
                    glyph = bounds(transform, (state.cursor + l, b, state.cursor + r, t))
                    if not contains(state.clip, glyph):
                        exact = glyph_bounds(font, code, width, font_cache)
                        if exact is not None:
                            l, b, r, t = (value / 1000 * state.size for value in exact)
                            glyph = bounds(transform, (state.cursor + l, b, state.cursor + r, t))
                            if not contains(state.clip, glyph):
                                return False
                    state.cursor += width / 1000 * state.size + state.char_space + (state.word_space if code == 32 else 0)
                    text_seen = True
        elif op in (b"BDC", b"BMC"):
            if operands[0] not in ("/P", "/Span", "/Artifact"):
                return False
            if op == b"BDC" and (not isinstance(operands[1], dict) or set(operands[1]) - {"/MCID", "/ActualText"}):
                return False
            marked_depth += 1
        elif op == b"EMC":
            marked_depth -= 1
            if marked_depth < 0:
                return False
        elif op in (b"cs", b"CS"):
            color = operands[0]
            if color not in ("/DeviceRGB", "/DeviceGray", "/DeviceCMYK"):
                color = resources["/ColorSpace"][color].get_object()
                if not isinstance(color, list) or color[0] != "/ICCBased" or color[1].get_object().get("/N") not in (1, 3, 4):
                    return False
        elif op not in (b"sc", b"SC", b"g", b"G", b"rg", b"RG", b"k", b"K"):
            return False
    return text_seen and not (stack or in_text or marked_depth or path or rectangle or pending_clip)
