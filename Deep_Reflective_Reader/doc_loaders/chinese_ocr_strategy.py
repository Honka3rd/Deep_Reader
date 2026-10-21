"""Local Chinese vertical-column recognition, separate from generic PDF orchestration."""

from dataclasses import dataclass
import csv
import hashlib
import io
import logging
import os
from pathlib import Path
import re
import tempfile

import numpy as np
from PIL import Image

from .document_load_errors import RawTextOcrFailedError
from .pdf_extraction_contract import PdfExtractionError, PdfExtractionFailureReason

logger = logging.getLogger(__name__)
POLICY_VERSION = "chinese-columns-v2"
MODEL = "chi_tra_vert"
MODEL_SHA256 = "1df02a4b210e5c217b783819538b63e9dfe6904e2b5e53b62664f1b9f7a989d0"
SIMPLIFIED_SHA256 = "20590de84725bab69cde93bd6e8ed360a13cc5421a7e7364ddeb93e9af53d6da"


def model_path():
    root = os.environ.get("TESSDATA_PREFIX", "/usr/share/tesseract-ocr/5/tessdata")
    return Path(root) / (MODEL + ".traineddata")


def identity():
    path = model_path()
    simplified = path.parent/"chi_sim_vert.traineddata"
    return {"policy": POLICY_VERSION, "model": MODEL,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
            "simplified_sha256": hashlib.sha256(simplified.read_bytes()).hexdigest() if simplified.is_file() else None,
            "auxiliary": {name: hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
                          for name in ("chi_tra", "chi_sim", "eng")
                          for p in [path.parent/(name+".traineddata")]}}


@dataclass(frozen=True)
class ChineseOcrResult:
    text: str
    regions: tuple


@dataclass(frozen=True)
class Column:
    left: int
    right: int
    top: int
    bottom: int
    glyph_bands: tuple


def runs(mask, gap=0):
    positions = np.flatnonzero(mask)
    if not len(positions):
        return []
    groups = np.split(positions, np.where(np.diff(positions) > gap + 1)[0] + 1)
    return [(int(g[0]), int(g[-1]) + 1) for g in groups]


def vertical_columns(image):
    """Projection evidence only: no recognized text or document profile involved."""
    ink = np.asarray(image.convert("L")) < 160
    height, width = ink.shape
    # Use width so cropping a header or adding vertical whitespace cannot change columns.
    bands = runs(ink.sum(axis=0) > max(4, width * .0125), gap=max(2, width // 180))
    columns = []
    for left, right in bands:
        if right - left > width * .12 or right - left < 5:
            continue
        left, right = max(0,left-10), min(width,right+10)
        strip = ink[:, left:right]
        extent = strip.shape[1] - strip.argmax(axis=1) - strip[:, ::-1].argmax(axis=1)
        extent[~strip.any(axis=1)] = 0
        rows = runs(extent >= max(4,(right-left)*.20), gap=int((right-left)*.65))
        glyphs = [(max(0,a-8), min(height,b+8)) for a, b in rows if b-a >= max(5, (right-left)*.3)]
        if glyphs and sum(b-a for a,b in glyphs) > (right-left)*.8 and glyphs[-1][1] - glyphs[0][0] > (right-left)*1.5:
            columns.append(Column(left, right, glyphs[0][0], glyphs[-1][1], tuple(glyphs)))
    return sorted(columns, key=lambda c: c.left, reverse=True)


def read_region(loader, image, path, *, language, psm, page_index):
    image.save(path)
    result = loader._run_process(
        [loader.tesseract_cmd, str(path), "stdout", "-l", language,
         "--tessdata-dir", str(model_path().parent), "--psm", psm,
         "-c", "tessedit_create_tsv=1"],
        purpose="text_recovery", stage="ocr", page=page_index)
    if result.returncode:
        raise PdfExtractionError(PdfExtractionFailureReason.EXTRACTION_FAILED,
                                 f"OCR engine failed on page {page_index + 1} (exit {result.returncode}).")
    words = [r for r in csv.DictReader(io.StringIO(result.stdout), delimiter="\t")
             if r.get("level") == "5" and r.get("text", "").strip()]
    tokens = []
    previous = None
    for word in words:
        key = tuple(word.get(k) for k in ("block_num","par_num","line_num"))
        text = word["text"]
        if previous and key != previous and not language.endswith("_vert"):
            tokens.append("\n")
        elif tokens and tokens[-1][-1:].isascii() and text[:1].isascii():
            tokens.append(" ")
        tokens.append(text)
        previous = key
    return ("".join(tokens),
            sum(float(w["conf"]) for w in words) / max(1, len(words)))


def horizontal_lines(ink):
    height,width = ink.shape
    lines = 0
    for top,bottom in runs(ink.any(axis=1),gap=3):
        xs = np.flatnonzero(ink[top:bottom].any(axis=0))
        if len(xs) and bottom-top < width*.10 and xs[-1]-xs[0] > width*.25:
            lines += 1
    return lines >= 2


def marginal_components(image, column):
    """Recognize geometry, not text: a margin label separated by a dotted rule."""
    if not (column.right < image.width * .08 or column.left > image.width * .92):
        return None
    crop = image.crop((column.left, column.top, column.right, column.bottom))
    ink = np.asarray(crop) < 160
    pieces = runs(ink.any(axis=1), gap=2)
    dots = []
    for index, (top, bottom) in enumerate(pieces):
        xs = np.flatnonzero(ink[top:bottom].any(axis=0))
        dots.append(bottom-top <= crop.width*.18 and xs[-1]-xs[0]+1 <= crop.width*.20)
    for first, last in runs(np.asarray(dots)):
        if last-first < 8 or first == 0 or last == len(pieces):
            continue
        top, bottom = pieces[first][0], pieces[last-1][1]
        if bottom-top < crop.height*.25:
            continue
        centers = [np.flatnonzero(ink[a:b].any(axis=0)).mean() for a,b in pieces[first:last]]
        if max(centers)-min(centers) > crop.width*.15:
            continue
        # Only separated 1-4 vertically stacked digits qualify as a page label.
        suffix = pieces[last:]
        if not 1 <= len(suffix) <= 4:
            continue
        return crop, pieces[0][0], pieces[first-1][1], suffix, (top,bottom)
    return None


def read_margin(loader, image, column, path, *, language, page_index):
    parts = marginal_components(image, column)
    if parts is None:
        return None
    crop, start, end, suffix, leader = parts
    title, confidence = read_region(loader, crop.crop((0,start,crop.width,end)), path,
                                    language=language, psm="5", page_index=page_index)
    digits = []
    for top,bottom in suffix:
        glyph = crop.crop((0,top,crop.width,bottom))
        padded = Image.new("L", (glyph.width+20,glyph.height+20),255)
        padded.paste(glyph,(10,10))
        digit, score = read_region(loader,padded,path,language="eng",psm="10",page_index=page_index)
        digit = digit.strip()
        if not re.fullmatch(r"[0-9]",digit) or score < 40:
            digit, score = read_region(loader,padded,path,language="eng",psm="13",page_index=page_index)
            digit = digit.strip()
        if not re.fullmatch(r"[0-9]",digit) or score < 40:
            return None
        confidence = min(confidence,score)
        digits.append(digit)
    if confidence < 40 or not re.search(r"[\u3400-\u9fff]",title):
        return None
    return title + " " + "".join(digits), confidence, {
        "region_role":"margin_label",
        "leader_bbox":[column.left,column.top+leader[0],column.right,column.top+leader[1]],
        "components":[{"role":"title","bbox":[column.left,column.top+start,column.right,column.top+end]},
                      *[{"role":"page_digit","bbox":[column.left,column.top+a,column.right,column.top+b]}
                        for a,b in suffix]],
    }


def recognize(loader, image_path, *, doc_name, page_index, _split=True, allow_review=False):
    """Return None when no confident vertical geometry exists; never route other languages."""
    languages = set(loader.ocr_language.split("+"))
    if not languages & {"chi_sim", "chi_sim_vert", "chi_tra", MODEL}:
        return None
    simplified = bool(languages & {"chi_sim", "chi_sim_vert"}) and not languages & {"chi_tra", MODEL}
    selected_model = "chi_sim_vert" if simplified else MODEL
    if not Path(image_path).is_file():
        return None
    with Image.open(image_path) as original:
        image = original.convert("L")
    columns = vertical_columns(image)
    if not any(c.bottom-c.top > (c.right-c.left)*5 for c in columns):
        if _split:
            return recognize_mixed_bands(loader,image,doc_name=doc_name,page_index=page_index)
        return None
    ink = np.asarray(image) < 160
    remainder = ink.copy()
    for column in columns:
        remainder[:,column.left:column.right] = False
    uncovered = float(remainder.sum()) / max(1,int(ink.sum()))
    logger.info("chinese_ocr_coverage doc=%s page=%s uncovered_ink_ratio=%.4f",doc_name,page_index+1,uncovered)
    if uncovered > .05 or len(columns) > 64:
        if _split and len(columns) <= 64:
            mixed = recognize_mixed_bands(loader,image,doc_name=doc_name,page_index=page_index)
            if mixed is not None:
                return mixed
        if horizontal_lines(ink):
            return None
        raise RawTextOcrFailedError(doc_name=doc_name,detail="chinese_layout_coverage_uncertain")
    fingerprint = identity()
    model_hash = fingerprint.get("simplified_sha256") if simplified else fingerprint["sha256"]
    logger.info("chinese_ocr_direction doc=%s page=%s direction=vertical columns=%s policy=%s model=%s sha256=%s",
                doc_name, page_index+1, len(columns), POLICY_VERSION, selected_model, model_hash)
    if model_hash != (SIMPLIFIED_SHA256 if simplified else MODEL_SHA256):
        raise PdfExtractionError(PdfExtractionFailureReason.ENGINE_UNAVAILABLE,
                                 "chinese_vertical_model_missing_or_unverified: install the pinned OCR model in the runtime image")
    texts = []
    regions = []
    uncertain_chars = 0
    total_chars = 0
    with tempfile.TemporaryDirectory(prefix="chinese-ocr-") as directory:
        for index, column in enumerate(columns):
            crop = image.crop((column.left, column.top, column.right, column.bottom))
            path = Path(directory) / f"column-{index}.png"
            margin = read_margin(loader,image,column,path,language=selected_model,page_index=page_index)
            if margin is not None:
                text,confidence,evidence = margin
                offset = sum(len(t)+1 for t in texts)
                regions.append({"region_index":index,"bbox":[column.left,column.top,column.right,column.bottom],
                                "char_start":offset,"char_end":offset+len(text),
                                "writing_mode":"vertical","column_order":"right_to_left",
                                "model":selected_model,"policy_version":POLICY_VERSION,
                                "confidence":confidence,"confidence_calibrated":False,
                                "quality_flags":[],**evidence})
                texts.append(text)
                total_chars += len(text)
                logger.info("chinese_ocr_margin doc=%s page=%s region=%s confidence=%.2f accepted=True",
                            doc_name,page_index+1,index,confidence)
                continue
            bands = list(column.glyph_bands)
            number = ""
            if len(bands) > 1 and bands[-1][0] > image.height*.75 and bands[-1][0]-bands[-2][1] > crop.width*3:
                a,b = bands.pop()
                number, number_confidence = read_region(loader,
                    image.crop((column.left,a,column.right,b)), path,
                    language="eng", psm="7", page_index=page_index)
                number = re.sub(r"\s+", "", number)
                if not re.fullmatch(r"\d{1,4}", number) or number_confidence < 40:
                    number, number_confidence = read_region(loader,
                        image.crop((column.left,a,column.right,b)),path,
                        language="eng",psm="13",page_index=page_index)
                    number = re.sub(r"\s+", "", number)
                logger.info("chinese_ocr_label doc=%s page=%s region=%s confidence=%.2f numeric=%s",
                            doc_name,page_index+1,index,number_confidence,bool(re.fullmatch(r"\d{1,4}",number)))
                if not re.fullmatch(r"\d{1,4}", number) or number_confidence < 40:
                    raise RawTextOcrFailedError(doc_name=doc_name, detail=f"chinese_column_label_uncertain:page={page_index+1}:region={index}")
            body = image.crop((column.left,bands[0][0],column.right,bands[-1][1]))
            cleaned = Image.new("L", (body.width+20,body.height+20),255)
            cleaned.paste(body,(10,10))
            text, confidence = read_region(loader, cleaned, path, language=selected_model, psm="5", page_index=page_index)
            if confidence < 40 and crop.height < crop.width*4:
                glyphs = runs((np.asarray(crop) < 160).sum(axis=1) > 0, gap=5)
                if 1 <= len(glyphs) <= 4:
                    parts = [read_region(loader, crop.crop((0,a,crop.width,b)), path,
                                         language="chi_sim" if simplified else "chi_tra", psm="13", page_index=page_index)
                             for a,b in glyphs]
                    text = "".join(t for t,c in parts)
                    confidence = min(c for t,c in parts)
            cjk = len(re.findall(r"[\u3400-\u9fff]", text))
            coverage = len(re.sub(r"\s", "", text)) / max(1, sum(b-a for a,b in bands)/(column.right-column.left))
            short_label = len(text) <= 4 and crop.height < crop.width*4
            accepted = cjk >= 1 and .65 <= coverage <= 2.5 and (confidence >= 40 or short_label)
            total_chars += len(text)
            if confidence < 40:
                uncertain_chars += len(text)
            logger.info("chinese_ocr_region doc=%s page=%s region=%s bbox=%s psm=5 confidence=%.2f coverage=%.2f accepted=%s",
                        doc_name, page_index+1, index,
                        (column.left, column.top, column.right, column.bottom), confidence, coverage, accepted)
            if not accepted and not allow_review:
                raise RawTextOcrFailedError(doc_name=doc_name, detail=f"chinese_vertical_region_quality_failed:page={page_index+1}:region={index}")
            logical = (text + (" " + number if number else "")) if accepted else ""
            offset = sum(len(t)+1 for t in texts)
            regions.append({"region_index":index,"bbox":[column.left,column.top,column.right,column.bottom],
                            "char_start":offset,"char_end":offset+len(logical),
                            "writing_mode":"vertical","column_order":"right_to_left",
                            "model":selected_model,"policy_version":POLICY_VERSION,
                            "direction_evidence":"vertical_projection_column",
                            "direction_confidence":min(1.0,(column.bottom-column.top)/max(1,column.right-column.left)/12),
                            "confidence_calibrated":False,
                            "confidence":confidence,"coverage":coverage,
                            "accepted":accepted,
                            "raw_candidate":text if not accepted else None,
                            "quality_flags":["recognition_uncertain"] if not accepted else (["low_confidence_short_label"] if confidence < 40 else [])})
            texts.append(logical)
    if uncertain_chars > total_chars * .05 and not allow_review:
        raise RawTextOcrFailedError(doc_name=doc_name, detail="chinese_page_confidence_insufficient")
    logger.info("chinese_ocr_page doc=%s page=%s uncertain_chars=%s total_chars=%s",
                doc_name, page_index+1, uncertain_chars,total_chars)
    return ChineseOcrResult("\n".join(texts),tuple(regions))


def recognize_mixed_bands(loader,image,*,doc_name,page_index):
    """Support spatially separated header/body bands; ambiguous interleaving is not guessed."""
    ink = np.asarray(image) < 160
    bands = runs(ink.any(axis=1),gap=max(20,image.height//40))
    if not 2 <= len(bands) <= 16:
        return None
    parts = [image.crop((0,max(0,a-8),image.width,min(image.height,b+8))) for a,b in bands]
    evidence = [vertical_columns(part) for part in parts]
    if not any(any(c.bottom-c.top > (c.right-c.left)*5 for c in cols) for cols in evidence):
        return None
    texts, regions = [], []
    with tempfile.TemporaryDirectory(prefix="chinese-mixed-") as directory:
        for index,(part,(a,b)) in enumerate(zip(parts,bands)):
            path = Path(directory)/f"band-{index}.png"
            part.save(path)
            result = recognize(loader,path,doc_name=doc_name,page_index=page_index,_split=False)
            offset = sum(len(t)+1 for t in texts)
            if result is not None:
                for region in result.regions:
                    box = list(region["bbox"])
                    box[1] += max(0,a-8)
                    box[3] += max(0,a-8)
                    regions.append({**region,"region_index":len(regions),"bbox":box,"char_start":offset+region["char_start"],
                                    "char_end":offset+region["char_end"]})
                texts.append(result.text)
            else:
                text,confidence = read_region(loader,part,path,language=loader.ocr_language,
                                               psm="6",page_index=page_index)
                if confidence < 40 or not loader._score_ocr_candidate(psm="6",text=text).passed:
                    raise RawTextOcrFailedError(doc_name=doc_name,detail="chinese_mixed_region_uncertain")
                regions.append({"region_index":len(regions),"bbox":[0,max(0,a-8),image.width,min(image.height,b+8)],
                                "char_start":offset,"char_end":offset+len(text),
                                "writing_mode":"horizontal","line_order":"top_to_bottom",
                                "confidence":confidence,"policy_version":POLICY_VERSION})
                texts.append(text)
    return ChineseOcrResult("\n".join(texts),tuple(regions))
