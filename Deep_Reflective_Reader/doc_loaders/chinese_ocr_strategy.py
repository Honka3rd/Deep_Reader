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

from .pdf_extraction_contract import PdfExtractionError, PdfExtractionFailureReason
from .pdf_ocr_language_policy import (
    VERTICAL_MODEL_HASHES, get_tesseract_language_for_document_language,
    normalize_tesseract_language_config,
)

logger = logging.getLogger(__name__)
POLICY_VERSION = "chinese-columns-v3-preserve-text"
MODEL = "chi_tra_vert"
MODEL_SHA256 = VERTICAL_MODEL_HASHES[MODEL]
SIMPLIFIED_SHA256 = VERTICAL_MODEL_HASHES["chi_sim_vert"]


def prefer_cjk_consensus(candidates):
    """Keep cross-script score outliers outside a strong observed CJK consensus."""
    nonempty = [candidate for candidate in candidates if candidate.text.strip()]
    # Match the scorer's CJK branch; tiny CJK labels cannot establish page script.
    supported = [candidate for candidate in nonempty
                 if candidate.cjk_ratio >= 0.35
                 and len(re.findall(r"[\u3400-\u9fff]", candidate.text)) >= 20]
    if len(supported) >= 2 and len(supported) * 3 >= len(nonempty) * 2:
        if len(supported) < len(nonempty):
            logger.info(
                "ocr_candidate_consensus candidate_count=%s nonempty_candidate_count=%s "
                "consensus_pool_count=%s rule=strong_cjk_consensus",
                len(candidates), len(nonempty), len(supported),
            )
        broad = [candidate for candidate in supported if candidate.psm != "6:zh"]
        refined = next((candidate for candidate in supported if candidate.psm == "6:zh"), None)
        if broad and refined is not None:
            baseline = max(broad, key=lambda candidate: (candidate.passed, candidate.score))
            # Fragmentation is advisory, not a reason to discard a fuller,
            # higher-scoring transcription established by observed Han evidence.
            # Never let a short/invalid refinement erase the broad result.
            allowed = {"fragmented_latin", "quality_rejected", "quality_passed", "too_short"}
            if (set(refined.reasons) <= allowed and refined.score >= baseline.score
                    and refined.cjk_ratio >= baseline.cjk_ratio
                    and len(re.findall(r"\S", refined.text)) >= .9 * len(re.findall(r"\S", baseline.text))
                    and len(re.findall(r"[\u3400-\u9fff]", refined.text))
                    >= len(re.findall(r"[\u3400-\u9fff]", baseline.text))):
                logger.info("ocr_candidate_consensus rule=observed_han_refinement")
                return [refined]
        return supported
    return nonempty


def horizontal_refinement_language(candidates, configured_language):
    """One bounded Chinese candidate after broad recognition establishes script.

    Installed packs alone never select this path. Preserve all broad candidates;
    Meaningful kana/Hangul evidence or substantial Latin content prevents narrowing;
    isolated cross-script recognition noise does not establish another script.
    """
    if configured_language != normalize_tesseract_language_config(None):
        return None
    nonempty = [candidate for candidate in candidates if candidate.text.strip()]
    for candidate in nonempty:
        other = len(re.findall(r"[\u3040-\u30ff\uac00-\ud7af]", candidate.text))
        if other >= 6 and other / max(1, sum(c.isalpha() for c in candidate.text)) >= .10:
            return None
    supported = [c for c in nonempty if c.cjk_ratio >= .65 and c.latin_ratio <= .20
                 and len(re.findall(r"[\u3400-\u9fff]", c.text)) >= 80]
    if len(supported) >= 2 and len(supported) * 3 >= len(nonempty) * 2:
        allowed = set(get_tesseract_language_for_document_language("zh").split("+"))
        # Preserve registry priority; changing model order changes transcription.
        return "+".join(model for model in configured_language.split("+") if model in allowed)
    return None


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
class ChineseOcrFallback:
    """Request bounded fallback with uncertainty in input-image coordinates."""

    reason: str
    bbox: tuple[int, int, int, int]


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
             if r.get("level") == "5" and r.get("text", "")]
    tokens = []
    previous = None
    for word in words:
        key = tuple(word.get(k) for k in ("block_num","par_num","line_num"))
        text = word["text"]
        if previous and key != previous and not language.endswith("_vert"):
            tokens.append("\n")
        elif (tokens and tokens[-1][-1:].isascii() and text[:1].isascii()
              and not tokens[-1][-1:].isspace() and not text[:1].isspace()):
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
    title = title if title.strip() else ""
    accepted = confidence >= 40 and bool(re.search(r"[\u3400-\u9fff]", title))
    components = [{"role":"title","bbox":[column.left,column.top+start,column.right,column.top+end],
                   "char_start":0,"char_end":len(title),"accepted":accepted,"confidence":confidence}]
    digits = []
    for top,bottom in suffix:
        glyph = crop.crop((0,top,crop.width,bottom))
        padded = Image.new("L", (glyph.width+20,glyph.height+20),255)
        padded.paste(glyph,(10,10))
        digit, score = read_region(loader,padded,path,language="eng",psm="10",page_index=page_index)
        if not re.fullmatch(r"[0-9]",digit.strip()) or score < 40:
            retry = read_region(loader,padded,path,language="eng",psm="13",page_index=page_index)
            if retry[0].strip():
                digit, score = retry
        digit = digit if digit.strip() else ""
        digit_accepted = bool(re.fullmatch(r"[0-9]",digit.strip())) and score >= 40
        offset = len(title) + (1 if title and (any(digits) or digit) else 0) + sum(map(len,digits))
        components.append({"role":"page_digit","bbox":[column.left,column.top+top,column.right,column.top+bottom],
                           "char_start":offset,"char_end":offset+len(digit),
                           "accepted":digit_accepted,"confidence":score})
        accepted = accepted and digit_accepted
        confidence = min(confidence,score)
        digits.append(digit)
    number = "".join(digits)
    text = title + (" " if title and number else "") + number
    return text, confidence, {
        "region_role":"margin_label",
        "accepted":accepted,"raw_candidate":text if not accepted else None,
        "quality_flags":["recognition_uncertain"] if not accepted else [],
        "leader_bbox":[column.left,column.top+leader[0],column.right,column.top+leader[1]],
        "components":components,
    }


def recognize(loader, image_path, *, doc_name, page_index, _split=True, allow_review=False):
    """Preserve recognized text with advisory quality; allow_review is compatibility-only."""
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
        # Let the caller's bounded full-image candidates cover unsupported geometry.
        return ChineseOcrFallback("chinese_layout_coverage_uncertain", (0, 0, image.width, image.height))
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
                offset = sum(len(t)+1 for t in texts) if text else len("\n".join(texts))
                evidence["components"] = [{**component,
                    "char_start":offset+component["char_start"],
                    "char_end":offset+component["char_end"]} for component in evidence["components"]]
                regions.append({"region_index":index,"bbox":[column.left,column.top,column.right,column.bottom],
                                "char_start":offset,"char_end":offset+len(text),
                                "writing_mode":"vertical","column_order":"right_to_left",
                                "model":selected_model,"policy_version":POLICY_VERSION,
                                "confidence":confidence,"confidence_calibrated":False,
                                "quality_flags":[],**evidence})
                if text:
                    texts.append(text)
                total_chars += len(text)
                if not evidence["accepted"]:
                    uncertain_chars += len(text)
                logger.info("chinese_ocr_margin doc=%s page=%s region=%s confidence=%.2f accepted=%s",
                            doc_name,page_index+1,index,confidence,evidence["accepted"])
                continue
            bands = list(column.glyph_bands)
            number = ""
            number_accepted = True
            if len(bands) > 1 and bands[-1][0] > image.height*.75 and bands[-1][0]-bands[-2][1] > crop.width*3:
                a,b = bands.pop()
                number, number_confidence = read_region(loader,
                    image.crop((column.left,a,column.right,b)), path,
                    language="eng", psm="7", page_index=page_index)
                if not re.fullmatch(r"\d{1,4}", number.strip()) or number_confidence < 40:
                    retry = read_region(loader,
                        image.crop((column.left,a,column.right,b)),path,
                        language="eng",psm="13",page_index=page_index)
                    if retry[0].strip():
                        number, number_confidence = retry
                number = number if number.strip() else ""
                number_accepted = bool(re.fullmatch(r"\d{1,4}", number.strip())) and number_confidence >= 40
                logger.info("chinese_ocr_label doc=%s page=%s region=%s confidence=%.2f numeric=%s",
                            doc_name,page_index+1,index,number_confidence,bool(re.fullmatch(r"\d{1,4}",number)))
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
                    replacement = "".join(t for t,c in parts if t.strip())
                    if replacement.strip() and (not text.strip() or all(t.strip() for t,c in parts)):
                        text = replacement
                        confidence = min(c for t,c in parts)
            text = text if text.strip() else ""
            cjk = len(re.findall(r"[\u3400-\u9fff]", text))
            coverage = len(re.sub(r"\s", "", text)) / max(1, sum(b-a for a,b in bands)/(column.right-column.left))
            short_label = len(text) <= 4 and crop.height < crop.width*4
            accepted = cjk >= 1 and .65 <= coverage <= 2.5 and (confidence >= 40 or short_label)
            logger.info("chinese_ocr_region doc=%s page=%s region=%s bbox=%s psm=5 confidence=%.2f coverage=%.2f accepted=%s",
                        doc_name, page_index+1, index,
                        (column.left, column.top, column.right, column.bottom), confidence, coverage, accepted)
            logical = text + (" " if text and number else "") + number
            offset = sum(len(t)+1 for t in texts) if logical else len("\n".join(texts))
            quality_flags = [] if accepted else ["recognition_uncertain"]
            if confidence < 40 and accepted:
                quality_flags.append("low_confidence_short_label")
            if not number_accepted:
                quality_flags.extend(["recognition_uncertain", "chinese_column_label_uncertain"])
            accepted = not quality_flags
            total_chars += len(logical)
            if quality_flags:
                uncertain_chars += len(logical)
            regions.append({"region_index":index,"bbox":[column.left,column.top,column.right,column.bottom],
                            "char_start":offset,"char_end":offset+len(logical),
                            "writing_mode":"vertical","column_order":"right_to_left",
                            "model":selected_model,"policy_version":POLICY_VERSION,
                            "direction_evidence":"vertical_projection_column",
                            "direction_confidence":min(1.0,(column.bottom-column.top)/max(1,column.right-column.left)/12),
                            "confidence_calibrated":False,
                            "confidence":confidence,"coverage":coverage,
                            "accepted":accepted,
                            "raw_candidate":logical if quality_flags else None,
                            "quality_flags":list(dict.fromkeys(quality_flags))})
            if len(bands) < len(column.glyph_bands):
                number_start = offset+len(text)+(1 if text and number else 0)
                regions[-1]["components"] = [
                    {"role":"body","bbox":[column.left,bands[0][0],column.right,bands[-1][1]],
                     "char_start":offset,"char_end":offset+len(text),"confidence":confidence},
                    {"role":"page_label","bbox":[column.left,a,column.right,b],
                     "char_start":number_start,"char_end":number_start+len(number),
                     "accepted":number_accepted,"confidence":number_confidence}]
            if logical:
                texts.append(logical)
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
            if isinstance(result, ChineseOcrResult):
                offset = sum(len(t)+1 for t in texts) if result.text else len("\n".join(texts))
                for region in result.regions:
                    box = list(region["bbox"])
                    box[1] += max(0,a-8)
                    box[3] += max(0,a-8)
                    translated = {**region,"region_index":len(regions),"bbox":box,
                                  "char_start":offset+region["char_start"],"char_end":offset+region["char_end"]}
                    if "components" in region:
                        translated["components"] = [{**component,
                            "bbox":[component["bbox"][0],component["bbox"][1]+max(0,a-8),
                                    component["bbox"][2],component["bbox"][3]+max(0,a-8)],
                            "char_start":offset+component["char_start"],
                            "char_end":offset+component["char_end"]} for component in region["components"]]
                    if "leader_bbox" in region:
                        leader = region["leader_bbox"]
                        translated["leader_bbox"] = [leader[0],leader[1]+max(0,a-8),leader[2],leader[3]+max(0,a-8)]
                    regions.append(translated)
                if result.text:
                    texts.append(result.text)
            else:
                text,confidence = read_region(loader,part,path,language=loader.ocr_language,
                                               psm="6",page_index=page_index)
                text = text if text.strip() else ""
                layout_uncertain = (isinstance(result, ChineseOcrFallback)
                                    or any(c.bottom-c.top > (c.right-c.left)*5 for c in evidence[index]))
                accepted = (not layout_uncertain and confidence >= 40
                            and loader._score_ocr_candidate(psm="6",text=text).passed)
                quality_flags = [] if accepted else ["recognition_uncertain", "chinese_mixed_region_uncertain"]
                if layout_uncertain:
                    quality_flags.append(result.reason if isinstance(result, ChineseOcrFallback)
                                         else "chinese_layout_coverage_uncertain")
                offset = sum(len(t)+1 for t in texts) if text else len("\n".join(texts))
                regions.append({"region_index":len(regions),"bbox":[0,max(0,a-8),image.width,min(image.height,b+8)],
                                "char_start":offset,"char_end":offset+len(text),
                                "writing_mode":"horizontal","line_order":"top_to_bottom",
                                "confidence":confidence,"confidence_calibrated":False,"policy_version":POLICY_VERSION,
                                "accepted":accepted,"raw_candidate":text if not accepted else None,
                                "quality_flags":quality_flags})
                if text:
                    texts.append(text)
    return ChineseOcrResult("\n".join(texts),tuple(regions))
