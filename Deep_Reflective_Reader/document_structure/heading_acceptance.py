"""Conservative, source-local heading evidence shared by parser candidates."""
import re
from dataclasses import replace

POLICY_VERSION = "heading-evidence-v1"
_NUMBERED = re.compile(
    r"^(?:第[零〇一二三四五六七八九十百千萬万\d]+[章節节部篇卷]|"
    r"(?:chapter|part|book|appendix)\s+(?:\d+|[ivxlcdm]+|[a-z])\b|"
    r"\d+(?:\.\d+)*[.、\s]+\S)", re.IGNORECASE)
_REGION = re.compile(r"^(?:序|序言|前言|引言|.{1,16}自序|後記|后记|目錄|目录|preface|foreword|afterword|introduction|table of contents|contents)$", re.IGNORECASE)


def heading_decision(text: str, start: int, title: str | None, *, corroborated=False, normalize_line=None, heading_kind=None) -> str:
    if title is None:
        return "implicit"
    title = title.strip()
    if not title or len(title) > 160 or not 0 <= start < len(text):
        return "rejected"
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", start)
    line_end = len(text) if line_end < 0 else line_end
    line = text[start:line_end].strip()
    if normalize_line is not None:
        line = normalize_line(line)
    if text[line_start:start].strip() or line != title:
        return "rejected"
    if re.search(r"[。！？!?；;]", title) or title.startswith(('"', "'", "「", "『", "“")):
        return "rejected"
    kind = heading_kind(title) if heading_kind is not None else None
    # Common supplies the same language evidence used during candidate detection.
    # Projection/LLM callers retain their separate corroborated-source contract.
    if heading_kind is not None and re.search(r"(?:\.{2,}|…|·{2,})\s*\d+\s*$", title):
        return "rejected"
    if ((kind == "strong") if heading_kind is not None else _NUMBERED.match(title)) and not title.endswith("."):
        return "accepted"
    if corroborated:
        return "accepted"
    before = text[:line_start]
    after = text[line_end:]
    isolated_before = not before.strip() or bool(re.search(r"\n\s*\n$", before))
    following = "".join(after.strip().split("\n\n", 1)[0].splitlines())[:400]
    substantive_after = len(following) >= 20 and bool(re.search(r"[。！？.!?]", following))
    is_region = kind == "region" if heading_kind is not None else bool(_REGION.fullmatch(title))
    if is_region:
        return "accepted" if isolated_before and substantive_after else "uncertain"
    if isolated_before and after.startswith("\n\n") and substantive_after and len(title) >= 2:
        return "accepted" if not title.endswith(".") else "uncertain"
    return "uncertain"


def filter_common_sections(text, sections, *, normalize_line=None, heading_kind=None):
    """Unsupported boundaries stay in the preceding slice, never lose source."""
    result = []
    for section in sections:
        decision = heading_decision(text, section.char_start, section.title,
                                    normalize_line=normalize_line, heading_kind=heading_kind)
        if section.title is not None and decision != "accepted":
            if result:
                previous = result[-1]
                result[-1] = replace(previous, char_end=section.char_end,
                                     content=text[previous.char_start:section.char_end])
                continue
            section = replace(section, title=None, container_title=None,
                              section_role=None, level=1)
        result.append(section)
    return result


def validate_projection(text, sections):
    """Even bookmarks/TOCs cannot invent a heading absent from the destination."""
    from document_structure.section_role import SectionRole
    if not sections:
        return []
    if any(heading_decision(text, item.char_start, item.title, corroborated=True) != "accepted"
           for item in sections):
        return []
    result = list(sections)
    if result[0].char_start:
        first = result[0]
        if text[:first.char_start].strip():
            result.insert(0, replace(first, title=None, container_title=None, level=1,
                char_start=0, char_end=first.char_start, content=text[:first.char_start],
                section_role=SectionRole.FRONT_MATTER, section_kind=None, is_implicit_section=True))
        else:
            result[0] = replace(first, char_start=0, content=text[:first.char_end])
    if "".join(item.content for item in result) != text:
        return []
    return [replace(item, section_id=f"projection-{index}", section_index=index)
            for index, item in enumerate(result)]
