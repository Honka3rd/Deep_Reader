from dataclasses import dataclass


@dataclass(frozen=True)
class LineInfo:
    """Line text plus absolute char range in original raw text."""

    line: str
    stripped: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class HeadingCandidate:
    """Heading candidate with absolute start offset."""

    title: str
    char_start: int
    level: int = 1


@dataclass(frozen=True)
class HeadingPrecedenceResult:
    """Post-processed heading list plus optional section container metadata."""

    headings: list[HeadingCandidate]
    container_title_by_start: dict[int, str]
