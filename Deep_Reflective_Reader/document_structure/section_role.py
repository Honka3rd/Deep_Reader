from enum import StrEnum


class SectionRole(StrEnum):
    """Canonical structured section region role."""

    TOC = "toc"
    FRONT_MATTER = "front_matter"
    MAIN_BODY = "main_body"
    APPENDIX = "appendix"
    BACK_MATTER = "back_matter"

    @classmethod
    def resolve(cls, value: "SectionRole | str | None") -> "SectionRole | None":
        """Resolve optional raw input to canonical SectionRole."""
        if value is None:
            return None
        if isinstance(value, cls):
            return value

        normalized = str(value).strip().lower().replace("-", "_")
        if not normalized:
            return None
        if normalized in cls._value2member_map_:
            return cls(normalized)
        return None
