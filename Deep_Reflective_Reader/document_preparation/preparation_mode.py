from enum import StrEnum


class PreparationMode(StrEnum):
    """Supported preparation workflows for document preparation pipeline."""

    BASE = "base"
    FREE_QA = "free_qa"

    @classmethod
    def resolve(cls, value: "PreparationMode | str") -> "PreparationMode":
        """Resolve external value to a valid preparation mode enum."""
        if isinstance(value, cls):
            return value

        normalized = value.strip().lower()
        try:
            return cls(normalized)
        except ValueError as error:
            supported = ", ".join(mode.value for mode in cls)
            raise ValueError(
                f"unsupported preparation mode: {value!r}. supported modes: {supported}"
            ) from error
