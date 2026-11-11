"""Pure logical document-name contract; no filesystem or lifecycle authority."""


def document_name(value: str) -> str:
    if not isinstance(value, str) or not value.isprintable():
        raise ValueError("invalid_document_name")
    name = value.strip()
    # Reject Windows drive-relative identities too, without treating ordinary
    # title punctuation ("English: A Reader", "Who am I?") as a path.
    drive_prefix = (len(name) >= 2 and name[0] in
                    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" and name[1] == ":")
    if (not 1 <= len(name) <= 255 or name in {".", ".."}
            or drive_prefix or any(c in name for c in "/\\")):
        raise ValueError("invalid_document_name")
    # A suffix cannot turn an empty/dot identity into a valid storage namespace.
    for suffix in (".structured.json", ".pdf", ".txt"):
        if name.lower().endswith(suffix) and name[:-len(suffix)].strip() in {"", ".", ".."}:
            raise ValueError("invalid_document_name")
    return name
