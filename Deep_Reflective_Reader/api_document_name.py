"""Public document-name type shared by JSON, path and query inputs."""
from typing import Annotated
from pydantic import AfterValidator, Field
from shared.document_name import document_name

DocumentName = Annotated[str, Field(strict=True, min_length=1, max_length=255),
                         AfterValidator(document_name)]
