from dataclasses import dataclass

from document_preparation.prepared_document_assets import PreparedDocumentAssets
from document_structure.structured_document import StructuredDocument
from retrieval.faiss_index_bundle import FaissIndexBundle


@dataclass
class PreparedDocumentResult:
    """Usable preparation result with readiness snapshot + loaded artifacts."""

    assets: PreparedDocumentAssets
    structured_document: StructuredDocument | None
    bundle: FaissIndexBundle | None
