from dataclasses import dataclass

@dataclass
class SearchMetadata:
    # FAISS 裡的物理 id
    faiss_id: int
    # LlamaIndex 的 node.id
    node_key: str
    text: str
    # 檢索距離
    score: float
    source: str | None = None
    chapter: str | None = None
    position: int | None = None