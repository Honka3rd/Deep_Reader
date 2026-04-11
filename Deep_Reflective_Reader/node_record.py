from llama_index.core.schema import BaseNode


class NodeRecord:
    def __init__(self, node: BaseNode, node_id: int) -> None:
        self._node_id = node_id
        self._node_key: str = getattr(node, "node_id", None) or getattr(node, "id_", "") or str(node_id)
        self._node_text: str = node.text.strip()
        self.metadata: dict = getattr(node, "metadata", {}) or {}

    def node_id(self):
        return self._node_id

    def node_key(self) -> str:
        return self._node_key

    def text(self):
        return self._node_text

    def source(self):
        return self.metadata.get("source")

    def chapter(self):
        return self.metadata.get("chapter")

    def position(self):
        return self.metadata.get("position")

    @classmethod
    def from_persisted_dict(cls, data: dict) -> "NodeRecord":
        obj = cls.__new__(cls)
        obj._node_id = data["node_id"]
        obj._node_key = data["node_key"]
        obj._node_text = data["text"]
        obj.metadata = {
            "source": data.get("source"),
            "chapter": data.get("chapter"),
            "position": data.get("position"),
        }
        return obj