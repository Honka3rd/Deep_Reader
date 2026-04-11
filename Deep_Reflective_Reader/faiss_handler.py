from typing import List, Dict, Set
import hashlib
import numpy as np
import json

import faiss
from llama_index.core.schema import BaseNode

from embedder import Embedder
from node_record import NodeRecord
from search_metadata import SearchMetadata


def hash_text(text: str) -> str:
    return hashlib.md5(text.strip().encode()).hexdigest()


def _normalize_text(text: str) -> str:
    """
    做最基本的文本標準化，用於 retrieval 去重。
    目前先做：
    - 去首尾空白
    - 多個空白壓成單個空格
    """
    return " ".join(text.strip().split())


class FaissHandler:
    faiss_index: faiss.IndexIDMap
    embedder: Embedder
    id_to_record: Dict[int, NodeRecord]
    next_id: int
    dimension: int

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        probe_vector: List[float] = self.embedder.get().get_text_embedding("dimension probe")
        self.dimension = len(probe_vector)
        self.reset()

    def reset(self) -> None:
        base_index: faiss.IndexFlatL2 = faiss.IndexFlatL2(self.dimension)
        self.faiss_index = faiss.IndexIDMap(base_index)
        self.id_to_record = {}
        self.next_id = 1

    def save_index(self, file_path: str) -> None:
        faiss.write_index(self.faiss_index, file_path)

    def load_index(self, file_path: str) -> None:
        loaded_index = faiss.read_index(file_path)
        self.faiss_index = loaded_index

    def save_records(self, file_path: str) -> None:
        payload: dict = {
            "next_id": self.next_id,
            "records": {
                str(node_id): {
                    "node_id": record.node_id(),
                    "node_key": record.node_key(),
                    "text": record.text(),
                    "source": record.source(),
                    "chapter": record.chapter(),
                    "position": record.position(),
                }
                for node_id, record in self.id_to_record.items()
            }
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load_records(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            payload: dict = json.load(f)

        self.next_id = payload["next_id"]
        self.id_to_record = {
            int(node_id): NodeRecord.from_persisted_dict(record_dict)
            for node_id, record_dict in payload["records"].items()
        }

    def add_nodes(self, nodes: List[BaseNode]) -> None:
        for index, node in enumerate(nodes, start=1):
            node_text: str = node.text.strip()
            node_id: int = self.next_id

            # 1. 向量化
            vector_list: List[float] = self.embedder.get().get_text_embedding(node_text)

            # 2. 明確轉成 FAISS 需要的格式
            vector_np: np.ndarray = np.array(vector_list, dtype=np.float32).reshape(1, -1)
            id_np: np.ndarray = np.array([node_id], dtype=np.int64)

            # 3. 存入 FAISS
            self.faiss_index.add_with_ids(vector_np, id_np)

            # 4. 建立映射
            self.id_to_record[node_id] = NodeRecord(node, node_id)
            self.next_id += 1

    def search(self, query: str, top_k: int = 3) -> List[SearchMetadata]:
        """
        根據 query 搜索最相似的文本塊。

        :param query: 使用者問題
        :param top_k: 返回前幾個相似結果
        :return: [(node_id, text), ...]
        """
        """
        根據 query 在 FAISS 中搜索相似 chunk。
        在 retrieval 階段做去重，避免相同文本重複返回。
        """
        # 1. 將 query 向量化
        query_vector_list: List[float] = self.embedder.get().get_text_embedding(query)
        query_vector_np: np.ndarray = np.array(query_vector_list, dtype=np.float32)

        # 2. 在 FAISS 中搜索
        distances, indices = self.faiss_index.search(
            np.array([query_vector_np]),
            top_k,
        )

        results: List[SearchMetadata] = []
        seen_texts: Set[str] = set()
        # 3. 根據返回的 node_id 拿回文本
        for score, faiss_id in zip(distances[0], indices[0]):
            if faiss_id == -1:
                continue

            record: NodeRecord | None = self.id_to_record.get(int(faiss_id))
            if record is None:
                continue

            text: str = record.text()
            normalized_text: str = _normalize_text(text)

            # retrieval 去重：保留第一個出現的文本
            if normalized_text in seen_texts:
                continue

            seen_texts.add(normalized_text)

            results.append(
                SearchMetadata(
                    faiss_id=int(faiss_id),
                    node_key=record.node_key(),
                    text=text,
                    score=float(score),
                    source=record.source(),
                    chapter=record.chapter(),
                    position=record.position()
                )
            )

        print("FaissHandler#search:", results)
        return results

    def build_context(self, query: str, top_k: int = 3) -> str:
        """
        給 LLM 使用的輔助方法：
        根據 query 搜到相關 chunk，拼成 context 字串。

        :param query: 使用者問題
        :param top_k: 返回前幾個 chunk
        :return: 拼接好的 context
        """
        results: List[SearchMetadata] = self.search(query, top_k)
        texts: List[str] = [result.text for result in results]
        print("FaissHandler#build_context:", texts)
        return "\n".join(texts)


    def get_record_by_id(self, node_id: int) -> NodeRecord | None:
        """
        根據 node_id 取回原文。
        """
        return self.id_to_record.get(node_id)

    def save_all(self, index_path: str, records_path: str) -> None:
        self.save_index(index_path)
        self.save_records(records_path)

    def load_all(self, index_path: str, records_path: str) -> None:
        self.load_index(index_path)
        self.load_records(records_path)