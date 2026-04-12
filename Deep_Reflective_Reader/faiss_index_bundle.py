from typing import Dict, List, Set

import faiss
import numpy as np

from embedder import Embedder
from node_record import NodeRecord
from search_metadata import SearchMetadata
from llm_provider import LLMProvider
from standardized.question_standardizer import QuestionStandardizer
from profile.document_profile import DocumentProfile
from prompt_assembler import PromptAssembler
from evaluated_answer.question_relevance import QuestionRelevanceEvaluator, AnswerMode

class FaissIndexBundle:
    faiss_index: faiss.IndexIDMap
    id_to_record: Dict[int, NodeRecord]
    dimension: int
    embedder: Embedder
    relevance_evaluator: QuestionRelevanceEvaluator
    profile: DocumentProfile
    node_key_to_faiss_id: Dict[str, int]
    chunk_index_to_faiss_id: Dict[int, int]

    def set_profile(self, profile: DocumentProfile):
        self.profile = profile
        return self

    def __init__(
        self,
        faiss_index: faiss.IndexIDMap,
        embedder: Embedder,
        llm_provider: LLMProvider,
        prompt_assembler: PromptAssembler,
        question_standardizer: QuestionStandardizer,
        relevance_evaluator: QuestionRelevanceEvaluator,
        id_to_record: Dict[int, NodeRecord],
        dimension: int,
        document_language: str,
    ):
        self.faiss_index = faiss_index
        self.embedder = embedder
        self.llm_provider = llm_provider
        self.prompt_assembler = prompt_assembler
        self.question_standardizer = question_standardizer
        self.relevance_evaluator = relevance_evaluator
        self.id_to_record = id_to_record
        self.dimension = dimension
        self.document_language = document_language
        self.node_key_to_faiss_id = {
            record.node_key(): faiss_id
            for faiss_id, record in self.id_to_record.items()
            if record.node_key()
        }
        self.chunk_index_to_faiss_id = {
            chunk_index: faiss_id
            for faiss_id, record in self.id_to_record.items()
            for chunk_index in [record.chunk_index()]
            if isinstance(chunk_index, int)
        }

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.strip().split())

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[SearchMetadata]:
        query_vector_list: List[float] = self.embedder.get_text_embedding(query)
        query_vector_np: np.ndarray = np.array(
            query_vector_list,
            dtype=np.float32
        ).reshape(1, -1)

        distances, indices = self.faiss_index.search(query_vector_np, top_k)

        results: List[SearchMetadata] = []
        seen_texts: Set[str] = set()

        for score, faiss_id in zip(distances[0], indices[0]):
            if faiss_id == -1:
                continue

            record: NodeRecord | None = self.id_to_record.get(int(faiss_id))
            if record is None:
                continue

            text: str = record.text()
            normalized_text: str = self._normalize_text(text)

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
                    position=record.position(),
                )
            )

        print("FaissIndexBundle#search:", results)
        return results

    def build_context(
        self,
        query: str,
        top_k: int = 3,
    ) -> str:
        results: List[SearchMetadata] = self.search(query, top_k)
        texts: List[str] = [result.text for result in results]
        print("FaissIndexBundle#build_context:", texts)
        return "\n".join(texts)

    def _record_from_faiss_id(self, faiss_id: int) -> NodeRecord | None:
        return self.id_to_record.get(int(faiss_id))

    def _resolve_neighbor_by_link(
        self,
        record: NodeRecord,
        is_prev: bool,
    ) -> NodeRecord | None:
        linked_node_key = record.prev_node_id() if is_prev else record.next_node_id()
        if linked_node_key is None:
            return None
        linked_faiss_id = self.node_key_to_faiss_id.get(str(linked_node_key))
        if linked_faiss_id is None:
            return None
        return self._record_from_faiss_id(linked_faiss_id)

    def _resolve_neighbor_by_chunk_index(
        self,
        record: NodeRecord,
        is_prev: bool,
    ) -> NodeRecord | None:
        center_chunk_index = record.chunk_index()
        if not isinstance(center_chunk_index, int):
            return None
        target_chunk_index = center_chunk_index - 1 if is_prev else center_chunk_index + 1
        target_faiss_id = self.chunk_index_to_faiss_id.get(target_chunk_index)
        if target_faiss_id is None:
            return None
        return self._record_from_faiss_id(target_faiss_id)

    def _resolve_neighbor(
        self,
        record: NodeRecord,
        is_prev: bool,
    ) -> NodeRecord | None:
        neighbor = self._resolve_neighbor_by_link(record, is_prev=is_prev)
        if neighbor is not None:
            return neighbor
        return self._resolve_neighbor_by_chunk_index(record, is_prev=is_prev)

    def _expand_side(
        self,
        center: NodeRecord,
        radius: int,
        is_prev: bool,
    ) -> List[NodeRecord]:
        side_nodes: List[NodeRecord] = []
        cursor: NodeRecord = center
        visited_node_ids: Set[int] = {center.node_id()}

        for _ in range(radius):
            neighbor = self._resolve_neighbor(cursor, is_prev=is_prev)
            if neighbor is None:
                break
            if neighbor.node_id() in visited_node_ids:
                break
            visited_node_ids.add(neighbor.node_id())
            side_nodes.append(neighbor)
            cursor = neighbor

        return side_nodes

    def build_local_window(
        self,
        center: SearchMetadata | int,
        radius: int = 1,
    ) -> List[str]:
        if radius < 0:
            raise ValueError("radius must be >= 0")

        center_faiss_id = center.faiss_id if isinstance(center, SearchMetadata) else int(center)
        center_record = self._record_from_faiss_id(center_faiss_id)
        if center_record is None:
            return []

        left_nodes = self._expand_side(center_record, radius=radius, is_prev=True)
        right_nodes = self._expand_side(center_record, radius=radius, is_prev=False)

        ordered_nodes: List[NodeRecord] = list(reversed(left_nodes)) + [center_record] + right_nodes
        return [node.text() for node in ordered_nodes]

    def build_context_with_window(
        self,
        query: str,
        top_k: int = 3,
        radius: int = 1,
    ) -> str:
        results: List[SearchMetadata] = self.search(query, top_k)
        merged_texts: List[str] = []
        seen_texts: Set[str] = set()

        for result in results:
            window_texts = self.build_local_window(result, radius=radius)
            for text in window_texts:
                normalized_text = self._normalize_text(text)
                if normalized_text in seen_texts:
                    continue
                seen_texts.add(normalized_text)
                merged_texts.append(text)

        print("FaissIndexBundle#build_context_with_window:", merged_texts)
        return "\n".join(merged_texts)

    def _extract_chunk_index_from_result(self, result: SearchMetadata) -> int | None:
        record = self.id_to_record.get(result.faiss_id)
        if record is None:
            return None
        chunk_index = record.chunk_index()
        if isinstance(chunk_index, int):
            return chunk_index
        return None

    @staticmethod
    def _build_retrieval_context_from_results(results: List[SearchMetadata]) -> str:
        return "\n".join(result.text for result in results)

    def _build_context_by_reading_position(
        self,
        results: List[SearchMetadata],
        session_active_chunk_index: int | None,
        near_chunk_threshold: int,
        local_window_radius: int,
    ) -> tuple[str, str]:
        if not results:
            print("FaissIndexBundle#context_mode: retrieval_mode (no_results)")
            return "", "retrieval_mode"

        best_result = results[0]
        best_chunk_index = self._extract_chunk_index_from_result(best_result)

        if (
            isinstance(session_active_chunk_index, int)
            and isinstance(best_chunk_index, int)
            and abs(best_chunk_index - session_active_chunk_index) <= near_chunk_threshold
        ):
            local_texts = self.build_local_window(best_result, radius=local_window_radius)
            if local_texts:
                print(
                    "FaissIndexBundle#context_mode: local_window_mode "
                    f"(active={session_active_chunk_index}, best={best_chunk_index}, "
                    f"threshold={near_chunk_threshold}, radius={local_window_radius})"
                )
                return "\n".join(local_texts), "local_reading_mode"

        print(
            "FaissIndexBundle#context_mode: retrieval_mode "
            f"(active={session_active_chunk_index}, best={best_chunk_index}, "
            f"threshold={near_chunk_threshold})"
        )
        return self._build_retrieval_context_from_results(results), "retrieval_mode"

    def answer_with_results(
        self,
        query: str,
        top_k: int = 3,
        session_active_chunk_index: int | None = None,
        near_chunk_threshold: int = 2,
        local_window_radius: int = 1,
    ) -> tuple[str, List[SearchMetadata]]:
        if self.profile is None:
            print("Warn:FaissIndexBundle.answer: profile is not ready")
        standardized_question = self.question_standardizer.standardize(
            query=query,
            document_language=self.document_language,
        )
        results = self.search(
            standardized_question.standardized_query,
            top_k,
        )
        answer_mode: AnswerMode = self.relevance_evaluator.evaluate(results)
        print("FaissIndexBundle#ask standardized_question:", standardized_question)
        print("FaissIndexBundle#ask answer_mode:", answer_mode)

        if answer_mode.level == "reject":
            print("FaissIndexBundle#context_mode: retrieval_mode (answer_reject)")
            return "Not found", results
        context, prompt_mode = self._build_context_by_reading_position(
            results=results,
            session_active_chunk_index=session_active_chunk_index,
            near_chunk_threshold=near_chunk_threshold,
            local_window_radius=local_window_radius,
        )
        prompt = self.prompt_assembler.build_answer_prompt(
            context=context,
            question=standardized_question,
            profile=self.profile,
            answer_mode=answer_mode,
            prompt_mode=prompt_mode,
        )
        return self.llm_provider.complete_text(prompt), results

    def answer(self, query: str, top_k: int = 3) :
        answer_text, _ = self.answer_with_results(query, top_k)
        return answer_text
