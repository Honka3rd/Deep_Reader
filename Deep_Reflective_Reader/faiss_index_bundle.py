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

    def answer(self, query: str, top_k: int = 3) :
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
            return "Not found"
        context = "\n".join(result.text for result in results)
        prompt = self.prompt_assembler.build_answer_prompt(
            context=context,
            question=standardized_question,
            profile=self.profile,
            answer_mode=answer_mode,
        )
        return self.llm_provider.complete_text(prompt)