# embeddings Detailed Design

## 1. Module Purpose

`embeddings/` 提供 embedding backend 抽象、OpenAI 實作，以及向量正規化/相似度計算工具。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Peripheral / Support Layer（retrieval/question semantic support）

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `embeddings/embedder.py` | embedding 抽象介面 | single/batch/probe dimension **[Code-Confirmed]** |
| `embeddings/openai_embedder.py` | OpenAI embedding 實作 | 透過 llama-index OpenAIEmbedding **[Code-Confirmed]** |
| `embeddings/embedding_similarity_service.py` | 向量 normalize + best similarity | FAISS IP similarity helper **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 提供 embedding 生成能力。 **[Code-Confirmed]**
2. 提供 semantic similarity 計算基礎工具。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不負責索引持久化。 **[Code-Confirmed]**
2. 不負責 parser/profile 決策。 **[Code-Confirmed]**
3. 不負責 LLM completion。 **[Code-Confirmed]**

## 6. Important Data Structures / Contracts

- `Embedder`
- `OpenAIEmbedder`
- `EmbeddingSimilarityService`

## 7. Module Relationships

- used by: `retrieval/`, `question/question_scope_resolver.py`, `section_tasks/embedding_semantic_boundary_scorer.py`
- depends on: `auth/`, OpenAI via llama-index

## 8. Main Flows Involving This Module

1. index build embedding。 **[Code-Confirmed]**
2. query embedding search。 **[Code-Confirmed]**
3. semantic scope/boundary similarity scoring。 **[Code-Confirmed]**

## 9. Persistence / Side Effects

- read/write persistence：否
- call external service：是（OpenAI embedding API）

## 10. Known Legacy / Compatibility Behavior

No known legacy compatibility responsibility.

## 11. Current Risks

1. risk：embedding model 切換造成索引不相容
- why：向量維度/語義分佈改變
- guardrail：fingerprint + rebuild policy

2. risk：外部 API latency/cost
- why：prepare/semantic path 延遲
- guardrail：batch、cache 與 fallback policy

## 12. Open Questions for Maintainer

1. 是否需要支援非 OpenAI embedding provider？ **[Needs Confirmation]**

## 13. Suggested Next Documentation Improvements

1. 補 embedding dimension/version compatibility note。
