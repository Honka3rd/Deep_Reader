# embeddings Detailed Design

## EMBED-01 SDK diagnostic boundary (2026-11-18, locally verified)

Embedding single/batch/dimension-probe calls retain their installed SDK behavior,
order, batching and exception propagation. Shared `logging_config` (main owner)
protects configured file/stream sinks at INFO and DEBUG: raw records from OpenAI,
llama-index, httpx and httpcore namespaces are not application diagnostics and must
not expose request/response bodies, source strings, credentials or arbitrary SDK
exception text. Application DEBUG remains available. Existing embedding stage IDs,
request correlation, duration/type and one safe completed-exchange event per HTTP
response remain; no vector/index/persistence/provider dispatch changes.

Acceptance uses real configure_logging, installed llama-index/OpenAI SDK and
MockTransport with synthetic source/key/provider markers and forbidden sockets.
Cover INFO/DEBUG, single/batch success/failure, probe, empty batch and partitioned
batch order/counts. Capture both configured sinks and check safe diagnostic counts.
No private books or live models are needed to reproduce this logging boundary.

Local evidence: `/tmp/embed-01-121cc96/report.txt`. New sink/SDK suite 13/13;
scoped formal regression 72 methods/functions passed, zero skips. Original external
DEBUG privacy assertion failed before the change and passes afterward; INFO control
and eight SDK adapter methods pass. Numeric 17/17 and caller 16/16 named controls
pass separately. No live model, private-book semantics, SQL, full app/TCP/UI or
Docker acceptance. Latest local review SHA still requires external approval.

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
