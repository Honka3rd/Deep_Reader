# llm Detailed Design

## Learning single-attempt dispatch (LLRN-15, 2026-10-28)

`OpenAILLMProvider.complete_text_once_with_output_budget` 是既有 provider 的一次請求入口。
Learning durable dispatch acknowledgement 只授權一次付費呼叫，因此此入口使用
request-local OpenAI SDK client、`max_retries=0`；成功／失敗後均 close，避免 adapter
與 SDK 隱藏 retry。沿用目前 provider 的 model capability、API base、key、temperature
和原有 output-budget timeout policy，不添加 model/key/env 配置來源，也不改普通
`complete_text_with_output_budget` 的 shared client／retry 行為。DI lazy adapter 只接受
這個公開契約，缺契約或能力時 preflight 明確 unavailable。

`scripts/test_learning_provider_once.py` 四項測試以真 SDK + HTTP mock transport 覆蓋
Responses/chat-completions 成功、429/500/503/timeout、budget rejection、client close
及原 shared defaults 保持。每個 failure case 只有一次 transport call；不宣稱真模型
語義品質、exactly-once billing 或 provider 已接受請求的已知結果，unknown 仍須 durable
operation 查詢／明確重試。沒有修改 learning prompt 或語義品質 gate。

## Offline Preflight Follow-up (2026-10-24)

**[Code-Confirmed: WAIT-02/03; WAIT-05 accepted 2026-10-24]** Request-time token
counting is offline-only; WAIT-03 packages verified encoding data with the application
and gates startup offline. Packaging commands and image/local evidence live in the
[scripts design](../scripts/module-detailed-design.md#wait-03-implementation-2026-10-24). Frontend fixed request
deadlines are temporarily disallowed, not merely lengthened. This is a new WAIT
track, not a reversal of completed SCLEAN/FULL work. Ordered owner tasks:
[WAIT checklist](module-checklist.md#offline-preflight-follow-up-2026-10-24).
Full incident-book offline HTTP and browser acceptance is recorded in the
[scripts delivery](../scripts/module-detailed-design.md#wait-05-delivery-2026-10-24);
it does not claim deployment of the acceptance image to the production service.

### Incident Evidence

The 2026-10-24 investigation of `POST /documents/暗水幽灵/smart-cleanup/preflight`
with `{}`, request `3fe5a300b7b842118a8779bbc6d69c52`, established:

- UTC 11:34:11.501 request start; accepted OCR source reuse at 11:34:11.795
  (176196 characters), input adaptation completed at 11:34:11.825 (23.8 ms).
- Capacity accepted at 11:36:49.012; HTTP 200 at 11:36:49.061, total 157571.2 ms.
  Required input 125047 tokens including overhead, available 964067: no overflow.
- UI's 120-second abort precedes this response. No cleanup execution/model call
  was observed for this attempt; HTTP 200 alone is not evidence of cleaned text.
- Container `o200k_base` cache was created at UTC 11:36:48.743755, immediately
  before counting completed. Installed tiktoken loads absent encoding data through
  `requests.get()` without an explicit timeout. This strongly supports cold encoding
  download as the delay; there was no separate download-span log to measure it exactly.
- Read-only same-book probe after caching: accepted input 0.188 s; full cleanup
  capacity calculation 0.304 s. No cleanup, OCR/profile regeneration or content write.
  These are diagnostic observations, not a latency SLA or implementation acceptance.

### Provider Boundary

WAIT-02 keeps model-to-encoding resolution and local counting in `llm/`, shared by
cleanup and structuring. Prompt composition, capacity limits/safety margin and
consent fingerprints remain with their existing owners. Do not duplicate tokenizer
initialization in routes/controllers or change model selection to mask the incident.

Counting must use verified local encoding assets only. Runtime absence, corrupt hash,
unsupported mapping or unavailable tokenizer must return an explicit unavailable
count promptly, never trigger download, return zero, truncate the prompt or silently
choose another encoding. Preserve the existing conservative byte-upper-bound path
where applicable and its measurement label; if capacity cannot safely be established,
retain unavailable/over-capacity refusal. Preflight and dispatch use the same counting
policy/identity; a changed fallback/encoding decision requires fresh consent, not
bypassing the existing capacity fingerprint guard.

WAIT-03 packages encoding assets for supported configured models, with version/hash
verification and a runtime-readable location outside ephemeral `/tmp`. Docker build
and explicit local development setup may acquire assets with bounded download
timeouts/errors; API requests and ordinary provider imports may not. Startup performs
offline verification and may warm the encoding in memory, but startup warmup alone
must not substitute for packaged data. Missing local assets have explicit diagnostic
status and safe fallback, not a hidden request-path repair. Container recreation must
not discard the packaged assets. No model API credential is required for acquisition.

Provider logs distinguish local asset validation, cold in-memory initialization and
encoding duration/cache outcome, with model/encoding/version and counts only. Do not
log prompt/source text, consent tokens, API keys or arbitrary upstream exceptions.
App-owned preflight stages and real-book acceptance are defined in the linked owners.

### WAIT-02 Delivery (2026-10-24)

`offline_token_counter.py` owns one immutable encoding manifest and one shared
thread-safe counter. The five currently supported provider models map explicitly to
`o200k_base`; unsupported names or a conflicting installed-library mapping return
None. The public `encoding_asset_for_model` manifest is the future setup command's
single source for filename, URL, SHA-256 and size. It does not download anything.

The counter reads `DEEP_READER_TOKENIZER_ASSET_DIR` at process initialization, or
defaults to `llm/tokenizer_assets/`. Construction/import performs no asset reads.
First use verifies bounded local file bytes against the pinned size/hash, then
constructs tiktoken's public Encoding with the pinned o200k recipe. It never calls
the downloadable encoding registry, patches library globals, repairs files, or falls
back to a different encoding. Special-token spellings in source remain ordinary text.
Only successful verified encodings are cached; a lock serializes first initialization
across provider instances, while tokenization runs outside the lock. Verified in-memory
bytes remain the process snapshot even if the disk copy changes; restart activates
replacement setup assets. Failed initialization is not cached, allowing explicit setup
to supply a missing file later. Empty valid text can count as zero; failure cannot.

Missing/corrupt/unreadable assets or unavailable tokenizer return None promptly.
The existing budget owner retains the labelled UTF-8 byte upper bound, output reserve,
safety margin, full prompt/schema accounting and over-capacity refusal. Precise counts
carry an internal identity containing recipe, encoding hash and tiktoken version via
SemanticPlanProvider into the existing capacity fingerprint. Fallback transitions and
changed identities require fresh consent even if counts match; no API field is added.

Safe logs distinguish asset verification, cold initialization and counting, with
elapsed time, outcome/cache state, model/encoding/recipe and character/token counts.
Source, paths, keys, consent tokens and raw exception messages are excluded. App-wide
stage spans remain WAIT-04, not a second logging implementation in the provider.

Validation: 129 tests pass, including 13 new offline-counter cases. Network-denied
fresh import/missing assets, real-asset cold/warm processes with empty ephemeral cache,
concurrent initialization, corruption/read failures/dependency failures, mapping drift,
ordinary special tokens, multilingual upstream parity, schema accounting, fingerprint
rejection and no-completion guards pass. Immutable Madame Bovary and Red Roulette
snapshot/manifest tests count complete accepted text and retain all backup hashes;
existing three-book cleanup-input tests also pass. Real encoding bytes were copied
read-only from the existing container cache to a temporary test directory and verified,
not downloaded by runtime. These are counter/budget tests, not full incident-book HTTP
latency, rebuilt-image acceptance or live-model cleanup quality. No source/profile/OCR
regeneration, production writes or service restart. WAIT-03/04/05 remain open.

Reproduce from the backend root with `PYTHONPATH=.:scripts`,
`DEEP_READER_TEST_TOKENIZER_ASSETS=<verified asset directory>`,
`DEEP_READER_TEST_TOKENIZER_BOOKS=data/test-books`, and
`DEEP_READER_TEST_CLEANUP_INPUT_BOOKS=data/test-books/cleanup-20261023`:
`.venv/bin/python -m unittest test_offline_token_counter test_plan_prompt_budget test_llm_capacity_contract test_plan_prompt_metrics test_ocr_cleanup_planner test_cleanup_shared_engine test_enhancement_preflight test_enhancement_capacity_preflight test_standalone_cleanup_service.StandaloneCleanupServiceTests`.
Without the opt-in asset/book variables those specific real-data checks skip, not pass.
Until WAIT-03 supplies the default assets, ordinary runtime intentionally uses the safe
fallback rather than silently populating cache; deployment readiness is not claimed.

## Semantic Planner Output Scope (2026-10-21)

Supporting document_structure SEM-03: `complete_text_with_output_budget` accepts a
validated per-call allowance and optional response JSON schema. It never mutates
the shared `effective_output_budget`. Responses uses `text.format` JSON Schema;
Chat Completions uses `response_format`. Larger-than-default calls receive a
300-second timeout. Existing `complete_text` preserves the original interaction
allowance and request options. The planner adapter reserves exactly the allowance
it dispatches and includes the additional response schema in input accounting.
Structured output is a format guard, never source evidence or parser authority.
Reference: [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
Tests exercise isolated/default budgets, strict schema dispatch and invalid limits;
real-provider generation is accepted in the owning SEM-04 task, not a duplicate LLM task.

## Full-Source Capacity Contract Plan (2026-10-20)

Runtime evidence: Compose API startup and requests b75c8cada17443509d0656d39a54a93d
and c963c8e6c987443b828d4d6a05ebbf45 use gpt-4.1-mini; default config agrees.
Official verification on 2026-10-20:
[GPT-4.1 mini model](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
lists context window 1047576 and maximum output 32768 tokens. Existing numeric
constants match; 1047576 is NOT a separately documented input-only maximum.
FULL-01 now stores this value in `context_window_tokens`; `max_input_tokens` is
optional and reserved for an independently documented input ceiling (None for all
current OpenAI entries). Verified numbers are unchanged. On 2026-10-20 the remaining
entries were independently checked against official pages:
[GPT-4.1](https://developers.openai.com/api/docs/models/gpt-4.1),
[nano](https://developers.openai.com/api/docs/models/gpt-4.1-nano) (1047576/32768),
[GPT-4o](https://developers.openai.com/api/docs/models/gpt-4o),
[4o mini](https://developers.openai.com/api/docs/models/gpt-4o-mini) (128000/16384).

Current provider output policy sets target=500 and a 10%-of-output-maximum floor,
producing effective_output_budget=3276. This is application policy, not the model
output ceiling. It is not evidence of sufficient output for every book plan. Keep
model selection unchanged; capacity does not establish account TPM/quota availability.

FULL-01 delivered DTO separates context_window_tokens, optional independently
documented max_input_tokens, max_output_tokens and effective output allowance.
Optional unknown limits must not be invented. Migrate normalizers, budget resolvers,
config consumers, fixtures and logs with explicit compatibility; no global reinterpretation
of legacy max_input_tokens without auditing its callers. Tokenizer/model identity
must match the selected provider; fallback is labelled conservative and fail-closed
when no trustworthy capacity can be established.

Available input = min(known input-only limit, known context window minus effective
output allowance) minus safety margin. Apply only applicable known constraints;
at least one trustworthy applicable capacity bound is required. Reserve output once,
not twice. Margin is configurable, initially max(256, ceil(5% of limiting capacity)).
Full serialized prompt and transport overhead are included. Freeze policy/model/
tokenizer identities for preflight and revalidate before dispatch. No model call
is needed to measure capacity. Account rate limits remain distinct runtime failures.

Execution order: FULL-01 (llm), FULL-02 (document_structure), FULL-03 (app),
FULL-04 (UI), FULL-05 (document_structure integrated acceptance). Track 3/5 (60%);
FULL-02/03 are delivered in their owners; FULL-04/05 remain pending.

Implementation checkpoint: positional construction preserves the former shared-window
meaning in the third slot; all in-repository keyword fixtures migrated explicitly.
Old max_input_tokens-only construction without context_window_tokens now fails rather
than silently changing semantics. Input-only models explicitly pass context=None.
Strict positive integer validation rejects bool/invalid/unknown capacity. Shared
input_token_limit subtracts output only from context, then applies any input ceiling.
QA's existing TokenBudgetManager subtracts output downstream, so combined_token_limit
adapts to its combined budget contract; configured targets/utilization remain unchanged.
Normalizer, reading interaction context, full-text QA, config resolver, planner budget,
provider/retrieval/metrics logs and fixtures were audited/migrated. No persistence or
parser authority change. Enhancement budget policy v2 applies configurable margin
after limiting input capacity; unknown capability still rejects before model dispatch.
Existing non-enhancement fallback/excerpt behavior is retained, not presented as a
full-request guarantee. FULL-02 still owns complete source and transport accounting;
preflight/consent identity changes are now delivered by app FULL-03.

Validation: 66 unittest cases (9 new capacity cases plus 57 planner regressions),
10 standalone profile/config/interaction scripts, and git diff --check passed.
No paid LLM, production mutation, Docker restart or live deployment. Test entry:
PYTHONPATH=.:scripts .venv/bin/python -m unittest test_llm_capacity_contract
test_plan_prompt_budget test_plan_prompt_metrics test_structure_plan_contract
test_structure_plan_application. Standalone scripts run directly, not unittest discovery.

## Enhancement Token Counting Support (2026-10-20)

Supporting ENH-PLAN-04 (counted in document_structure only): LLMProvider exposes
optional count_input_tokens(text), default None. The original encoding_for_model
adapter could download on first initialization; WAIT-02 above replaces it with the
shared verified offline counter. Unsupported/missing tokenizer returns None without
logging source or exception text. encode uses disallowed_special=() so document
special-token spellings are ordinary input. This counts the local prompt, not
transport framing or billed provider usage. No completion request is made by this method.
Budget enforcement, conservative byte fallback,
output reserve and safe decision logs belong to document_structure, not provider
or parser authority. Mocked tokenizer adapter tests cover model selection, ordinary
special-token handling configuration and failure fallback; no live API evaluation.

## 1. Module Purpose

`llm/` 提供 completion provider 抽象與 OpenAI 實作，並定義模型能力 metadata，供 prompt 正規化與 token budget 計算使用。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Prompt / LLM Interaction Layer（provider sub-layer）

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `llm/llm_provider.py` | 抽象 provider + prompt text normalization helper | capability-aware excerpt normalize **[Code-Confirmed]** |
| `llm/openai_llm_provider.py` | OpenAI completion 實作 + model capability mapping | endpoint 分為 responses/chat_completions **[Code-Confirmed]** |
| `llm/llm_model_capabilities.py` | capability DTO | max input/output + endpoint kind **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 封裝 completion backend。 **[Code-Confirmed]**
2. 提供模型能力資訊供 budgeting/normalization。 **[Code-Confirmed]**
3. 以統一 helper 進行 prompt excerpt normalize。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不負責 prompt 業務內容設計。 **[Code-Confirmed]**
2. 不負責 parser 邏輯。 **[Code-Confirmed]**
3. 不負責 persistence。 **[Code-Confirmed]**

## 6. Important Data Structures / Contracts

- `LLMProvider`
- `OpenAILLMProvider`
- `OpenAIModelName`
- `LLMModelCapabilities`

## 7. Module Relationships

- used by: `document_preparation/`, `profile/`, `question/`, `section_tasks/`, `app/qa_coordinator.py`
- depends on: `auth/`

## 8. Main Flows Involving This Module

1. QA answer generation completion。 **[Code-Confirmed]**
2. profile/question standardization/scope fallback LLM calls。 **[Code-Confirmed]**
3. prompt excerpt normalization by capability。 **[Code-Confirmed]**

## 9. Persistence / Side Effects

- read/write persistence：否
- call external service：是（OpenAI）
- diagnostics：provider init/complete logs

## 10. Known Legacy / Compatibility Behavior

1. model name underscore alias compatibility（`gpt_4.1_mini` -> `gpt-4.1-mini`）。 **[Code-Confirmed]**

## 11. Current Risks

1. risk：capability mapping 過期
- why：token budget 可能不準
- guardrail：定期校驗官方模型規格 **[Needs Confirmation]**

2. risk：輸出 token floor policy 與成本衝突
- why：輸出預留會影響延遲/成本
- guardrail：保留 config target + floor log

## 12. Open Questions for Maintainer

1. 是否要把 capability mapping 外部化配置？ **[Needs Confirmation]**

## 13. Suggested Next Documentation Improvements

1. 補 provider capability version policy。

## 14. Future Direction Note: Reading Interaction Capability Boundary

> 本節記錄 reading interaction 對 model capability metadata 的使用邊界；不代表目前 implementation。 **[Maintainer-Provided] + [Future Direction]**

1. Reading interaction context selection should use provider capability metadata to decide whether full target context fits in one request. **[Maintainer-Provided] + [Future Direction]**
2. If target content exceeds model capability or configured budget, orchestration should use compact semantic context rather than forcing oversized prompts. **[Maintainer-Provided] + [Future Direction]**
3. The LLM provider should remain a completion/capability boundary. It should not own reading target resolution, prompt business wording, artifact persistence, permission policy, or quiz count defaults. **[Code-Confirmed] + [Future Direction]**
4. Interaction artifacts should record the effective model/capability context used by upper layers where available, but provider metadata itself must not become parser or hierarchy authority. **[From HLD] + [Future Direction]**
