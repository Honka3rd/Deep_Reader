# llm Detailed Design

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
optional count_input_tokens(text), default None. OpenAILLMProvider uses tiktoken
encoding_for_model with its configured capability model name; unsupported/missing
tokenizer returns None without logging source or exception text. encode uses
disallowed_special=() so document special-token spellings are ordinary input.
This counts the local prompt, not transport framing or billed provider usage;
first encoding initialization can populate/download tokenizer cache. No completion
request is made by this method. Budget enforcement, conservative byte fallback,
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
