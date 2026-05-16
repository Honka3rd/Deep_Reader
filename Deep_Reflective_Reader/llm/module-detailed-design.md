# llm Detailed Design

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
