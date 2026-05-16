# language Detailed Design

## 1. Module Purpose

`language/` 提供語言代碼正規化、語言檢測、以及多語言 registry（scope/discourse/script 等），支援 parser/profile/question scope 的語言相關判斷。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Shared Utility Layer（跨 parser/profile/question）

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `language/language_code.py` | `LanguageCode` 與 resolver/infer heuristics | canonical language enum **[Code-Confirmed]** |
| `language/document_language_detector.py` | 文件主語言檢測與快取讀取 | profile/records/LLM fallback **[Code-Confirmed]** |
| `language/language_profile_registry.py` | 多語言 scope/local signals registry | `question/` 主要消費方 **[Code-Confirmed]** |
| `language/language_script_registry.py` | script system detection registry | `profile/parser_metadata_extractor` 消費 **[Code-Confirmed]** |
| `language/language_discourse_registry.py` | dialogue/discourse cue registry | 保守 speech hints policy **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 統一語言 code contract 與 alias normalize。 **[Code-Confirmed]**
2. 提供文件語言偵測流程（優先載入既有 artifacts，再 LLM）。 **[Code-Confirmed]**
3. 提供多語言 keyword/signal/script/discourse registry。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不負責 parser 邊界切分。 **[Code-Confirmed]**
2. 不直接做 task-layout projection。 **[Code-Confirmed]**
3. 不應成為 API 層 DTO。 **[Code-Confirmed]**

## 6. Important Data Structures / Contracts

- `LanguageCode`
- `LanguageCodeResolver`
- `LanguageProfileRegistry`
- `LanguageScriptRegistry`
- `LanguageDiscourseRegistry`
- `DocumentLanguageDetector`

## 7. Module Relationships

- used by: `document_structure/`, `profile/`, `question/`, `document_preparation/`
- depends on: `llm/`（only detector）
- reads from: profile/records artifact paths（detector）

## 8. Main Flows Involving This Module

1. prepare flow 語言檢測。 **[Code-Confirmed]**
2. question scope flow 語言相關 keyword/signal 決策。 **[Code-Confirmed]**
3. parser metadata flow script/discourse detection。 **[Code-Confirmed]**

## 9. Persistence / Side Effects

- read persistence：部分（language detector 讀 profile/records）
- write persistence：否
- call LLM：部分（detector fallback）

## 10. Known Legacy / Compatibility Behavior

1. `document_language_detector` 可從舊 records/profile 讀語言，作為兼容載入路徑。 **[Code-Confirmed]**

## 11. Current Risks

1. risk：語言偵測 fallback 到 EN 可能誤判
- why：下游 prompt/scope 可能偏移
- guardrail：保留 warning log + explicit unknown handling

2. risk：registry 擴張時配置分散
- why：不同 registry 可能語義漂移
- guardrail：維持 registry boundary 文檔

3. risk：多語言詞彙表偏差
- why：scope/discourse 誤判
- guardrail：保守 fallback + regression tests

## 12. Open Questions for Maintainer

1. `language_profile_registry.py` 長期是否仍保留，或再進一步收斂 registry 邊界？ **[Needs Confirmation]**

## 13. Suggested Next Documentation Improvements

1. 補各 registry consumer matrix。
