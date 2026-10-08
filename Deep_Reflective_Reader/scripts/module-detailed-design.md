# scripts Detailed Design

## 1. Module Purpose

`scripts/` 目前承擔主要 regression/smoke 測試入口（以 Python 腳本與 shell 腳本為主），覆蓋 hierarchy、task-layout、artifact persistence、profile metadata、real-document 流程等。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Tests Layer（package-like scripts folder）

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `scripts/test_task_layout_*.py` | task-layout projection/availability/persistence 回歸 | hierarchy-first 與 diagnostics **[Code-Confirmed]** |
| `scripts/test_hierarchy_*.py` | hierarchy write/read/target resolution 回歸 | section/chapter/task-unit 路徑 **[Code-Confirmed]** |
| `scripts/test_document_profile_*.py` | profile/parser_metadata/post_structure metadata 回歸 | schema/fallback/semantics **[Code-Confirmed]** |
| `scripts/test_post_structure_metadata_enrichment.py` | post metadata enrichment 統計與風險判斷 | title uniqueness/shape/task-unit coverage **[Code-Confirmed]** |
| `scripts/test_language_*` | script/discourse/registry 能力回歸 | multi-language policy **[Code-Confirmed]** |
| `scripts/test_rest_*.sh` | REST smoke/manual 验证輔助 | 非單元測試 **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 提供主要 regression 保護（目前未見獨立 `tests/` 目錄）。 **[Code-Confirmed]**
2. 提供真實文檔流程 smoke scripts。 **[Code-Confirmed]**
3. 驗證 legacy fallback 收斂與 hierarchy-only 進度。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不負責 production runtime。 **[Code-Confirmed]**
2. 不應承擔 API 文件。 **[Code-Confirmed]**

## 6. Important Data Structures / Contracts

- 無單一核心 DTO；主要消費各 module contract
- 關鍵測試群組契約：task-layout、profile metadata、hierarchy persistence

## 7. Module Relationships

- used by: 開發者/CI
- depends on: 全專案 runtime modules

## 8. Main Flows Involving This Module

1. regression test flow。 **[Code-Confirmed]**
2. real-document REST smoke flow。 **[Code-Confirmed]**

## 9. Persistence / Side Effects

- read/write persistence：部分（測試流程會讀寫 artifacts）
- external service：部分（需 OpenAI/embedding 時）

## 10. Known Legacy / Compatibility Behavior

1. 多個測試專門覆蓋 legacy 相容讀取與 fallback 收緊退場。 **[Code-Confirmed]**

## 11. Current Risks

1. risk：scripts 作為主要測試層，命名與分群易擴散
- why：維護與定位成本升高
- guardrail：維持 module docs 索引與測試群組規範

2. risk：部分測試依賴外部 API/環境
- why：本地可重現性不穩
- guardrail：區分 pure unit-like vs integration smoke

## 12. Open Questions for Maintainer

1. `scripts/` 是否長期作為正式 test layer，還是將來遷移到 `tests/`？ **[Needs Confirmation]**
2. 是否需要定義 smoke test 最小集與 full 集？ **[Needs Confirmation]**

## 13. Suggested Next Documentation Improvements

1. 建立 scripts 測試分類索引（unit-like / integration / real-doc smoke）。

## 14. Future Direction Note: Reading Interaction Regression Coverage

> 本節記錄 future tests for reading interactions；不代表目前 tests 已存在。 **[Future Direction]**

1. Regression coverage should prove read routes for persisted interaction artifacts do not auto-generate or call LLM when artifacts are absent. **[Maintainer-Provided] + [Future Direction]**
2. Target resolver coverage should verify document/chapter/section/task-unit ids, optional parent consistency checks, and fail-fast rejection of title-primary or legacy fallback targeting. **[Maintainer-Provided] + [Future Direction]**
3. Analysis coverage validates strict JSON success, insufficient-content success, invalid-output failure, current-artifact read behavior, and explicit read/generate split semantics. **[Code-Confirmed]**
4. Quiz coverage validates configured max counts, allowed type enum, fewer-than-max generation, answer payload validation, insufficient-content behavior, current-artifact read behavior, and explicit read/generate split semantics. **[Code-Confirmed]**
5. Critical-thinking coverage should validate question generation persistence, generated-but-unanswered status, answer submission, evaluation failure recovery, retry behavior, and completed status. **[Maintainer-Provided] + [Future Direction]**
6. Context strategy coverage verifies full-target mode within model budget, semantic compact fallback when the target exceeds capability, provenance metadata, and no title fallback for empty resolved content. **[Code-Confirmed]**
7. Artifact persistence coverage verifies hierarchy-aware replacement cleanup for interaction artifacts, including file-backed structured payload cleanup and PostgreSQL parser-level replacement payload sanitization. **[Code-Confirmed]**
8. Artifact-aware context coverage should verify higher-level generation can include compact lower-level artifact summaries as secondary context without replacing primary source context. **[Maintainer-Provided] + [Future Direction]**
9. Deduplication coverage should verify higher-level quiz generation receives lower-level quiz coverage signals and avoids repeating equivalent questions when possible. **[Maintainer-Provided] + [Future Direction]**
10. Critical-thinking abstraction coverage should verify higher-level critical-thinking prompts can build on lower-level session focus areas while still producing one first-version question. **[Maintainer-Provided] + [Future Direction]**
11. Persistence coverage verifies generated higher-level artifacts record referenced artifact ids/metadata, and hard reparse cleanup removes referenced-artifact metadata from replacement structured payloads. **[Code-Confirmed]**
12. Common artifact mapping coverage verifies `analysis`, `quiz`, and `critical_thinking_session` reading interaction artifacts map to one shared `CommonArtifact` entity shape with hierarchy-aware targets and no category-specific serialization roots. **[Code-Confirmed]**
