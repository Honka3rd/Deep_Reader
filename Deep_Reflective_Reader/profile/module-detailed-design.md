# profile Detailed Design

## 1. Module Purpose

`profile/` 提供文件 profile 與 metadata 能力：
- pre-structure parser metadata
- post-structure metadata enrichment
- profile persistence/reload
- 給 recommendation/diagnostics 的 advisory signals 來源

**[Code-Confirmed]**

## 2. Position in Overall Architecture

- Profile / Metadata Layer

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `profile/document_profile.py` | `DocumentProfile` 與 metadata data model | 包含 legacy `structure_profile` compatibility **[Code-Confirmed]** |
| `profile/document_profile_builder.py` | profile build orchestration（deterministic + LLM classification） | builder 主流程不主動產生 structure_profile **[Code-Confirmed]** |
| `profile/parser_metadata_extractor.py` | deterministic parser-relevant metadata extraction | script/discourse/region likelihood 等 **[Code-Confirmed]** |
| `profile/document_profile_evidence_builder.py` | profile classification evidence package build | candidate lines 是 evidence-only **[Code-Confirmed]** |
| `profile/post_structure_metadata_enricher.py` | post-parse hierarchy grounded metadata 統計 | title uniqueness / shape 等 **[Code-Confirmed]** |
| `profile/document_profile_store.py` | profile artifact load/save/exists/clear | JSON persistence **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 定義 profile 與 metadata schema（含 enum/value contract）。 **[Code-Confirmed]**
2. 產生 pre-structure parser metadata（deterministic first + LLM merge）。 **[Code-Confirmed]**
3. 在 structured 後做 post metadata enrichment（基於 hierarchy 事實）。 **[Code-Confirmed]**
4. 為 task-layout diagnostics 與 recommendation 提供 persisted signals。 **[Code-Confirmed]**
5. 保持舊 profile/structure_profile 載入相容。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不應直接控制 parser split boundary。 **[From HLD]**
2. 不應在 task-layout read path 主動回寫 mutation。 **[From HLD]**
3. 不應替代 document_structure 成為 hierarchy source-of-truth。 **[From HLD]**
4. 不承擔 hierarchy persistence contract（`chapters[].sections[].task_units[]` 的 source-of-truth 責任屬 `document_structure/`）。 **[Code-Confirmed] + [From HLD]**
5. 不承擔 artifact persistence contract（section/chapter/task-unit artifacts write path 由 coordinator + repository 管控）。 **[Code-Confirmed] + [From HLD]**
6. 不應成為 hidden mutation layer（不得藉由 diagnostics/runtime projection 覆寫 persisted profile）。 **[Code-Confirmed] + [From HLD]**

## 6. Important Data Structures / Contracts

- `DocumentProfile`
- `ParserRelevantMetadata`
- `PostStructureMetadata`
- `DocumentStructureProfile`（legacy compatibility）
- `LanguageCode`（profile document language）

## 7. Metadata Semantics Boundary

| Field Family | Nature | Source | Update Time | Authority |
|---|---|---|---|---|
| `parser_metadata` | pre-structure snapshot | deterministic + lightweight LLM | prepare Step 3 | advisory only |
| `post_structure_metadata` | hierarchy-grounded snapshot | structured hierarchy stats | prepare/reparse 時的 enrichment snapshot | advisory only |
| `profile_diagnostics` (API) | runtime projection | profile snapshot + current layout state | task-layout request time | advisory only |

**[Code-Confirmed] + [From HLD]**

## 8. Module Relationships

- depends on:
  - `language/` registries
  - `llm/` provider
  - `document_structure/`（post enrich read hierarchy）
- used by:
  - `document_preparation/`（prepare step）
  - `app/section_task_coordinator.py`（diagnostics/recommendation context）
  - `prompts/prompt_assembler.py`（QA prompt render profile block）
- reads/writes:
  - profile artifact JSON
- advisory relationship:
  - 對 parser 與 task layout 提供 advisory signals（非 authority） **[From HLD]**

## 9. Main Flows Involving This Module

1. pre-structure profile generation flow。 **[Code-Confirmed]**
2. deterministic metadata extraction flow。 **[Code-Confirmed]**
3. LLM classification merge/fallback flow。 **[Code-Confirmed]**
4. post-structure enrichment flow。 **[Code-Confirmed]**
5. diagnostics source flow（被 coordinator 消費；profile module 不負責 runtime diagnostics projection 組裝）。 **[Code-Confirmed]**

## 10. Cache-First and Invalidation Contract (Draft)

1. profile 層支援 cache-first（優先重用既有 profile）。 **[Code-Confirmed] + [From Proposal]**
2. manual reparse/force rebuild 是必要手動刷新機制。 **[From Proposal] + [From HLD]**
3. parser_metadata 對 common parser 不施加硬約束，僅允許省計算或輔助其他模組。 **[From Proposal]**
4. 最終 cache key / invalidation contract 尚待定案。 **[Needs Confirmation]**
5. API 是否回傳可觀測 invalidation reason code：已確認「有價值，保留方向」。 **[From Proposal] + [Needs Confirmation]**

## 11. Persistence / Side Effects

- read persistence：是（profile store load）
- write persistence：是（profile store save）
- mutate structured document：否
- generate runtime projection：否（提供資料來源）
- call LLM：是（builder classification）
- diagnostics only：否（同時有持久化責任；且不負責 task-layout diagnostics write-back）

## 12. Known Legacy / Compatibility Behavior

1. `DocumentProfile` 保留 legacy `structure_profile` 欄位與解析。 **[Code-Confirmed]**
2. 舊 profile JSON 可讀（缺 parser/post fields）。 **[Code-Confirmed]**
3. 新 profile 以 `parser_metadata` / `post_structure_metadata` 為主。 **[Code-Confirmed]**
4. `structure_profile` 為 compatibility-only，非主流程 authority。 **[Code-Confirmed] + [From HLD]**

## 13. Terminology Governance Audit

| Term | Current Meaning in This Module | Classification | Governance Decision |
|---|---|---|---|
| parser authority | profile metadata 不具 parser hard-control authority | **[Code-Confirmed] + [From HLD]** | 禁止將 metadata 寫成 parser hard dependency |
| `parser_metadata` | pre-structure advisory snapshot | **[Code-Confirmed]** | 允許 cache/省算輔助，不得直接控制 common parser 規則 |
| `post_structure_metadata` | prepare/reparse 時產生的 hierarchy-grounded snapshot | **[Code-Confirmed]** | 不等同 runtime 即時狀態 |
| diagnostics write-back | 在本模組語義下不允許 | **[Code-Confirmed] + [From HLD]** | task-layout diagnostics 不應回寫 profile |
| profile mutation | 合法 mutation 僅限 profile build/enrich 明確 write path | **[Code-Confirmed]** | 禁止 runtime projection 隱性覆寫 profile |
| LLM classification | 輔助性訊號，需 deterministic merge/fallback | **[Code-Confirmed]** | 不得成為 parser truth authority |
| `structure_profile` | legacy compatibility field | **[Code-Confirmed]** | 僅可讀相容，不可回到主流程 authority |
| `structure_nodes` | profile module 不擁有其主流程語義 | **[Code-Confirmed] + [Inferred]** | 避免在 profile 文檔暗示其為結構真值來源 |
| hierarchy persistence | 不屬 profile module 主要契約 | **[Code-Confirmed] + [From HLD]** | 由 `document_structure/` 持有 |
| artifact persistence | 不屬 profile module 主要契約 | **[Code-Confirmed] + [From HLD]** | 由 coordinator/repository 持有 |
| parser strategy | profile 提供 advisory signal，策略決策不在 profile module | **[Code-Confirmed] + [From HLD]** | 防止 authority creep |
| runtime projection | `profile_diagnostics` 屬 task-layout runtime projection | **[Code-Confirmed]** | profile 僅提供 source material，不組裝 projection |
| snapshot semantics | pre/post metadata 均為 snapshot，非即時 runtime state | **[Code-Confirmed]** | 文檔需固定 snapshot 語義 |

### Terminology Validation Notes

1. profile 文檔已明確 advisory-first：metadata 提供 signal，不提供 parser authority。  
2. pre/post metadata 均以 snapshot 語義呈現，避免與 runtime projection 混淆。  
3. diagnostics 已固定為 runtime projection，且禁止 write-back profile。  
4. `structure_profile` 已固定為 compatibility-only，避免 legacy revival。  

## 14. Current Risks

1. risk：metadata 被誤當 parser authority
- why：會造成 parser 行為不穩、循環依賴
- guardrail：固定 advisory-only contract

2. risk：LLM classification 輸出品質波動
- why：可能導致 metadata 偏差
- guardrail：deterministic first + strict parse fallback

3. risk：pre/post metadata 混用語義不清
- why：會導致 diagnostics 誤讀
- guardrail：在 DTO/文件標示 snapshot vs runtime projection

4. risk：legacy structure_profile 長期殘留增加維護負擔
- why：schema 解釋成本高
- guardrail：維持 compatibility-only 標註並設退場策略

## 15. Open Questions for Maintainer

1. `structure_profile` compatibility 欄位目前採短期保留策略（先不作為當前阻塞項）；待 runtime legacy fallback 收斂後再評估退場時程。 **[From Proposal]**
2. parser metadata cache-first policy 是否要在 profile schema 加入顯式 cache contract 欄位？ **[Needs Confirmation]**
3. cache invalidation reason code 的對外語義是否放在 profile diagnostics 還是 recommendation metadata？ **[Needs Confirmation]**

## 16. Suggested Next Documentation Improvements

1. 增加 pre/post metadata lifecycle 圖。
2. 增加 enum glossary（script/shape/risk）專章。
3. 補 profile artifact versioning policy。
