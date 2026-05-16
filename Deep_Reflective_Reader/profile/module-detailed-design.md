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
| `post_structure_metadata` | hierarchy-grounded snapshot | structured hierarchy stats | prepare Step 4.5 | advisory only |
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
5. diagnostics source flow（被 coordinator 消費）。 **[Code-Confirmed]**

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
- diagnostics only：否（同時有持久化責任）

## 12. Known Legacy / Compatibility Behavior

1. `DocumentProfile` 保留 legacy `structure_profile` 欄位與解析。 **[Code-Confirmed]**
2. 舊 profile JSON 可讀（缺 parser/post fields）。 **[Code-Confirmed]**
3. 新 profile 以 `parser_metadata` / `post_structure_metadata` 為主。 **[Code-Confirmed]**

## 13. Current Risks

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

## 14. Open Questions for Maintainer

1. `structure_profile` compatibility 欄位目前採短期保留策略（先不作為當前阻塞項）；待 runtime legacy fallback 收斂後再評估退場時程。 **[From Proposal]**
2. parser metadata cache-first policy 是否要在 profile schema 加入顯式 cache contract 欄位？ **[Needs Confirmation]**
3. cache invalidation reason code 的對外語義是否放在 profile diagnostics 還是 recommendation metadata？ **[Needs Confirmation]**

## 15. Suggested Next Documentation Improvements

1. 增加 pre/post metadata lifecycle 圖。
2. 增加 enum glossary（script/shape/risk）專章。
3. 補 profile artifact versioning policy。
