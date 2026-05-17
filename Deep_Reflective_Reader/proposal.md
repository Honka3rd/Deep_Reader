# Deep Reflective Reader Proposal

> 文件定位：本 proposal 是「基於目前 codebase 狀態」的架構定位與下一階段規劃文件。  
> 事實標註規則：
> - **[Code-Confirmed]**：可由目前 source code 直接確認
> - **[Maintainer-Provided]**：由 maintainer 明確提供的需求或決策
> - **[Inferred]**：由多份文檔/模組關係合理推論
> - **[Future Direction]**：尚未實作，僅作下一階段提案

## 1. Project Purpose

Deep_Reflective_Reader 不是普通 parser。它是面向深度閱讀互動的 hierarchy-aware 系統：

`Document -> Structured Hierarchy -> Task Layout Projection -> Task-Unit Interaction -> Artifact Interaction -> Summary/Quiz Workflow`

核心目標：
- 穩定結構化閱讀互動（chapter/section/task-unit） **[Code-Confirmed]**
- 明確 read/write boundary，避免 hidden mutation **[Code-Confirmed]**
- 以 metadata/diagnostics 提升可觀測性（advisory-only） **[Code-Confirmed]**

## 2. Current Architecture Snapshot

### 2.1 Current Data Flow

```text
Raw Document
  -> Document Preparation Pipeline
  -> Pre-structure Profile (parser_metadata)
  -> Structured Hierarchy (chapters[].sections[].task_units[])
  -> Post-structure Metadata
  -> Task Layout Projection (read path)
  -> Task-Unit Content On-demand API
  -> Summary / Quiz / Artifacts
  -> Diagnostics / Recommendation
```

### 2.2 Runtime Contract

- Runtime hierarchy source：`chapters[].sections[].task_units[]` **[Code-Confirmed]**
- Runtime lookup：hierarchy-only + fail-fast **[Code-Confirmed]**
- Task-layout：metadata/projection read path，不承載 heavy content **[Code-Confirmed]**
- Task-unit content：透過獨立 read-only API 按需讀取 **[Code-Confirmed]**

## 3. Confirmed Implemented Capabilities

1. **Hierarchy-first runtime contract 已收斂**  
- `get_effective_sections` 與 runtime lookup 使用 hierarchy path。 **[Code-Confirmed]**

2. **Pure hierarchy persistence defaults 已落地**  
- 新 JSON 預設不輸出 root `sections[]` / `structure_nodes[]` mirror。 **[Code-Confirmed]**

3. **Runtime legacy fallback 已退場**  
- `allow_legacy_fallback` ordinary runtime surface 已移除。 **[Code-Confirmed]**
- runtime 不再默默 fallback 到 root `sections` / `structure_nodes`。 **[Code-Confirmed]**

4. **Legacy compatibility 已隔離**  
- normal `StructuredDocument.from_dict/from_json` 不再接受 legacy-only 作 runtime path。 **[Code-Confirmed]**
- legacy 僅保留 explicit migration-only loader 路徑。 **[Code-Confirmed]**

5. **Task-layout contract 已固定**  
- `/documents/task-layout` 維持 chapters-first projection；不回傳 top-level legacy mirrors。 **[Code-Confirmed]**

6. **Task-unit content lookup API 已存在**  
- `GET /documents/{doc_name}/task-units/{task_unit_id}/content`。 **[Code-Confirmed]**

7. **Profile/metadata 邊界已固定為 advisory**  
- `parser_metadata`（pre）+ `post_structure_metadata`（snapshot）已接入。 **[Code-Confirmed]**
- diagnostics 為 runtime projection，不回寫 profile。 **[Code-Confirmed]**

8. **Recommendation exists, manual reparse policy**  
- enhanced parse recommendation 有 score/reasons/metrics。 **[Code-Confirmed]**
- 目前策略為提示用戶手動 reparse，不自動觸發。 **[Maintainer-Provided]**

## 4. Legacy Retirement Status (Refreshed)

### 4.1 Obsolete Wording Removed

以下舊語意已不再適用：
- 「normal `StructuredDocument.from_dict()` 可直接以 legacy `sections` / `structure_nodes` 作 runtime path」→ 已過時 **[Code-Confirmed]**
- 「`allow_legacy_fallback` 尚未決定退場」→ 已過時 **[Code-Confirmed]**

### 4.2 Current Position

- runtime lookup 已 hierarchy-only **[Code-Confirmed]**
- legacy compatibility 已隔離為 explicit migration-only path **[Code-Confirmed]**
- compatibility 不是 runtime primary path，也不是 primary contract **[Inferred]**

## 5. Task Layout + On-demand Content Philosophy

1. `/documents/task-layout`：只做 metadata/projection read **[Code-Confirmed]**
2. 不做 hidden mutation，不寫回 profile **[Code-Confirmed]**
3. 前端互動路徑：

```text
task-layout
  -> task_unit_id
    -> on-demand task-unit content API
```

4. task-layout 不回傳 heavy full content；content lookup API 負責內容取回 **[Code-Confirmed]**
5. task-unit content endpoint 是 read-only，不承擔 persistence mutation **[Code-Confirmed]**

## 6. Artifact Governance Boundary

- artifact 是 interaction output，不是 hierarchy truth source **[Code-Confirmed]**
- artifact 不控制 parser authority，不控制 runtime hierarchy source **[Code-Confirmed]**
- hierarchy truth / artifact persistence / runtime projection 三者分離 **[Code-Confirmed]**

ownership split：
- hierarchy truth：`document_structure` contract **[Code-Confirmed]**
- artifact write path：coordinator + repository **[Code-Confirmed]**
- availability/diagnostics projection：task-layout/coordinator response **[Code-Confirmed]**

## 7. Next-Phase Proposal: Task Unit Rich Content Model

> 本節是下一階段規劃，不代表已落地。

### 7.1 Proposal Statement

`TaskUnit.content` 不應長期停留在 simple string。未來應演進為可定位、可互動、可掛 artifact 的 rich content model。 **[Maintainer-Provided]** + **[Future Direction]**

### 7.2 Suggested Terminology

- `TaskUnit`：reading interaction container **[Future Direction]**
- `TaskUnitContent` / `ContentBlock`：task unit 內可渲染內容單元 **[Future Direction]**
- `content_block_id` / `content_segment_id`：句子/段落/區塊級 stable id **[Future Direction]**
- `content`：內容字串本體 **[Future Direction]**
- `artifacts`：可掛載於 content block 的 interaction output（非 hierarchy truth） **[Future Direction]**

### 7.3 Why This Matters

- 前端需要句子/段落級選取與提問能力 **[Maintainer-Provided]**
- LLM 回答需要能引用具體 segment target **[Maintainer-Provided]**
- artifact 需要 finer-grained target，而不只 chapter/section/task-unit **[Maintainer-Provided]**
- 未來 note/annotation/evidence quote 需要 stable segment id **[Inferred]**
- task_unit_id 只能定位閱讀單元，不足以定位句子級互動 **[Inferred]**

### 7.4 Boundary Rules

- content segment 不是新的 persisted hierarchy level **[Future Direction]**
- 不引入 `Document -> Chapter -> Section -> TaskUnit -> Sentence` 的持久化層級 **[Future Direction]**
- `chapters[].sections[].task_units[]` 仍是 hierarchy source **[Code-Confirmed]**
- rich content 屬 task-unit internal render/interaction model **[Future Direction]**
- content-block artifact 是 interaction output，不是 hierarchy truth **[Future Direction]**
- 不把 LLM answer/artifact 當 parser authority **[Future Direction]**
- 不破壞 task-layout projection contract **[Code-Confirmed]** + **[Future Direction]**
- task-layout 仍不回傳 heavy full content payload **[Code-Confirmed]**
- on-demand content API 可演進為 rich payload **[Future Direction]**

## 8. Phased Rollout Plan (Proposal)

### Phase 1 — Proposal / Architecture Documentation Only
- 更新 proposal，固定方向與邊界
- 不改 code

### Phase 2 — Child-agent Checklist Preparation
- 更新相關 checklist（shared/document_structure/section_tasks/api_schemas/app/question/evaluated_answer）
- 視需要納入 retrieval/context

### Phase 3 — Model Design
- 在 shared 或 document_structure 設計 `TaskUnitContent` / `ContentBlock` DTO
- 提供 backward adapter：simple string content -> single content block

### Phase 4 — API Evolution
- 擴展 task-unit content endpoint 返回 rich content blocks
- 保持 task-layout heavy payload policy 不變

### Phase 5 — Artifact Targeting
- 支援 content-block-level artifact target
- target scope 可表達：document/chapter/section/task_unit/content_block

### Phase 6 — LLM / QA Integration
- question/evaluated_answer 支援 `content_block_id` 引用
- answer 可綁定 evidence/quote/artifact

> 上述各 phase 均屬 **[Future Direction]**。

## 9. Explicit Non-goals (This Round)

- 本輪不 coding **[Maintainer-Provided]**
- 本輪不改 API behavior **[Maintainer-Provided]**
- 本輪不改 persistence schema **[Maintainer-Provided]**
- 本輪不引入 sentence parser **[Maintainer-Provided]**
- 本輪不做 automatic LLM answer grounding **[Maintainer-Provided]**
- 本輪不改 task-layout response contract **[Maintainer-Provided]**
- 本輪不引入 retrieval dependency **[Maintainer-Provided]**

## 10. Open Questions for Maintainer

1. rich content model 的最小 segment granularity（句子/段落/混合）偏好？ **[Future Direction]**
2. content_block_id 是否需要跨 reparse 穩定，還是只需單次 version 穩定？ **[Future Direction]**
3. content-block-level artifact 的最小 metadata contract（source_hash/version/trace）是否先行定義？ **[Future Direction]**
4. rich content endpoint 是否要分版本（例如 `/v2/task-units/.../content`）？ **[Future Direction]**

## 11. Status Summary

- hierarchy-first runtime contract：已落地 **[Code-Confirmed]**
- pure hierarchy persistence defaults：已落地 **[Code-Confirmed]**
- runtime legacy fallback retirement：已落地 **[Code-Confirmed]**
- task-layout projection/read boundary：已固定 **[Code-Confirmed]**
- task-unit on-demand content API：已落地 **[Code-Confirmed]**
- rich task-unit content model：下一階段提案 **[Maintainer-Provided]** + **[Future Direction]**
