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

## 8. Next-Phase Proposal: DB-Centric Persistence Migration

> 本節是 DB-centric persistence 的 documentation/governance preparation，不代表目前 runtime behavior 已改變。 **[Maintainer-Provided] + [Future Direction]**

### 8.1 Proposal Statement

Deep_Reflective_Reader 未來應逐步從目前 file-based `data/` storage 演進到 DB-centric persistence。此方向是 future target，不是目前 runtime behavior；現有 `data/` file storage 在 migration 期間仍是有效、受支援的 temporary persistence scheme。 **[Maintainer-Provided] + [Future Direction]**

### 8.2 Migration Boundary Rules

- 目前 file-based `data/` storage 在 migration 期間仍有效，不得在本階段破壞。 **[Maintainer-Provided]**
- DB-centric persistence 是 future target，不是 current runtime behavior。 **[Maintainer-Provided] + [Future Direction]**
- JSON 與 relational persistence models 可在 migration 期間共存。 **[Maintainer-Provided] + [Future Direction]**
- file storage 在 DB readiness 被證明前，仍作 compatibility / fallback / migration source。 **[Maintainer-Provided] + [Future Direction]**
- hierarchy truth 仍是 `chapters[].sections[].task_units[]`。 **[Code-Confirmed] + [Maintainer-Provided]**
- DB migration 不得重新引入 root `sections[]`、`structure_nodes[]`、或 flat `task_units` 作 primary source。 **[Maintainer-Provided]**
- DB migration 必須保留 hierarchy-first lookup、fail-fast runtime behavior、與 explicit migration-only legacy handling。 **[Maintainer-Provided] + [Future Direction]**
- `document_structure` owns hierarchy persistence contract。 **[Code-Confirmed] + [Maintainer-Provided]**
- `config` 後續應擁有 backend selection / storage policy configuration。 **[Maintainer-Provided] + [Future Direction]**
- `document_preparation` 後續應透過 storage abstraction 寫入，而不是直接假設 file paths。 **[Maintainer-Provided] + [Future Direction]**
- `profile`、`retrieval`、與 artifact storage 需要各自獨立的 future DB migration tracks。 **[Maintainer-Provided] + [Future Direction]**
- user-uploaded documents are user-scoped and must not be shared across users. **[Maintainer-Provided]**
- raw file ownership/copyright boundary must be preserved; DB migration must not imply cross-user document sharing. **[Maintainer-Provided]**
- `data/` retirement 必須 gradual 且 gated by validation，不得以 breaking deletion 完成。 **[Maintainer-Provided] + [Future Direction]**

### 8.3 Phased Rollout Plan

### Phase 1 — Documentation and checklist preparation only
- 更新 proposal / module detailed design / checklist / progress。 **[Maintainer-Provided] + [Future Direction]**
- 不改 code、不改 runtime behavior、不新增 DB dependency。 **[Maintainer-Provided]**

### Phase 2 — Storage contract inventory
- 盤點 structured/profile/retrieval/artifact/raw file storage contracts。 **[Future Direction]**
- 明確區分 hierarchy truth、artifact output、profile metadata、retrieval index、raw source ownership。 **[Future Direction]**

### Phase 3 — Storage abstraction design
- 設計 file/DB coexistence storage abstraction。 **[Future Direction]**
- 保留 file-backed implementation 作 compatibility/fallback/migration source。 **[Future Direction]**

### Phase 4 — DB schema proposal, JSON + relational coexistence
- 提出 DB schema proposal，允許 JSON document 與 relational projection coexist。 **[Future Direction]**
- DB schema 只作 persistence representation，不作 parser authority 或 hierarchy identity authority。 **[Future Direction]**

### Phase 5 — dual-write or import/export migration tooling proposal
- 評估 dual-write、one-shot import、export/replay、validation tooling。 **[Future Direction]**
- migration tooling 必須可驗證 hierarchy parity 與 artifact target consistency。 **[Future Direction]**

### Phase 6 — read-path switch behind configuration
- 在 configuration/backend selection 後方切換 read path。 **[Future Direction]**
- 切換前必須保留 hierarchy-first fail-fast 與 explicit legacy migration-only boundary。 **[Future Direction]**

### Phase 7 — file storage retirement after validation
- 僅在 DB readiness、data parity、rollback/migration policy、user-scope isolation 驗證後，才進行 `data/` retirement。 **[Future Direction]**
- retirement 必須是 gradual policy，不是 breaking deletion。 **[Maintainer-Provided] + [Future Direction]**

> 上述 DB migration phases 均屬 **[Future Direction]**；本輪僅建立 proposal-level direction 與 checklist preparation。

### 8.4 Phase 1 StructuredDocument JSONB-First Evaluation Plan

The current Phase 1 `StructuredDocument` DB evaluation plan is documented in `db/structured-document-jsonb-evaluation.md`. It is documentation-only planning for evaluating PostgreSQL JSONB as an assumed evaluation backend for semantic hierarchy parity; it does not approve schema design, repository interfaces, migration execution, runtime read-path switching, or backend cutover. **[Maintainer-Provided] + [Future Direction]**

### 8.5 DB-Era Identity / Versioning / Reparse Golden Source

The current DB-era identity, document-level `structure_version`, hard reparse, derived-resource cleanup, and minimal parse event policy is documented in `db/module-detailed-design.md`. **[Maintainer-Provided] + [Future Direction]**

Current direction:

- DB-generated primary keys are the default internal identity and relational link foundation. **[Maintainer-Provided]**
- Separate public/domain IDs should be introduced only for concrete external stability requirements, not automatically for every hierarchy node. **[Maintainer-Provided]**
- Existing Python-generated `unit_id` values from file-based JSON are reference/import evidence only and must not become production DB identity. **[Maintainer-Provided]**
- `documents.current_structure_version` is the authoritative document-level hierarchy version; initial parse starts at `1`, and successful hard reparse advances it monotonically. **[Maintainer-Provided]**
- Early DB hierarchy persistence is current-state-only: no immutable hierarchy snapshots, no old row aliases, no staged candidate hierarchy persistence, and no per-row hierarchy versions. **[Maintainer-Provided]**
- Successful hard reparse validates the candidate hierarchy first, then atomically replaces current hierarchy rows, advances `current_structure_version`, explicitly deletes all document artifacts/content blocks, writes a minimal parse event record, and commits. **[Maintainer-Provided]**

This golden source remains documentation-only. It does not approve schema design, ORM models, repository interfaces, migrations, fixtures, runtime behavior changes, or API changes. **[Maintainer-Provided] + [Future Direction]**

## 9. Storage Contract Inventory

> 本節是 storage contract inventory preparation，用於 future DB-centric migration planning；不代表 schema design、runtime switch、或 storage abstraction implementation。 **[Code-Confirmed] + [Future Direction]**

### 9.1 Structured Document

- Owner: `document_structure` **[Code-Confirmed]**
- Current Persistence: structured JSON **[Code-Confirmed]**
- Truth Source: `chapters[].sections[].task_units[]` **[Code-Confirmed]**
- Future DB Candidate: Yes **[Future Direction]**
- Migration Priority: Highest **[Future Direction]**

### 9.2 Profile

- Owner: `profile` **[Code-Confirmed]**
- Current Persistence: profile artifact **[Code-Confirmed]**
- Truth Source: metadata snapshot only **[Code-Confirmed]**
- Future DB Candidate: Yes **[Future Direction]**
- Migration Priority: Medium **[Future Direction]**

### 9.3 Retrieval

- Owner: `retrieval` **[Code-Confirmed]**
- Current Persistence: FAISS + records + metadata **[Code-Confirmed]**
- Truth Source: Not authoritative **[Code-Confirmed]**
- Future DB Candidate: Partial **[Future Direction]**
- Migration Priority: Low **[Future Direction]**

### 9.4 Artifacts

- Owner: `document_structure` **[Code-Confirmed]**
- Current Persistence: artifact repository **[Code-Confirmed]**
- Truth Source: No **[Code-Confirmed]**
- Future DB Candidate: Yes **[Future Direction]**
- Migration Priority: Medium **[Future Direction]**

### 9.5 Raw Documents

- Owner: `doc_loaders` **[Code-Confirmed]**
- Current Persistence: uploaded files **[Code-Confirmed]**
- Truth Source: Canonical user document **[Inferred]**
- Future DB Candidate: Needs Confirmation **[Needs Confirmation]**
- Migration Priority: Separate Track **[Future Direction]**

### 9.6 Runtime Bundles

- Owner: `bundle_provider` / `bundle_factory` **[Code-Confirmed]**
- Current Persistence: runtime cache **[Code-Confirmed]**
- Truth Source: No **[Code-Confirmed]**
- Future DB Candidate: No **[Future Direction]**
- Migration Priority: None **[Future Direction]**

## 10. Future Direction: Storage Abstraction Boundary

> 本節定義 future storage abstraction ownership boundary；不代表 repository interface、schema、ORM、migration tooling、dual-write、read-path switch、或 runtime behavior 已實作。 **[Maintainer-Provided] + [Future Direction]**

### 10.1 Principle 1 — Business Modules Must Not Depend on Physical Storage Implementation

Business modules should depend on their domain storage contracts, not the physical backend representation. **[Maintainer-Provided] + [Future Direction]**

For example, `document_structure` should not care whether structured persistence is represented as:
- JSON
- JSONB
- PostgreSQL
- SQLite
- S3

The backend representation is an implementation detail behind future storage contracts, not a domain behavior contract. **[Future Direction]**

### 10.2 Principle 2 — Storage Backend Selection Belongs to `config`

`config` owns future backend selection, storage policy, and rollout policy. **[Maintainer-Provided] + [Future Direction]**

`config` does not own:
- hierarchy semantics
- profile semantics
- retrieval semantics

Backend selection can decide where persistence is routed, but it must not redefine what persisted data means. **[Future Direction]**

### 10.3 Principle 3 — Persistence Ownership Remains Domain-Owned

- `document_structure` owns hierarchy persistence semantics. **[Code-Confirmed] + [Future Direction]**
- `profile` owns profile persistence semantics. **[Code-Confirmed] + [Future Direction]**
- `retrieval` owns retrieval persistence semantics. **[Code-Confirmed] + [Future Direction]**
- `doc_loaders` owns raw document persistence semantics. **[Code-Confirmed] + [Future Direction]**

Future migration work must preserve these ownership boundaries instead of centralizing persistence meaning inside configuration, database schema, or infrastructure wiring. **[Maintainer-Provided] + [Future Direction]**

### 10.4 Principle 4 — Storage Abstraction Must Not Become New Authority

- Backend selection != truth source. **[Maintainer-Provided] + [Future Direction]**
- Database schema != hierarchy contract. **[Maintainer-Provided] + [Future Direction]**
- Storage implementation != parser authority. **[Maintainer-Provided] + [Future Direction]**

Hierarchy truth remains `chapters[].sections[].task_units[]`; profile and retrieval remain advisory/rebuildable surfaces; runtime caches remain cache-only. **[Code-Confirmed] + [Future Direction]**

## 11. Phased Rollout Plan (Proposal)

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

## 12. Explicit Non-goals (This Round)

- 本輪不 coding **[Maintainer-Provided]**
- 本輪不改 API behavior **[Maintainer-Provided]**
- 本輪不改 persistence schema **[Maintainer-Provided]**
- 本輪不引入 sentence parser **[Maintainer-Provided]**
- 本輪不做 automatic LLM answer grounding **[Maintainer-Provided]**
- 本輪不改 task-layout response contract **[Maintainer-Provided]**
- 本輪不引入 retrieval dependency **[Maintainer-Provided]**
- 本輪不實作 DB code、不新增 database dependency、不建立 database schema **[Maintainer-Provided]**
- 本輪不移除、不遷移、不破壞既有 `data/` file storage path **[Maintainer-Provided]**
- 本輪不改 parser/task-layout/API/runtime behavior **[Maintainer-Provided]**

## 13. Open Questions for Maintainer

1. rich content model 的最小 segment granularity（句子/段落/混合）偏好？ **[Future Direction]**
2. 若未來出現非 hard-reparse 的 partial update / history / rollback requirement，content_block_id 是否需要跨版本穩定？ **[Future Direction]**
3. content-block-level artifact 的最小 metadata contract（source_hash/version/trace）是否先行定義？ **[Future Direction]**
4. rich content endpoint 是否要分版本（例如 `/v2/task-units/.../content`）？ **[Future Direction]**

## 14. Status Summary

- hierarchy-first runtime contract：已落地 **[Code-Confirmed]**
- pure hierarchy persistence defaults：已落地 **[Code-Confirmed]**
- runtime legacy fallback retirement：已落地 **[Code-Confirmed]**
- task-layout projection/read boundary：已固定 **[Code-Confirmed]**
- task-unit on-demand content API：已落地 **[Code-Confirmed]**
- rich task-unit content model：下一階段提案 **[Maintainer-Provided]** + **[Future Direction]**
- DB-centric persistence migration：下一階段 documentation/checklist preparation **[Maintainer-Provided] + [Future Direction]**
- DB-era identity/versioning/reparse golden source：已完成 documentation-only baseline，位於 `db/module-detailed-design.md`；no schema/ORM/migration/runtime/API change implemented **[Maintainer-Provided] + [Future Direction]**
- storage contract inventory：已完成 documentation-only inventory preparation **[Code-Confirmed]**
- storage abstraction boundary：future-direction ownership design only；no repository interface/schema/runtime switch implemented **[Maintainer-Provided] + [Future Direction]**
