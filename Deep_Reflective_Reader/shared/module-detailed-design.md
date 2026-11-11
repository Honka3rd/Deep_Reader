# shared Detailed Design

## SCHEMA-03 name compatibility (2026-11-11, implemented)

The shared validator preserves printable punctuation in logical titles while
rejecting separators and ASCII drive prefixes. Platform-specific reserved-character
lists must not hide existing safe POSIX documents. Public schemas and repository
resolution consume the same rule; path containment stays in storage.
Evidence: `scripts/test_legacy_document_names.py` (file/real PostgreSQL HTTP and
TCP), `scripts/test_api_schema_audit.py` (unsafe paths and outside links) and
`scripts/test_learning_api_schemas.py` (shared input contract) pass.

## SCHEMA-01 shared document names (2026-11-11, implemented)

A pure document_name validator defines safe logical leaf identities for both
public schemas and file repository resolution. It uses only the standard library,
returns the normalized name without stripping a valid suffix, performs no IO and
does not establish document existence or lifecycle authorization. Pydantic/HTTP
validation remains in api_schemas; resolved path containment remains in storage.

## Vocabulary assessment contract (2026-11-09)

`learning_hint_assessment.py` defines immutable `HintAssessment` and `HintIssue`.
Nonnegative integer counts satisfy candidate = accepted + filtered + rejected;
accepted equals the saved item's count. Omitted sentences are counted separately;
allowlisted content-free issue codes have positive counts and no duplicates.
`LearningHints.assessment` is optional and appended after existing constructor
fields. Legacy payloads without it remain readable as unknown assessment, not
fabricated zero omissions. Combining batches sums counts/reasons; any legacy
unassessed batch makes the aggregate assessment unknown. Domain selection remains
in section_tasks; shared performs no IO, language detection or policy decisions.
The DTO/codec/filtering tests verify round trips, counts and legacy compatibility.

## Explicit document rename references (LLRN-12)

`SentenceResource.with_document_id(new_id)` 是純 identity migration helper：保留 namespace/hierarchy/epochs/hash、segmentation/schema version、source length、sentence spans，只更新 document binding 並重新計算 deterministic sentence/resource IDs。不讀寫資料、不切句、不驗證目前來源。Lifecycle owner 需在來源與偏好鎖／交易內驗證 rename 並遷移所有 typed key/result/operation/receipt/checkpoint references；不可把新 DTO 當作有效來源授權。

## Language Learning Reading Mode Plan (2026-10-28)

**[Code-Confirmed; LLRN-03 complete]** [Canonical design](../section_tasks/module-detailed-design.md#language-learning-reading-mode-plan-2026-10-28)；[task dependency queue](../section_tasks/module-checklist.md#language-learning-reading-mode-queue-2026-10-28)。已交付純資料契約與結構驗證；切句算法、生成、儲存、API 和 UI 仍由後續任務實作。

`learning_source.py` 定義來源與句子資源，`learning_results.py` 定義完整對齊譯文與詞語提示 payload。皆為 immutable dataclasses，只依賴標準庫及 shared；不查 profile/registry、不讀寫檔案、不呼叫 LLM。句子不成為 navigation hierarchy 或新的 artifact target level；`TaskUnit`、`TaskUnitContentBlock`、既有 common artifact types 及序列化保持相容。

### Source identity and exact projection

- `LearningSourceBinding` 包含 `scope_kind="workspace"`、server-resolved namespace/document/chapter/section/task-unit/content-block IDs、`structure_version`、`document_epoch`、`content_epoch`、`effective_content_hash`。Source hash 為 effective block 原文 UTF-8 SHA-256，不 normalize、trim 或改換行。Epoch 為既有 mutation token 類型的 opaque string，非可比較的整數；legacy 無值時明確為 None。DTO 不自行建立 epoch，實際 edit/Undo/reparse 已由 LLRN-12 source owner 持久化不回退的 identity 並在提交時重驗；DTO 本身仍不是 lifecycle 或生成授權。
- `SourceSpan(start,end)` 使用 **effective block-local Unicode code-point `[start,end)`**，不可為空或負值，不接受 bool/float/string coercion。不是舊 `TaskUnit.content` 的全單元 quote span，也不是 OCR page coordinates 或 UTF-16 index。
- `SentenceResource` 保存 schema_version=1、source binding、segmentation_version、可空 segmentation_language、source_length、ordered `SentenceSpan` tuple。`from_spans` 僅替既定 spans 賦予可重現 ID，不選擇切句邊界。ID 依 source binding、切句版本/語言、位置順序與 span 計算；resource reference 另包含完整 ordered spans 的 digest。來源 ownership/epoch、切句規則/語言或邊界改變會產生不同 reference，不按能力級別複製句子。
- Spans 必須依序、不重疊、不越界，decode 驗證 ID 是 backend deterministic ID。`project(effective_text)` 檢查原文 hash/length，產生 `SourceTextPart(sentence_id, span, text)`，以 sentence_id=None 表示前後及句間空白。Gaps 只允許 whitespace，遺漏標點或文字拒絕；每個句子不得純空白。所有 parts 串接精確還原正文。空字串/純空白來源可沒有句子，不製造 synthetic sentence。
- 前端後續只接收 backend 文字投影與 ID，不用 JS UTF-16 重新切片。Code-point 不等於 grapheme；實際切句器仍需自行避免不合理語言邊界。DTO 不判斷某組 spans 是否具語言學意義。

### Result payload contracts

| Contract | Contents / invariant |
| --- | --- |
| `SentenceResourceRef` | 完整 source binding + resource_id；不複製正文或全部 sentence spans |
| `LearningResultContext` | resource ref、source/explanation languages、policy_version、正整數 result_revision、provider/model/prompt_version provenance；原文與解釋語言字串不能相同 |
| `AlignedTranslation` | schema_version=1、context、ordered `TranslatedSentence(sentence_id, translated_text)`；文字型別且非空，ID 不重複，validate_against 要求與來源 ID 完整同序相等 |
| `LearningHints` | schema_version=1、context、`LearningProficiencyRef(scheme_id,scheme_version,level_id)`、items tuple；空 items 是有效已存結果，不等於 missing |
| `VocabularyHint` | sentence_id、block-local span、source_form、language_code、kind（word/phrase/idiom）、母語語境 meaning、可空 lemma/part_of_speech；同位置不重複，宣告語言須與 context.source_language 相同 |

Hints `validate_against` 要求 reference 相符、原文 hash 正確、每個詞語 span 在指定句子內且原文切片精確等於 source_form。重複出現的詞以明確位置識別，不用全文搜尋猜測。這只驗證來源與宣告語言一致；實際混語識別與能力適用性由 LLRN-11 處理，不把宣告 language_code 當成語言判定證據。

譯文不含 proficiency；提示不依賴任何 translation result revision。Result revision、model/prompt provenance 不改 sentence reference；完整 cache/admission key 與 operation 狀態仍屬 LLRN-05。本契約只描述完整結果，不能把部分 rows 當成完整 AlignedTranslation；chunk/partial/progress 契約由 LLRN-05/10 另定，不先發明不明確狀態。

### Validation and serialization boundary

`to_dict/from_dict` 提供 detached JSON-compatible copies，內部集合為 tuple、傳輸為 arrays。Schema version 必須明確等於 1，除上述向後相容的 assessment 外，未知/缺失/額外欄位與錯誤型別拒絕，不 coercion 或 read-time migration；錯誤碼不回顯正文。底層 JSON parser 的重複 key/資源限制由後續 transport/storage 邊界處理。

**反序列化只驗證資料結構，不證明來源仍有效。** Repository/service 使用順序：先以 hierarchy 解析 authoritative source binding 與 effective text，再呼叫 `sentences.validate_source(expected_source, effective_text)`，最後對譯文或提示呼叫 `validate_against(sentences[, effective_text])`。本層不能證明傳入 IDs 確實有該 parent 關係、持久化 epoch 已更新、語言/scheme/level 在 registry 有效，或翻譯語義正確。這些分別由 hierarchy repository、lifecycle、language/section_tasks 與生成驗收 owner 負責。Internal DTO 不是 client 可提交的 source-authority schema；API 欄位由 LLRN-14 固定。

### Verification (2026-10-28)

`scripts/test_learning_contracts.py`：25 項單元測試，涵蓋 Unicode/emoji/組合字/CRLF exact projection、empty/whitespace、source ownership/hash/epoch/reference、illegal spans/IDs/alignment、提示定位/空結果/級別分離、strict schema、immutability 與舊 TaskUnit 相容性。另執行既有 `test_shared_task_unit_content_blocks` 19 項與 `test_reading_interaction_common_artifact_model` 6 項函式回歸。無實際切句演算法、LLM、DB 或 UI 行為改動，不以此宣稱語義或端到端驗收完成。

## Generated Unit Identity Binding (2026-10-23)

Supporting CLEAN-10: pure `TaskUnit.with_unit_id` rebinds exact generated
`old_unit:content:<nonnegative integer>` block IDs when allocating a new unit ID.
Existing block metadata identity keys follow that assignment; custom block IDs,
text and source coordinates are preserved. The same-ID call is a no-op. This is
allocation support, not artifact migration, cleanup renumbering or persistence.
Normalizer and SQL writers use it explicitly; no reader mutates stored identities.
`test_cleanup_identity.py` covers generated/custom blocks, immutability, both
normalizers and real SQL projection/Undo after allocation.

## 1. Module Purpose

`shared/` 提供跨模組共用 DTO/抽象結果契約，包含 task artifacts、task units 與通用 result 基類。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Shared Utility Layer

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `shared/abstract_result.py` | Generic result 抽象基類 | success/payload/reason/cache_hit **[Code-Confirmed]** |
| `shared/artifact_target_model.py` | artifact target level/ref shared contract | single source of truth for target-level enum **[Code-Confirmed]** |
| `shared/common_artifact_model.py` | common artifact entity DTO for typed interaction outputs | one entity shape using `artifact_type` rather than category-specific artifact tables **[Code-Confirmed]** |
| `shared/task_artifacts.py` | summary/quiz/task artifacts schema | section/task-unit/document-level contracts **[Code-Confirmed]** |
| `shared/learning_source.py` | source-bound sentence DTOs、exact source projection、deterministic IDs | LLRN-03 **[Code-Confirmed]** |
| `shared/learning_results.py` | aligned translation / vocabulary hint typed payloads、reference validation | LLRN-03 **[Code-Confirmed]** |
| `shared/task_unit_model.py` | `TaskUnit` 與 `TaskUnitContentBlock` schema | parent_section_id、artifact nested fields、string->block adapter foundation **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 定義可序列化的共用資料契約。 **[Code-Confirmed]**
2. 統一 summary/quiz artifact 的 metadata 欄位。 **[Code-Confirmed]**
3. 作為 document_structure/section_tasks/profile 等模組共用基礎。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不負責業務流程（解析、查找、API route）。 **[Code-Confirmed]**
2. 不負責 persistence store。 **[Code-Confirmed]**

## 6. Important Data Structures / Contracts

- `AbstractResult[PayloadT]`
- `SummaryArtifact`
- `QuizArtifact`
- `TaskArtifacts`
- `DocumentTaskArtifacts`
- `ArtifactTargetLevel`
- `ArtifactTargetRef`
- `CommonArtifact`
- `CommonArtifactTarget`
- `TaskUnit`
- `TaskUnitContentBlock`

## 7. Module Relationships

- used by: `document_structure/`, `section_tasks/`, `app/`, `profile/`
- depends on: 無重依賴（基礎資料層）

## 8. Main Flows Involving This Module

1. artifact write/read flow 的 schema 載體。 **[Code-Confirmed]**
2. task-layout response 與 summary/quiz service payload 載體。 **[Code-Confirmed]**

## 9. Persistence / Side Effects

- persistence：無（DTO only）
- side effects：無

## 10. Known Legacy / Compatibility Behavior

1. `TaskUnit.is_fallback_generated` 欄位保留歷史 fallback 來源訊號。 **[Code-Confirmed]**
2. `DocumentTaskArtifacts` 保留 chapter_artifacts map compatibility。 **[Code-Confirmed]**

## 11. Current Risks

1. risk：schema 欄位增加後未同步所有 consumer
- why：序列化/反序列化不一致
- guardrail：保持 round-trip 測試

2. risk：artifact metadata version 演進缺少集中策略
- why：cache validity 判斷可能漂移
- guardrail：維持 version 欄位並文件化

## 12. Open Questions for Maintainer

1. `AbstractResult` 是否要擴充標準錯誤碼欄位？ **[Needs Confirmation]**

## 13. Suggested Next Documentation Improvements

1. 補 artifact metadata field glossary。

## 14. Rich Task-Unit Content Foundation and Future Direction

1. shared 層已新增 `TaskUnitContentBlock` 與 deterministic block-id helper（`<task_unit_id>:content:<index>`）作為 caller-neutral foundation。 **[Code-Confirmed]**
2. `TaskUnit` 現在含有 additive `content_blocks: list[TaskUnitContentBlock]` 內部表示；`TaskUnit.content` 仍維持 `str` 並保留 compatibility role。 **[Code-Confirmed]**
3. `TaskUnit` 初始化/反序列化時，若缺少 `content_blocks` 且 `content` 非空，會自動穩定化成單一 deterministic block；空字串內容則維持空 block 清單。 **[Code-Confirmed]**
4. 當 `content_blocks` 已存在時，`TaskUnit.to_content_blocks()` 回傳穩定化後的現有 blocks；`content` 仍可供 compatibility 使用。 **[Code-Confirmed]**
5. `TaskUnit.to_dict(include_content_blocks=False)` 預設維持既有 serialization 輸出；`include_content_blocks=True` 提供 additive rich-content round-trip。 **[Code-Confirmed]**
6. content block 定位為 task-unit 內部 render/interaction segmentation，不是 chapter/section/task_unit 之外的新 hierarchy level。 **[Maintainer-Confirmed] + [Code-Confirmed]**
7. content block 不得成為 parser authority；block-level artifact metadata/ids 僅作 interaction/evidence metadata，不是 hierarchy truth source。 **[Maintainer-Confirmed] + [Doc-Confirmed]**
8. 本輪僅 shared-layer internal stabilization，未在本任務中變更 task-layout DTO、API schema/route、或 persistence migration 機制。 **[Code-Confirmed]**
9. retrieval/LLM/artifact persistence integration 與更細粒度互動 targeting 仍屬 future direction。 **[Future Direction]**
10. shared 層已新增 `ArtifactTargetLevel`（`document/chapter/section/task_unit/content_block`）與 `ArtifactTargetRef` 作為 content-block artifact target foundation metadata。 **[Code-Confirmed]**
11. `TaskUnitContentBlock` 新增 additive `artifact_target_refs`（optional）欄位；舊 payload 缺少該欄位時仍可正常反序列化，`artifact_ids` compatibility 行為不變。 **[Code-Confirmed]**
12. `artifact_target_refs` 僅是 target metadata，不代表 artifact persistence write path，不具 parser authority，且不改變 hierarchy truth（仍以 `chapters[].sections[].task_units[]` 為準）。 **[Code-Confirmed] + [From HLD]**
13. 本輪未引入 artifact repository 依賴、未引入 task-layout/API/persistence migration，question/evaluated_answer/retrieval 的 block-level integration 仍是後續工作。 **[Code-Confirmed] + [Future Direction]**
14. `ArtifactTargetLevel` / `ArtifactTargetRef` 已抽取至 `shared/artifact_target_model.py`，`shared/task_unit_model.py` 與 `api_schemas.py` 共用同一 target-level contract，避免 enum duplicated-definition drift。 **[Code-Confirmed]**

## 15. Future Direction Note: Content Block Segmentation Design Preparation

> 本節是 segmentation algorithm 的 documentation/design preparation，非 implementation。 **[Doc-Confirmed]**

### 15.1 Segmentation Input/Output Contract

1. input baseline：`TaskUnit.content`（string）；可選 future inputs 僅限 deterministic metadata/context（例如 parser-originated boundary hints）。 **[Future Direction]**
2. output baseline：ordered deterministic `TaskUnitContentBlock[]`，需保留原文順序並保持可回溯到原始 `content`。 **[Future Direction]**
3. compatibility requirement：`TaskUnit.content` 仍保留 compatibility role；multi-block generation 不得破壞既有 simple string consumer。 **[Code-Confirmed] + [Future Direction]**
4. segmentation output 屬 task-unit internal rich-content representation，不是 hierarchy truth。 **[Maintainer-Confirmed] + [From HLD]**

### 15.2 Segmentation Priority Strategy (Deterministic-Only)

1. 建議優先級：paragraph-first -> heading-aware merge/split -> list-aware split -> sentence fallback。 **[Future Direction]**
2. table/code block 先記為 future consideration，需 deterministic parsing rule 才能納入。 **[Future Direction]**
3. segmentation 必須 deterministic，不依賴 LLM、不依賴 semantic hallucination-style splitting。 **[Maintainer-Confirmed] + [Future Direction]**

### 15.3 Deterministic Block ID Strategy

1. 候選策略：
   - positional id（可讀性高、但插入漂移敏感）
   - hash-assisted id（穩定性較好、可讀性較低）
   - span-assisted id（對 quote linkage 友善、需 offset policy）
2. block id 不得依賴 random uuid / LLM output / retrieval result。 **[Maintainer-Confirmed] + [Future Direction]**
3. 方向：保留人類可讀前綴 + deterministic segment token（平衡可讀性與重算穩定性）。 **[Future Direction]**

### 15.4 Source Hash Strategy Direction

1. source hash 建議以 normalized text hash 為核心，需先定義 whitespace normalization policy。 **[Future Direction]**
2. 可評估 document-level hash + block-local hash 雙層策略，以區分全局編輯與局部漂移。 **[Future Direction]**
3. hash 只作 reference/invalidation support，不作 hierarchy authority。 **[From HLD] + [Future Direction]**

### 15.5 Quote Span Semantics Direction

1. quote/span metadata 可評估：absolute character offsets、block-local offsets、paragraph-local span。 **[Future Direction]**
2. quote span 需有 deterministic extraction rule，且與 `content_block_id` 一起描述 evidence target。 **[Future Direction]**
3. quote spans 不等於 hierarchy position，也不等於 parser authority。 **[Maintainer-Confirmed] + [Future Direction]**

### 15.6 Empty Content Behavior

1. default direction：empty content -> empty block list。 **[Future Direction]**
2. 不建議 synthetic empty semantic block，避免 downstream 對空內容誤判為有效 evidence target。 **[Future Direction]**

### 15.7 Block Size Policy Direction

1. 需定義 min/max block size、heading merge/split policy、oversized paragraph deterministic fallback。 **[Future Direction]**
2. policy 必須 parser-safe、deterministic、non-LLM。 **[Maintainer-Confirmed] + [Future Direction]**

### 15.8 Reparse Stability Risks

1. 主要漂移來源：paragraph insertion、heading renumbering、OCR normalization、whitespace normalization、parser version drift。 **[Future Direction]**
2. artifact/evidence linkage 在 reparse 後可能失配；後續需明確 stale/unresolved 分類邊界。 **[Future Direction]**

### 15.9 Compatibility Strategy (String -> Multi-Block)

1. staging direction：
   - Stage A：保留 `content` + 單塊 adapter（已存在）
   - Stage B：引入 deterministic multi-block generation（future）
   - Stage C：維持 compatibility mirror，逐步讓 consumers 以 `content_blocks` 為主
2. old persisted payload（無 `content_blocks`）仍需可載入並自動得到安全 fallback。 **[Code-Confirmed] + [Future Direction]**
3. 本節不引入 persistence migration、API change、task-layout payload change。 **[Doc-Confirmed]**

### 15.10 Guardrails (This Preparation Pass)

1. content block != hierarchy level。
2. segmentation 不改 parser authority。
3. segmentation 不改 hierarchy truth（仍以 `chapters[].sections[].task_units[]` 為準）。
4. segmentation 不依賴 LLM/retrieval。
5. segmentation 不新增 artifact persistence behavior。
6. segmentation metadata 僅 advisory/runtime support。

以上均屬 design preparation，不代表 implementation-ready algorithm 已落地。 **[Doc-Confirmed] + [Future Direction]**

## 16. Deterministic Content Block Segmentation Foundation (Implemented)

> 本節描述已落地的 shared-layer segmentation foundation；僅 shared 內部能力，不代表 endpoint/API/task-layout 行為變更。 **[Code-Confirmed]**

1. 已新增 explicit opt-in segmentation path：`TaskUnit.segment_content_blocks()` 與 shared helper `segment_task_unit_content(task_unit)`。 **[Code-Confirmed]**
2. `TaskUnit.to_content_blocks()` 預設行為維持 compatibility-safe 單塊適配，不會因本次變更自動改成多塊輸出。 **[Code-Confirmed]**
3. 已實作 deterministic split rules（目前最小可用）：
   - paragraph-first（blank-line paragraph boundaries）
   - list-item split（僅當整段每個 non-empty line 都是 deterministic list item pattern）
   - 無安全分割時 fallback 為單塊
   - 不做 LLM/NLP 語義切句。 **[Code-Confirmed]**
4. block id 策略：沿用 `build_default_content_block_id`，格式 `<task_unit_id>:content:<index>`；不依賴 random uuid/LLM/retrieval。 **[Code-Confirmed]**
5. segmented block metadata（advisory-only）包含：`source_hash`、`content_block_id`、`quote_span_start`、`quote_span_end`、`schema_version`。 **[Code-Confirmed]**
6. `source_hash` 目前以原始 `TaskUnit.content` 原文（不做 normalization）計算 deterministic SHA-256；同一輸入字串輸出穩定。 **[Code-Confirmed]**
7. quote span 語義：absolute character offsets，`quote_span_start` inclusive、`quote_span_end` exclusive，對應原始 `TaskUnit.content` 切片。 **[Code-Confirmed]**
8. empty content 行為維持：`content == ""` 時 segmentation 回傳空 block list（不生成 synthetic empty semantic block）。 **[Code-Confirmed]**
9. repeated segmentation 呼叫具 idempotent 結果；在同一內容輸入下輸出 block id、span、hash 穩定。 **[Code-Confirmed] + [Test-Confirmed]**
10. 本實作不變更 hierarchy truth、parser authority、artifact persistence、task-layout/API/persistence schema，也不接入 retrieval/LLM/question/evaluated_answer。 **[Code-Confirmed] + [Doc-Confirmed]**

## 17. Common Artifact Entity Contract

1. `CommonArtifact` provides one shared DTO shape for reading interaction artifacts, using `artifact_type` to distinguish `analysis`, `quiz`, and `critical_thinking_session`. **[Code-Confirmed]**
2. `CommonArtifactTarget` preserves hierarchy-aware target metadata (`document_id`, optional chapter/section/task-unit/content-block ids) and normalizes `book` to the shared `document` target level. **[Code-Confirmed]**
3. The common entity contract keeps type-specific content inside `payload` and bounded provenance inside `metadata`; it does not create category-specific artifact tables or category-specific top-level serialization roots. **[Code-Confirmed]**
4. The DTO is side-effect-free and does not own persistence, route mapping, parser authority, hierarchy truth, or runtime backend selection. **[Code-Confirmed] + [From HLD]**
