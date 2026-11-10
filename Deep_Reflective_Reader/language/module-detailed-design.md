# language Detailed Design

## Language Learning Reading Mode Plan (2026-10-28)

**[Code-Confirmed: LLRN-01 foundation only, 2026-10-28]** 已實作獨立 reading proficiency registry，完整模式、設定持久化、API、生成與 UI 尚待後續任務。[Canonical design](../section_tasks/module-detailed-design.md#language-learning-reading-mode-plan-2026-10-28)；[task dependency queue](../section_tasks/module-checklist.md#language-learning-reading-mode-queue-2026-10-28)。

### Reading proficiency contract

`language_proficiency_registry.py` 是專用無副作用資料/驗證模組；LanguageProfileRegistry 仍是唯一支援語言來源，排除 unknown。分級 association 不決定語言是否支援；新增語言缺 mapping 或移除後有殘留 mapping 時，coverage gate 明確失敗，不偷偷配英文或通用級別。未改現有 profile/scope/script/discourse/OCR registry，也不新增依賴、網路、儲存、模型呼叫或 parser authority。

- `ProficiencyLevel`：穩定字串 level_id、由低到高 order、label、reading_description。
- `ProficiencyScheme`：scheme_id、version、label、scope_note、immutable levels 與 reference_urls。
- `LanguageProficiencyDefinition`：目前 registry code 對應一個 scheme。
- `ResolvedReadingProficiency`：已驗證 language/scheme/level；identity 是 `(language_code, scheme_id, scheme_version, level_id)`，不是完整來源/生成 cache key，也不包含使用者或母語偏好。
- `LanguageProficiencyRegistry.list_definitions()` 先驗 coverage/資料完整性，依 language registry 順序返回 tuple。`get_scheme(code)` 僅接受目前精確 code，不接受 unknown、別名、region tags 或隱式 normalize。
- `resolve_selection(language_code=..., scheme_id=..., scheme_version=..., level_id=...)` 嚴格拒絕錯語言方案、過期版本、錯 level 及型別 coercion；TOPIK/TOCFL 的數字級別仍是字串，JLPT 以 N5→N1 排序，不能依字母或數字字面排序。
- `ProficiencySelectionError.code` 區分 language invalid、scheme mismatch、version mismatch、level invalid；`ProficiencyConfigurationError` 表示 maintained data/coverage 錯誤，不可當 missing preference。錯誤不回顯輸入。

Definitions/levels/results 皆 frozen dataclass 與 tuple；mapping 使用不可變映射。可用 dataclasses.asdict 產生獨立資料投影，未定義 REST response；傳輸、設定與 UI tasks 各自消費此公開契約。未來 DI readiness 可呼叫 `validate_coverage()`，本項未接入啟動流程。

### Curated language assignments

**[Product selection, verified references on 2026-10-28]** 本表是當前 19 語言的可審閱選擇；不是第二份 runtime allowlist。CEFR 適用於多語言的閱讀自評，本產品選用其級別名稱，不宣稱下表各語言有相同考試或認證規則。中文選 TOCFL Level 1–6；準備級不在第一版選單內，不代表 TOCFL 只有六個可測層級，也不限制簡繁書寫。日/韓使用其讀者熟悉的方案。另五語言明確採產品自評，不是假定該語言沒有正式測驗。

| Language code | 語言 | Scheme ID / levels | 選擇理由 / 依據 |
| --- | --- | --- | --- |
| en | 英語 | cefr_reading / A1→C2 | CEFR 通用閱讀自評框架，來源 C |
| zh | 中文 | tocfl_reading / 1→6 | 明確中文級別與閱讀描述，來源 T |
| ja | 日語 | jlpt_reading / N5→N1 | 保留 JLPT 級別和由易到難順序，來源 J |
| ko | 韓語 | topik_reading / 1→6 | 保留 TOPIK I/II 的級別標識，閱讀描述由產品整理，來源 K |
| es | 西班牙語 | cefr_reading / A1→C2 | 使用 C 的通用能力描述，不綁定特定國家考試 |
| pt | 葡萄牙語 | cefr_reading / A1→C2 | 使用 C；不從 code 推定國家或地域考試 |
| fr | 法語 | cefr_reading / A1→C2 | 使用 C 的閱讀自評，不換算考試分數 |
| de | 德語 | cefr_reading / A1→C2 | 使用 C 的閱讀自評，不換算考試分數 |
| it | 義大利語 | cefr_reading / A1→C2 | 使用 C 的閱讀自評，不綁定考試機構 |
| ru | 俄語 | cefr_reading / A1→C2 | 選用 C 的跨語言框架，不宣稱對應俄語考試等級 |
| ar | 阿拉伯語 | deep_reader_reading / beginner→proficient | 產品 P 的白話閱讀自評，不要求讀者知道特定地區考試 |
| hi | 印地語 | deep_reader_reading / beginner→proficient | 產品 P；不用不熟悉的考試編號作必要輸入 |
| tr | 土耳其語 | cefr_reading / A1→C2 | 選用 C 的跨語言框架，不宣稱官方考試等值 |
| nl | 荷蘭語 | cefr_reading / A1→C2 | 使用 C 的閱讀自評，不綁定特定考試 |
| pl | 波蘭語 | cefr_reading / A1→C2 | 使用 C 的閱讀自評，不換算考試分數 |
| uk | 烏克蘭語 | cefr_reading / A1→C2 | 選用 C 的跨語言框架，不宣稱官方考試等值 |
| id | 印尼語 | deep_reader_reading / beginner→proficient | 產品 P；描述閱讀能力，不換算當地考試 |
| vi | 越南語 | deep_reader_reading / beginner→proficient | 產品 P；直接用白話描述閱讀能力 |
| th | 泰語 | deep_reader_reading / beginner→proficient | 產品 P；直接用白話描述閱讀能力 |

C：[Council of Europe CEFR levels](https://www.coe.int/en/web/common-european-framework-reference-languages/level-descriptions) 與 [CEFR reading descriptors](https://rm.coe.int/1680459f97)。採 A1/A2/B1/B2/C1/C2 的能力遞進，不建立跨考試對照。

T：[TOCFL 官方閱讀能力描述](https://tocfl.edu.tw/tocfl/index.php/test/reading/list/3) 與 [級別名稱](https://tocfl.edu.tw/tocfl/index.php/sign_up/rule/list/1)。保留入門、基礎、進階、高階、流利、精通標識；不換算 HSK/CEFR。

J：[JLPT 官方各級閱讀概要](https://www.jlpt.jp/e/about/levelsummary.html)。保留 N5→N1；不把讀者自評當作通過完整 JLPT。

K：[NIIED TOPIK 官方格式與級別](https://www.niied.go.kr/web/NIIED/contents/niiedEng/eng_topikOverview)。只取 I/II 級別標識；產品閱讀描述不是官方各級通過條件，也不代表 speaking 或綜合考試得分。

P：產品自定義 beginner、elementary、intermediate、advanced、proficient，明確在 scheme label/scope_note 標示 Deep Reader。閱讀描述從辨識少量詞語，到理解連續文章/隱含意思遞進；高級仍可遇到生僻詞與專業術語，無官方認證/CEFR 等值主張。

### Labels, versions and reuse boundary

初版 label 與 reading_description 使用目前 UI 可用的簡明英語，TOCFL 另保留中文級名；全部描述是產品整理的閱讀摘要，可供讀者自評與未來 LLM 使用，沒有複製完整官方評分規準。尚未交付 19 語言的介面翻譯。未來顯示翻譯不可改持久化 ID，也不能讓 LLM 推測級別序號。

全部方案初版 version 為字串 `1`，是 Deep Reader 定義版本，不是外部考試年度。改 level ID/order/閱讀語義時必須提升該方案 version；不能把已保存舊版本默認解讀為新版。語言方案改選時 scheme ID 也參與校驗；由之後的偏好服務處理重新選擇/顯式 migration。不同語言即使共用 CEFR 級名也有不同 selection identity；沒有跨方案 rank 比較或 score conversion API。

### Validation evidence

`python -m unittest scripts.test_language_proficiency_registry -v`：18 tests pass，涵蓋所有 registry codes、全部有效級別、scheme/level/version/型別拒絕、immutable/JSON projection、N5→N1、registry 增減/unknown/重複、missing/invalid definitions、版本升級與 identity 隔離。既有 `test_language_script_registry.py`、`test_language_discourse_registry.py` 各自執行通過。沒有 prompt/parser/SQL/UI 改動；不需要模型或真書生成驗收，也不宣稱完整模式已可用。

## Learning Sentence Tailoring (2026-10-28)

**[Code-Confirmed; supports section_tasks-owned LLRN-04]** `learning_sentence_rules.py` 提供 immutable `SentenceLanguageRules(prefix_abbreviations)`，公開 `sentence_rules(exact_code)` 與 RULES_VERSION。支援語言唯一來源是 `LanguageProfileRegistry`；mapping 必須完整對應目前 codes（排除 unknown），registry 新增/移除/重複時明確拒絕不完整設定，沒有默認英文 fallback 或額外可選語言白名單。

資料為可審閱的 prefix abbreviation baseline，包含不同語言的稱謂/引用等常見縮寫；zh/ja/ko 明確使用空的 prefix set，共用標點處理仍由 section_tasks 負責。只接受精確 str code，不接受 unknown、別名、region tags 或推測語言。回傳 frozenset，mapping 不可變；資料改動需提升 RULES_VERSION，使句子 resource identity 改變。

不是完整辭典或新語言偵測器；同形縮寫在句尾仍有歧義。切分、source hash/spans、版本組合及無標點 fallback 由 [LLRN-04 設計](../section_tasks/module-detailed-design.md#implemented-multilingual-sentence-segmentation-llrn-04) 擁有。未修改既有 code/profile/discourse/script/proficiency registry 的語義。

證據：`scripts/test_learning_sentence_segmenter.py` 19 tests passed，含全部 19 語言預期句界與 code/coverage failure tests；與 LLRN-03 DTO 25 項及舊 boundary/content-block 31 項回歸一起驗證。LLRN-04 只在 section_tasks checklist 計一次；本節不新增語言任務編號。

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
| `language/language_proficiency_registry.py` | Registry-bound reading levels and strict selection/version validation | Pure LLRN-01 foundation; UI/API integration pending **[Code-Confirmed]** |
| `language/language_script_registry.py` | script system detection registry | `profile/parser_metadata_extractor` 消費 **[Code-Confirmed]** |
| `language/language_discourse_registry.py` | dialogue/discourse cue registry | 保守 speech hints policy **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 統一語言 code contract 與 alias normalize。 **[Code-Confirmed]**
2. 提供文件語言偵測流程（優先載入既有 artifacts，再 LLM）。 **[Code-Confirmed]**
3. 提供多語言 keyword/signal/script/discourse registry。 **[Code-Confirmed]**
4. 提供全部 registry 語言的閱讀自評分級、不可變順序與版本化選擇驗證。 **[Code-Confirmed: LLRN-01]**

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
