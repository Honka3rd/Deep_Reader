export type RequestStatus = "initial" | "loading" | "success" | "error" | "empty";

export interface TaskUnitMetadata {
  unit_id: string;
  title?: string | null;
  container_title?: string | null;
  source_section_ids?: string[];
  is_fallback_generated?: boolean;
}

export interface SectionLayout {
  section_id: string;
  title?: string | null;
  container_title?: string | null;
  task_units?: TaskUnitMetadata[];
  anchor_evidence?: AnchorEvidence | null;
}

export interface ChapterLayout {
  chapter_id: string;
  title?: string | null;
  sections?: SectionLayout[];
  anchor_evidence?: AnchorEvidence | null;
}

export interface DocumentTaskLayout {
  document_id?: string;
  title?: string | null;
  chapters?: ChapterLayout[];
  parse_provenance?: {
    requested_parser_mode?: string | null;
    effective_parser_mode?: string | null;
    fallback_used?: boolean;
    fallback_reason?: string | null;
    source?: string | null;
  } | null;
}

export type StructureParserMode = "common" | "llm_enhanced";
export type ManualStructureAnchorType = "char_range" | "page_range";

export interface AnchorEvidence {
  anchor_type: ManualStructureAnchorType;
  status: string;
  reason?: string | null;
  char_start?: number | null;
  char_end?: number | null;
  page_start_index?: number | null;
  page_end_index?: number | null;
  page_start_label?: string | null;
  page_end_label?: string | null;
}

export interface ManualStructureAnchorRequest {
  anchor_type: ManualStructureAnchorType;
  char_start?: number | null;
  char_end?: number | null;
  page_start_index?: number | null;
  page_end_index?: number | null;
}

export interface ManualStructureEntryRequest {
  title: string;
  level: 1 | 2;
  anchor: ManualStructureAnchorRequest;
  external_id?: string | null;
  notes?: string | null;
}

export interface ManualStructurePlanRequest {
  entries: ManualStructureEntryRequest[];
  source_hash?: string | null;
}

export interface ManualStructureValidationRequest {
  doc_name: string;
  manual_structure: ManualStructurePlanRequest;
}

export interface ManualStructureValidationIssue {
  code: string;
  message: string;
  severity: "error" | "warning" | "info";
  entry_index?: number | null;
  external_id?: string | null;
}

export interface ManualStructurePreviewSection {
  title: string;
  external_id?: string | null;
  entry_index?: number | null;
}

export interface ManualStructurePreviewChapter {
  title: string;
  external_id?: string | null;
  entry_index?: number | null;
  sections: ManualStructurePreviewSection[];
}

export interface ManualStructureValidationResponse {
  doc_name: string;
  valid: boolean;
  normalized_entries: Array<
    ManualStructureEntryRequest & {
      projected_char_start?: number | null;
      projected_char_end?: number | null;
      projected_page_start_index?: number | null;
      projected_page_end_index?: number | null;
    }
  >;
  errors: ManualStructureValidationIssue[];
  warnings: ManualStructureValidationIssue[];
  preview_chapters: ManualStructurePreviewChapter[];
  parse_provenance_preview?: {
    parser_mode: "manual_structure";
    source_hash?: string | null;
    anchor_types: ManualStructureAnchorType[];
  } | null;
}

export interface ReparseDocumentStructureResponse {
  success: boolean;
  doc_name: string;
  parser_mode: StructureParserMode | string;
  structured_document_path?: string | null;
  error?: string | null;
  section_count?: number | null;
}

export interface PrepareDocumentResponse {
  doc_name: string;
  mode: string;
  structured_parser_mode: StructureParserMode | string;
  success: boolean;
  structured_document_ready: boolean;
  structured_document_path?: string | null;
  faiss_ready: boolean;
  profile_ready: boolean;
  bundle_ready: boolean;
  errors: string[];
}

export interface DocumentListItem {
  doc_name: string;
  title?: string | null;
  source: string;
}

export interface DocumentListResponse {
  items: DocumentListItem[];
  query?: string | null;
  total: number;
}

export interface ContentBlock {
  block_id: string;
  content: string;
  block_type?: string | null;
  metadata?: Record<string, unknown>;
}

export interface TaskUnitContent {
  document_id?: string;
  document_title?: string | null;
  task_unit_id: string;
  title?: string | null;
  container_title?: string | null;
  content_blocks?: ContentBlock[];
  chapter_title?: string | null;
  section_title?: string | null;
}

export interface BatchTaskUnitContentResponse {
  document_id?: string;
  document_title?: string | null;
  contents: TaskUnitContent[];
}

export interface SectionSelection {
  chapter: ChapterLayout;
  section: SectionLayout;
  taskUnits: TaskUnitMetadata[];
}

export type ReadingInteractionTargetType =
  | "document"
  | "chapter"
  | "section"
  | "task_unit";

export interface ReadingInteractionTargetRequest {
  doc_name: string;
  target_type: ReadingInteractionTargetType;
  chapter_id?: string | null;
  section_id?: string | null;
  task_unit_id?: string | null;
  source_structure_version?: number | null;
  source_hash?: string | null;
}

export interface ReadingInteractionTargetResponse {
  doc_name: string;
  target_type: ReadingInteractionTargetType;
  target_id: string;
  document_id?: string | null;
  document_title?: string | null;
  chapter_id?: string | null;
  section_id?: string | null;
  task_unit_id?: string | null;
  title?: string | null;
}

export interface ArtifactAwareInteractionMetadataResponse {
  artifact_context_mode?: string | null;
  primary_source_evidence_ids?: string[];
  referenced_artifact_ids?: string[];
  referenced_artifact_types?: string[];
  referenced_artifact_target_levels?: ReadingInteractionTargetType[];
  coverage_counts?: Record<string, number>;
  deduplication_hint_applied?: boolean;
  abstraction_hint_applied?: boolean;
  artifact_context_pruned_reason?: string | null;
}

export interface ReadingInteractionResponseEnvelope {
  target: ReadingInteractionTargetResponse;
  interaction_type: string;
  status: string;
  artifact_id?: string | null;
  session_id?: string | null;
  generated_at?: string | null;
  updated_at?: string | null;
  schema_version?: string | null;
  prompt_instruction_version?: string | null;
  source_structure_version?: number | null;
  source_hash?: string | null;
  reason?: string | null;
  artifact_context_metadata?: ArtifactAwareInteractionMetadataResponse | null;
}

export interface AnalysisArtifactPayloadResponse {
  summary: string;
  reasoning: string;
  interpretation: string;
  explanation?: string | null;
  key_points?: string[];
}

export interface AnalysisInteractionResponse {
  envelope: ReadingInteractionResponseEnvelope;
  payload?: AnalysisArtifactPayloadResponse | null;
}
