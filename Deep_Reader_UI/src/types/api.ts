export type RequestStatus = "initial" | "loading" | "success" | "error" | "empty";
export type TocHint = "auto" | "present" | "absent";
export type EnhancementIntent = "restore_source" | "generate_reading_outline";

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
  is_implicit_section?: boolean;
  title_origin?: "source" | "implicit" | "generated" | null;
}

export interface ChapterLayout {
  chapter_id: string;
  title?: string | null;
  sections?: SectionLayout[];
  anchor_evidence?: AnchorEvidence | null;
  metadata?: { title_origin?: "source" | "implicit" | "generated"; [key: string]: unknown };
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
    toc_hint?: TocHint | null;
    toc_observation?: string | null;
    enhancement_intent?: EnhancementIntent | null;
    implicit_single_body?: boolean | null;
    declaration_unconfirmed?: boolean | null;
  } | null;
}

export type StructureParserMode = "common" | "llm_enhanced";
export type PdfExtractionMode = "fast" | "ocr" | "auto";
export interface PdfExtractionMetadata {
  requested_mode: PdfExtractionMode;
  effective_mode: PdfExtractionMode | null;
  provenance: "unknown" | "not_applicable" | "applied" | "reused";
}
export type PreparedTaskLayout = DocumentTaskLayout & { pdf_extraction?: PdfExtractionMetadata };
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
  structure_version?: number | null;
  change_kind?: "unchanged" | "titles_only" | "structure_changed" | null;
  committed?: boolean | null;
  interactions_cleared?: boolean | null;
  reading_layout_required?: boolean | null;
  retryable?: boolean;
}

export interface EnhancementOperationResponse {
  status: "not_started" | "running" | "completed" | "failed" | "unknown";
  result: ReparseDocumentStructureResponse | null;
  reason: string | null;
  retryable: boolean;
  durability: "process_local";
}

export interface EnhancementCapacity {
  status: "fits" | "exceeded" | "unavailable" | "resource_limited";
  reason_code?: string | null;
  model?: string | null;
  required_input_tokens?: number | null;
  available_input_tokens?: number | null;
  output_reserve_tokens?: number | null;
  max_output_tokens?: number | null;
  count_method?: "tokenizer" | "conservative_estimate" | "unavailable";
  full_source?: boolean | null;
}

export interface EnhancementPreflightResponse {
  capacity?: EnhancementCapacity | null;
  ready: boolean;
  execution_available: boolean;
  reason: string | null;
  structure_version: number | null;
  interaction_count: number | null;
  includes_user_notes: boolean | null;
  cleanup_scope: string;
  cleanup_condition?: "structure_changed_only";
  confirmation_token: string | null;
  expires_at: number | null;
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

export type DocumentManagementSourceType = "txt" | "pdf" | "unknown";
export type DocumentManagementSourceStatus =
  | "unknown"
  | "available"
  | "missing"
  | "unsupported"
  | "failed";
export type DocumentManagementPreparationStatus =
  | "not_prepared"
  | "ready"
  | "failed"
  | "stale"
  | "unknown";

export interface DocumentManagementItem {
  reviewRequired?: boolean;
  docName: string;
  title?: string | null;
  author?: string | null;
  language?: string | null;
  profileLabel?: string | null;
  profileSummary?: string | null;
  source?: string | null;
  sourceFilename?: string | null;
  sourceType?: DocumentManagementSourceType;
  sourceSizeBytes?: number | null;
  sourceHash?: string | null;
  sourceStatus?: DocumentManagementSourceStatus;
  preparationStatus?: DocumentManagementPreparationStatus;
  uploadedAt?: string | null;
  updatedAt?: string | null;
  lastReadAt?: string | null;
  lastError?: string | null;
}

export interface DocumentManagementListResult {
  items: DocumentManagementItem[];
  query?: string | null;
  total: number;
}

export type DocumentUploadConflictPolicy = "reject" | "replace" | "create_distinct";
export type DocumentUploadSourceStatus = "uploaded" | "failed";
export type DocumentUploadFailureReason =
  | "missing_source"
  | "unsupported_file_type"
  | "file_too_large"
  | "duplicate_conflict"
  | "unsafe_filename"
  | "validation_failure"
  | "storage_failure";

export interface DocumentUploadResponse {
  doc_name: string;
  title?: string | null;
  source_filename: string;
  source_type: DocumentManagementSourceType;
  source_size_bytes: number;
  source_hash?: string | null;
  uploaded_at?: string | null;
  source_status: DocumentUploadSourceStatus;
  preparation_status: DocumentManagementPreparationStatus;
  conflict_policy?: DocumentUploadConflictPolicy | null;
  reason?: DocumentUploadFailureReason | string | null;
}

export type DocumentLifecycleAction =
  | "archive"
  | "restore"
  | "delete"
  | "rename";

export type DocumentLifecycleStatus =
  | "accepted"
  | "rejected"
  | "conflict"
  | "failed";

export type DocumentLifecycleFailureReason =
  | "missing_document"
  | "unsupported_action"
  | "conflict"
  | "validation_failure"
  | "storage_failure"
  | "unsafe_name"
  | "not_implemented";

export interface DocumentLifecycleActionRequest {
  doc_name: string;
  action: DocumentLifecycleAction;
  new_doc_name?: string | null;
  expected_source_hash?: string | null;
  expected_structure_version?: string | null;
  confirm?: boolean;
}

export interface DocumentLifecycleActionResponse {
  doc_name: string;
  action: DocumentLifecycleAction;
  status: DocumentLifecycleStatus;
  new_doc_name?: string | null;
  reason?: DocumentLifecycleFailureReason | string | null;
  source_affected?: boolean;
  structured_artifacts_affected?: boolean;
  derived_resources_affected?: boolean;
  archived?: boolean | null;
  deleted?: boolean | null;
}

export interface ContentBlock {
  block_id: string;
  content: string;
  block_type?: string | null;
  metadata?: Record<string, unknown>;
  artifact_target_refs?: ContentBlockArtifactTargetRef[] | null;
  interaction_availability?: ContentBlockInteractionAvailability | null;
}

export interface ContentBlockArtifactTargetRef {
  target_level?: ReadingInteractionTargetType | string | null;
  target_id?: string | null;
  task_unit_id?: string | null;
  content_block_id?: string | null;
  metadata?: Record<string, unknown> | null;
}

export interface ContentBlockInteractionAvailabilitySlot {
  interaction_type: string;
  status: string;
  artifact_id?: string | null;
  session_id?: string | null;
  reason?: string | null;
  generated_at?: string | null;
  updated_at?: string | null;
}

export interface ContentBlockInteractionAvailability {
  insight?: ContentBlockInteractionAvailabilitySlot | null;
  quiz?: ContentBlockInteractionAvailabilitySlot | null;
  critical_thinking?: ContentBlockInteractionAvailabilitySlot | null;
}

export interface TaskUnitContent {
  document_id?: string;
  document_title?: string | null;
  chapter_id?: string | null;
  section_id?: string | null;
  task_unit_id: string;
  title?: string | null;
  container_title?: string | null;
  content?: string | null;
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
  | "task_unit"
  | "content_block";

export interface ReadingInteractionTargetRequest {
  doc_name: string;
  target_type: ReadingInteractionTargetType;
  chapter_id?: string | null;
  section_id?: string | null;
  task_unit_id?: string | null;
  content_block_id?: string | null;
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
  content_block_id?: string | null;
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

export type QuizArtifactItemType =
  | "short_answer"
  | "multiple_choice"
  | "true_false";

export interface QuizArtifactItemResponse {
  item_id: string;
  item_type: QuizArtifactItemType;
  prompt: string;
  options?: string[] | null;
  answer: string | boolean;
  explanation?: string | null;
}

export interface QuizArtifactPayloadResponse {
  items: QuizArtifactItemResponse[];
  max_items?: number | null;
}

export interface QuizInteractionResponse {
  envelope: ReadingInteractionResponseEnvelope;
  payload?: QuizArtifactPayloadResponse | null;
}

export interface CriticalThinkingEvaluationResponse {
  feedback: string;
  score?: number | null;
  suggested_refinement?: string | null;
  strengths?: string | null;
  improvements?: string | null;
}

export interface CriticalThinkingSessionPayloadResponse {
  question?: string | null;
  submitted_answer?: string | null;
  evaluation?: CriticalThinkingEvaluationResponse | null;
  retry_eligible: boolean;
}

export interface CriticalThinkingSessionResponse {
  envelope: ReadingInteractionResponseEnvelope;
  payload?: CriticalThinkingSessionPayloadResponse | null;
}
