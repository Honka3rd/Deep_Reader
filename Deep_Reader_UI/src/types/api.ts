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
}

export interface ChapterLayout {
  chapter_id: string;
  title?: string | null;
  sections?: SectionLayout[];
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

export interface SectionSelection {
  chapter: ChapterLayout;
  section: SectionLayout;
  taskUnits: TaskUnitMetadata[];
}
