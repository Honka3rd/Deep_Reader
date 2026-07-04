PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_document_id TEXT,
    title TEXT NOT NULL,
    source_path TEXT,
    language TEXT,
    raw_text TEXT NOT NULL DEFAULT '',
    current_structure_version INTEGER NOT NULL CHECK (current_structure_version >= 1),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_source_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL UNIQUE REFERENCES documents(id) ON DELETE CASCADE,
    source_location TEXT,
    original_filename TEXT,
    mime_type TEXT,
    file_size INTEGER,
    checksum TEXT,
    uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS parse_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL CHECK (event_type IN ('initial_parse', 'hard_reparse')),
    previous_structure_version INTEGER,
    new_structure_version INTEGER NOT NULL,
    parser_name TEXT,
    preparation_mode TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chapters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chapter_order INTEGER NOT NULL,
    reference_chapter_id TEXT,
    title TEXT,
    level INTEGER NOT NULL,
    chapter_role TEXT,
    metadata_json TEXT,
    UNIQUE(document_id, chapter_order)
);

CREATE TABLE IF NOT EXISTS sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chapter_id INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    section_order INTEGER NOT NULL,
    reference_section_id TEXT,
    title TEXT,
    level INTEGER NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    char_start INTEGER NOT NULL DEFAULT 0,
    char_end INTEGER NOT NULL DEFAULT 0,
    container_title TEXT,
    section_role TEXT,
    section_kind TEXT,
    is_implicit_section INTEGER NOT NULL DEFAULT 0,
    UNIQUE(chapter_id, section_order)
);

CREATE TABLE IF NOT EXISTS task_units (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    section_id INTEGER NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
    task_unit_order INTEGER NOT NULL,
    reference_unit_id TEXT,
    title TEXT,
    container_title TEXT,
    content TEXT NOT NULL DEFAULT '',
    source_section_ids_json TEXT NOT NULL DEFAULT '[]',
    is_fallback_generated INTEGER NOT NULL DEFAULT 0,
    UNIQUE(section_id, task_unit_order)
);

CREATE INDEX IF NOT EXISTS idx_raw_source_metadata_document_id
    ON raw_source_metadata(document_id);

CREATE INDEX IF NOT EXISTS idx_parse_events_document_id_created_at
    ON parse_events(document_id, created_at);

CREATE INDEX IF NOT EXISTS idx_chapters_document_id_order
    ON chapters(document_id, chapter_order);

CREATE INDEX IF NOT EXISTS idx_sections_document_id_chapter_id_order
    ON sections(document_id, chapter_id, section_order);

CREATE INDEX IF NOT EXISTS idx_task_units_document_id_section_id_order
    ON task_units(document_id, section_id, task_unit_order);
