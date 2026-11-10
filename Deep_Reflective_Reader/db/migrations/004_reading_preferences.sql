-- Explicit additive migration. Run in one transaction after 001..003.
-- Runtime construction/reads never execute this migration.
-- Logical document identities are exact, namespace-scoped names, not SQL row IDs.
-- No document FK: defaults may precede a document and tombstones must outlive it.
CREATE TABLE IF NOT EXISTS reading_workspace_preferences (
    namespace TEXT COLLATE "C" PRIMARY KEY CHECK (namespace <> ''),
    scope_kind TEXT NOT NULL DEFAULT 'workspace' CHECK (scope_kind = 'workspace'),
    schema_version SMALLINT NOT NULL DEFAULT 1 CHECK (schema_version = 1),
    revision BIGINT NOT NULL DEFAULT 0 CHECK (revision >= 0),
    native_language TEXT,
    receipt JSONB CHECK (receipt IS NULL OR jsonb_typeof(receipt) = 'object')
);

CREATE TABLE IF NOT EXISTS reading_book_preferences (
    namespace TEXT COLLATE "C" NOT NULL REFERENCES reading_workspace_preferences(namespace),
    document_id TEXT COLLATE "C" NOT NULL CHECK (document_id <> ''),
    schema_version SMALLINT NOT NULL DEFAULT 1 CHECK (schema_version = 1),
    revision BIGINT NOT NULL CHECK (revision >= 0),
    settings JSONB NOT NULL CHECK (jsonb_typeof(settings) = 'object'),
    deleted BOOLEAN NOT NULL,
    receipt JSONB CHECK (receipt IS NULL OR jsonb_typeof(receipt) = 'object'),
    PRIMARY KEY (namespace, document_id),
    CHECK (NOT deleted OR settings = '{"active_mode":"normal","source_language_override":null,"modes":{}}'::jsonb)
);
