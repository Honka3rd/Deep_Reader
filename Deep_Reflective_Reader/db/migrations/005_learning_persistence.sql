-- Explicit additive migration; apply in one transaction after 001..004.
-- Ownership uses documents.id, resolved by exact namespace/logical document name.
-- Payloads use the strict learning codec, version 1; never a whole-book snapshot.
CREATE TABLE IF NOT EXISTS learning_sentences (
    document_row_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    resource_id TEXT COLLATE "C" NOT NULL,
    schema_version SMALLINT NOT NULL DEFAULT 1 CHECK (schema_version = 1),
    payload JSONB NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    PRIMARY KEY (document_row_id, resource_id)
);
CREATE TABLE IF NOT EXISTS learning_cache_entries (
    document_row_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    cache_id TEXT COLLATE "C" NOT NULL,
    resource_id TEXT COLLATE "C" NOT NULL,
    revision BIGINT NOT NULL CHECK (revision >= 0),
    last_fence BIGINT NOT NULL CHECK (last_fence >= 0),
    result_revision BIGINT NOT NULL CHECK (result_revision >= 0),
    schema_version SMALLINT NOT NULL DEFAULT 1 CHECK (schema_version = 1),
    payload JSONB NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    PRIMARY KEY (document_row_id, cache_id),
    FOREIGN KEY (document_row_id, resource_id) REFERENCES learning_sentences(document_row_id, resource_id)
);
CREATE TABLE IF NOT EXISTS learning_operations (
    document_row_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    operation_id TEXT COLLATE "C" NOT NULL,
    cache_id TEXT COLLATE "C" NOT NULL,
    fence BIGINT NOT NULL CHECK (fence > 0),
    result_revision BIGINT CHECK (result_revision > 0),
    schema_version SMALLINT NOT NULL DEFAULT 1 CHECK (schema_version = 1),
    payload JSONB NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    PRIMARY KEY (document_row_id, operation_id),
    UNIQUE (document_row_id, cache_id, fence),
    UNIQUE (document_row_id, cache_id, result_revision),
    FOREIGN KEY (document_row_id, cache_id) REFERENCES learning_cache_entries(document_row_id, cache_id)
);
CREATE TABLE IF NOT EXISTS learning_commands (
    document_row_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    command_id TEXT COLLATE "C" NOT NULL,
    operation_id TEXT COLLATE "C",
    schema_version SMALLINT NOT NULL DEFAULT 1 CHECK (schema_version = 1),
    payload JSONB NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    PRIMARY KEY (document_row_id, command_id),
    FOREIGN KEY (document_row_id, operation_id) REFERENCES learning_operations(document_row_id, operation_id)
);
CREATE TABLE IF NOT EXISTS learning_checkpoints (
    document_row_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    operation_id TEXT COLLATE "C" NOT NULL,
    batch_index INTEGER NOT NULL CHECK (batch_index >= 0),
    schema_version SMALLINT NOT NULL DEFAULT 1 CHECK (schema_version = 1),
    payload JSONB NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    PRIMARY KEY (document_row_id, operation_id, batch_index),
    FOREIGN KEY (document_row_id, operation_id) REFERENCES learning_operations(document_row_id, operation_id)
);
