-- Prevent delayed relational writers from recreating stale interactions after reparse.
CREATE OR REPLACE FUNCTION enforce_current_interaction_version()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE current_version integer;
BEGIN
    SELECT current_structure_version INTO current_version
      FROM documents WHERE id = NEW.document_id FOR KEY SHARE;
    IF current_version IS NULL OR NEW.source_structure_version <> current_version THEN
        RAISE EXCEPTION 'stale_interaction_write' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS artifacts_current_version ON artifacts;
CREATE TRIGGER artifacts_current_version
BEFORE INSERT OR UPDATE ON artifacts
FOR EACH ROW EXECUTE FUNCTION enforce_current_interaction_version();

DROP TRIGGER IF EXISTS content_blocks_current_version ON content_blocks;
CREATE TRIGGER content_blocks_current_version
BEFORE INSERT OR UPDATE ON content_blocks
FOR EACH ROW EXECUTE FUNCTION enforce_current_interaction_version();
