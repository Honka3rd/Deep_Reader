-- Explicit DESTRUCTIVE downgrade: stop preference writers, back up both tables,
-- then run in one transaction. Restore that backup after re-upgrade to retain data.
-- Never loaded by the forward migration glob. No CASCADE: future dependencies
-- must be explicitly downgraded first. Existing document/hierarchy data is untouched.
DROP TABLE IF EXISTS reading_book_preferences;
DROP TABLE IF EXISTS reading_workspace_preferences;
