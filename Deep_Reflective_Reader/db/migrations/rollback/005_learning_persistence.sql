-- DESTRUCTIVE explicit downgrade. Stop learning writers and back up all five
-- tables first. Reapply 005 and restore sentences -> entries -> operations ->
-- commands/checkpoints to retain paid results. No CASCADE; future dependencies
-- require their own downgrade first. Execute in a single transaction.
DROP TABLE IF EXISTS learning_checkpoints;
DROP TABLE IF EXISTS learning_commands;
DROP TABLE IF EXISTS learning_operations;
DROP TABLE IF EXISTS learning_cache_entries;
DROP TABLE IF EXISTS learning_sentences;
