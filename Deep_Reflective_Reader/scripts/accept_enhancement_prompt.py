"""Offline plus disposable PostgreSQL acceptance; never calls a live model.

Run with DEEP_READER_TEST_POSTGRES_DSN pointing to a local test-capable database.
The PostgreSQL harness creates and removes a unique schema, not user documents.
"""
import json
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if not os.environ.get("DEEP_READER_TEST_POSTGRES_DSN"):
    raise SystemExit("DEEP_READER_TEST_POSTGRES_DSN required; integration acceptance cannot be skipped")
if os.environ.get("ENHANCEMENT_SOURCE_DOCUMENT"):
    raise SystemExit("Unset ENHANCEMENT_SOURCE_DOCUMENT: this acceptance uses synthetic fixtures only")

modules = ["test_plan_prompt_metrics", "test_plan_prompt_budget", "test_plan_rejection",
           "test_structure_plan_contract", "test_structure_plan_application",
           "test_structure_enhancement_candidate", "test_enhancement_postgres_preflight",
           "test_llm_capacity_contract", "test_full_source_enhancement",
           "test_enhancement_capacity_preflight", "test_enhancement_preflight"]
suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
result = unittest.TextTestRunner(verbosity=1).run(suite)
unexpected_skips = [reason for test, reason in result.skipped
                    if "test_opt_in_source_read_only_copy_to_disposable_namespace" not in test.id()]
print(json.dumps({"tests_run": result.testsRun, "failures": len(result.failures),
                  "errors": len(result.errors), "skips": len(result.skipped),
                  "live_llm_evaluated": False, "production_deployment": False}))
sys.exit(0 if result.wasSuccessful() and not unexpected_skips else 1)
