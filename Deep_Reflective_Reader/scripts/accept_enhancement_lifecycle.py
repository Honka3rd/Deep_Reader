"""ENH-REV-06 acceptance: synthetic tests only, fail closed on missing integration.

Requires local test PostgreSQL, running UI_URL and installed Playwright/Chromium.
Creates only disposable namespaces. Never reads credentials or invokes a live LLM.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


def run():
    if not os.environ.get("DEEP_READER_TEST_POSTGRES_DSN"):
        raise SystemExit("Explicit disposable PostgreSQL DSN required")
    if os.environ.get("ENHANCEMENT_SOURCE_DOCUMENT"):
        raise SystemExit("Unset ENHANCEMENT_SOURCE_DOCUMENT; synthetic acceptance only")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    modules = ["test_enhancement_change", "test_enhancement_autonomy",
        "test_semantic_structure_plan", "test_enhancement_plan_repair",
        "test_enhancement_outcome_api", "test_enhancement_operations",
        "test_enhancement_capacity_preflight", "test_full_source_enhancement",
        "test_structure_plan_contract", "test_structure_plan_application",
        "test_task_layout_readiness"]
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    unexpected_skips = [test.id() for test, _ in result.skipped
        if not test.id().endswith("test_opt_in_source_read_only_copy_to_disposable_namespace")]
    if not result.wasSuccessful() or unexpected_skips:
        raise SystemExit(1)
    ui = root.parent / "Deep_Reader_UI"
    subprocess.run(["npm", "run", "check"], cwd=ui, check=True)
    for name in ("test-enhancement-confirmation.mjs", "test-enhancement-outcomes.mjs", "test-reader-guards.mjs"):
        subprocess.run(["node", str(ui / "scripts" / name)], cwd=ui, check=True, timeout=180)
    subprocess.run([sys.executable, str(root / "scripts/accept_enhancement_browser.py")], cwd=root, check=True, timeout=300)
    print(json.dumps({"engineering_acceptance": "passed", "backend_tests": result.testsRun,
        "skipped_private_source": len(result.skipped), "viewports": [1280, 390],
        "live_model_usefulness": "not_evaluated", "production_deployment": False}))


if __name__ == "__main__":
    run()
