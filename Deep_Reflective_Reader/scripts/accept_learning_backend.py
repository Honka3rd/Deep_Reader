"""LLRN-20 reproducible offline-provider acceptance on disposable file and SQL storage.

DEEP_READER_TEST_POSTGRES_DSN must point to an isolated test database. Migrations
run only inside random schemas owned and removed by the existing test fixtures.
No default app DSN, user book, API key or live provider is consulted by this runner.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from time import monotonic
import unittest

import psycopg


INTEGRATION = 'scripts.test_learning_backend_integration'
REGRESSIONS = (
    'test_language_proficiency_registry', 'test_reading_preferences',
    'test_reading_preferences_postgres', 'test_reading_preferences_service',
    'test_learning_contracts', 'test_learning_sentence_segmenter',
    'test_learning_generation_policy', 'test_learning_file_repository',
    'test_learning_postgres_repository', 'test_learning_translation_service',
    'test_learning_hints_service', 'test_learning_lifecycle',
    'test_learning_orchestration', 'test_learning_api_schemas',
    'test_learning_provider_once', 'test_learning_container_wiring', 'test_learning_routes',
    'test_learning_live_recording',
    'test_translation_service', 'test_translation_action',
    'test_translation_persistence', 'test_translation_persistence_postgres',
)


def cases(suite):
    for case in suite:
        if isinstance(case, unittest.TestSuite):
            yield from cases(case)
        else:
            yield case


def schemas(dsn):
    with psycopg.connect(dsn) as conn:
        return {r[0] for r in conn.execute('SELECT schema_name FROM information_schema.schemata')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--integration-only', action='store_true', help='Diagnostic subset; does not pass the full gate.')
    args = parser.parse_args(argv)
    started = monotonic()
    report = {'task': 'LLRN-20', 'created_at': datetime.now(timezone.utc).isoformat(),
              'status': 'Needs Confirmation', 'live_provider': False, 'semantic_quality': 'not assessed',
              'scope': 'integration-only' if args.integration_only else 'full',
              'cleanup_verified': False}
    dsn = os.environ.get('DEEP_READER_TEST_POSTGRES_DSN')
    code = 1
    try:
        if not dsn:
            raise RuntimeError('DEEP_READER_TEST_POSTGRES_DSN is required; no default database fallback')
        before = schemas(dsn)
        names = [INTEGRATION] + ([] if args.integration_only else ['scripts.'+n for n in REGRESSIONS])
        loader = unittest.TestLoader()
        unique = {}
        for name in names:
            for case in cases(loader.loadTestsFromName(name)):
                unique.setdefault(case.id(), case)
        # Imported TestCase classes can appear in multiple modules; execute each ID once.
        ids = list(unique)
        root = Path(__file__).resolve().parents[1]
        paths = {root / (name.replace('.', '/')+'.py') for name in names}
        report['suite_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(paths) if p.is_file()}
        result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(unique.values()))
        after = schemas(dsn)
        report['cleanup_verified'] = before == after
        report['remaining_new_schemas'] = sorted(after-before)
        from scripts.test_learning_backend_integration import EVIDENCE
        report['journeys'] = EVIDENCE
        report['tests'] = {'run': result.testsRun, 'ids': ids,
                           'failures': [t.id() for t, _ in result.failures],
                           'errors': [t.id() for t, _ in result.errors],
                           'skipped': [t.id() for t, _ in result.skipped],
                           'expected_failures': [t.id() for t, _ in result.expectedFailures]}
        ok = (result.wasSuccessful() and not result.skipped and not result.expectedFailures
              and not loader.errors and report['cleanup_verified'] and result.testsRun == len(ids))
        report['status'] = ('subset passed' if args.integration_only else 'passed') if ok else 'failed'
        code = 0 if ok else 1
    except Exception as exc:
        # Do not serialize exception messages: driver errors may contain connection details.
        report['prerequisite_error_type'] = type(exc).__name__
        print('Acceptance incomplete; verify the explicit disposable database and test environment.', file=sys.stderr)
    finally:
        report['duration_seconds'] = round(monotonic()-started, 3)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(f"LLRN-20: {report['status']}; report: {args.report}")
    return code


if __name__ == '__main__':
    raise SystemExit(main())
