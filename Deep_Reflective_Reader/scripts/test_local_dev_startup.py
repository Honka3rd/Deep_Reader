"""Exercise the real launcher with isolated env and command stubs, never real keys."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class LocalDevStartupTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "repo with spaces"
        (self.root / "scripts").mkdir(parents=True)
        (self.root / "Deep_Reader_UI").mkdir()
        migration = self.root / "Deep_Reflective_Reader/db/migrations"
        migration.mkdir(parents=True)
        (migration / "003_interaction_version_guards.sql").write_text("SELECT 1;\n")
        for name in ("start_local_dev.sh", "load_local_env.sh"):
            shutil.copyfile(ROOT / "scripts" / name, self.root / "scripts" / name)
        shutil.copyfile(ROOT / "docker-compose.yml", self.root / "docker-compose.yml")
        self.log = self.root / "commands.jsonl"
        binaries = self.root / "bin"
        binaries.mkdir()
        import sys
        stub = f"#!{sys.executable}\n" + """
import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
if name == 'lsof':
    sys.exit(0 if os.environ.get('OCCUPIED') == '1' else 1)
with open(os.environ['COMMAND_LOG'], 'a') as out:
    out.write(json.dumps({'name': name, 'args': sys.argv[1:],
        'proxy': os.environ.get('DEEP_READER_CONTAINER_PROXY'),
        'key': os.environ.get('OPENAI_API_KEY'), 'cwd': os.getcwd()}) + '\\n')
if name == 'docker' and '--wait' in sys.argv and sys.argv[-1] == 'api':
    sys.exit(int(os.environ.get('API_EXIT', '0')))
if name == 'docker' and '--wait' in sys.argv and sys.argv[-1] == 'postgres':
    sys.exit(int(os.environ.get('POSTGRES_EXIT', '0')))
if name == 'docker' and 'exec' in sys.argv and 'postgres' in sys.argv:
    sys.exit(int(os.environ.get('MIGRATION_EXIT', '0')))
if name == 'docker' and sys.argv[-1] == 'scripts.check_llm_connectivity':
    sys.exit(int(os.environ.get('CONNECTIVITY_EXIT', '0')))
if name == 'npm':
    sys.exit(int(os.environ.get('NPM_EXIT', '0')))
"""
        for name in ("docker", "npm", "lsof"):
            path = binaries / name
            path.write_text(stub)
            path.chmod(0o700)
        self.env = {**os.environ, "PATH": f"{binaries}:/usr/bin:/bin", "COMMAND_LOG": str(self.log),
                    "DEEP_READER_CONTAINER_PROXY": "http://stale.invalid:1234"}
        for name in ("OPENAI_API_KEY", "DEEP_READER_LLM_API_KEY"):
            self.env.pop(name, None)

    def run_start(self, *args, proxy="http://host.docker.internal:7897"):
        (self.root / ".env.local").write_text(
            "DEEP_READER_UI_PORT=5173\nDEEP_READER_API_PORT=8000\n"
            "DEEP_READER_POSTGRES_PORT=5433\nDEEP_READER_LLM_API_KEY=fake-test-key\n"
            f"DEEP_READER_CONTAINER_PROXY={proxy}\n")
        return subprocess.run(["bash", str(self.root / "scripts/start_local_dev.sh"), *args],
                              env=self.env, cwd="/tmp", text=True, capture_output=True, timeout=10)

    def commands(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_default_loads_proxy_and_key_then_starts_ui_after_healthy_api(self):
        result = self.run_start()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands()
        self.assertEqual([c["name"] for c in commands], ["docker"] * 4 + ["npm"])
        self.assertEqual(commands[0]["args"][-4:], ["up", "-d", "--wait", "postgres"])
        self.assertEqual(commands[2]["args"][-5:], ["up", "-d", "--build", "--wait", "api"])
        self.assertEqual(commands[3]["args"][-6:], ["exec", "-T", "api", "python", "-m", "scripts.check_llm_connectivity"])
        self.assertTrue(all(c["proxy"] == "http://host.docker.internal:7897" for c in commands))
        self.assertTrue(all(c["key"] == "fake-test-key" for c in commands))
        self.assertEqual(Path(commands[-1]["cwd"]).resolve(), (self.root / "Deep_Reader_UI").resolve())
        self.assertNotIn("fake-test-key", result.stdout + result.stderr)
        self.assertNotIn("host.docker.internal", result.stdout + result.stderr)

    def test_backend_only_does_not_touch_occupied_ui(self):
        self.env["OCCUPIED"] = "1"
        result = self.run_start("--backend-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([c["name"] for c in self.commands()], ["docker"] * 4)

    def test_empty_proxy_overrides_stale_shell_setting(self):
        result = self.run_start("--backend-only", proxy="")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(c["proxy"] == "" for c in self.commands()))
        self.assertIn("direct connection", result.stdout)

    def test_api_failure_never_starts_ui(self):
        self.env["API_EXIT"] = "7"
        result = self.run_start()
        self.assertEqual(result.returncode, 7)
        self.assertNotIn("npm", [c["name"] for c in self.commands()])

    def test_connectivity_failure_warns_and_starts_ui_without_route_fallback(self):
        for proxy in ("http://host.docker.internal:7897", ""):
            with self.subTest(proxy=proxy):
                self.log.unlink(missing_ok=True)
                self.env["CONNECTIVITY_EXIT"] = "1"
                result = self.run_start(proxy=proxy)
                self.assertEqual(result.returncode, 0, result.stderr)
                commands = self.commands()
                self.assertEqual([c["name"] for c in commands], ["docker"] * 4 + ["npm"])
                self.assertEqual(commands[3]["args"][-1], "scripts.check_llm_connectivity")
                self.assertTrue(all(c["proxy"] == proxy for c in commands))
                self.assertIn("Warning: Model API connectivity could not be confirmed", result.stderr)
                self.assertIn("model-dependent actions may fail", result.stderr)
                self.assertIn("no route fallback was attempted", result.stderr)
                self.assertIn("Starting UI", result.stdout)

    def test_backend_only_warns_without_touching_existing_ui(self):
        self.env["CONNECTIVITY_EXIT"] = "1"
        self.env["OCCUPIED"] = "1"
        result = self.run_start("--backend-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.commands()), 4)
        self.assertIn("Warning:", result.stderr)
        self.assertIn("Backend ready; existing UI processes were left unchanged", result.stdout)

    def test_postgres_failure_stops_before_migration_api_and_ui(self):
        self.env["POSTGRES_EXIT"] = "8"
        result = self.run_start()
        self.assertEqual(result.returncode, 8)
        self.assertEqual(len(self.commands()), 1)

    def test_migration_failure_stops_before_api_and_ui(self):
        self.env["MIGRATION_EXIT"] = "9"
        result = self.run_start()
        self.assertEqual(result.returncode, 9)
        self.assertEqual(len(self.commands()), 2)

    def test_ui_failure_is_not_hidden_by_connectivity_warning(self):
        self.env["CONNECTIVITY_EXIT"] = "1"
        self.env["NPM_EXIT"] = "10"
        result = self.run_start()
        self.assertEqual(result.returncode, 10)
        self.assertEqual(self.commands()[-1]["name"], "npm")

    def test_occupied_ui_is_not_killed_or_silently_replaced(self):
        self.env["OCCUPIED"] = "1"
        self.assertEqual(self.run_start().returncode, 1)
        self.assertEqual(self.commands(), [])

    def test_unknown_argument_fails_before_startup(self):
        self.assertEqual(self.run_start("--typo").returncode, 2)
        self.assertEqual(self.commands(), [])


if __name__ == "__main__":
    unittest.main()
