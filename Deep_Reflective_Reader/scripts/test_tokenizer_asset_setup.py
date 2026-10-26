import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from llm.offline_token_counter import EncodingAsset, OfflineTokenCounter, encoding_asset_for_model
from scripts import prepare_tokenizer_assets as setup


BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent


class TokenizerAssetSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.data = b"verified encoding fixture"
        self.asset = EncodingAsset("test", "test.tiktoken", "https://example.invalid/asset",
                                   hashlib.sha256(self.data).hexdigest(), len(self.data))

    def test_manifest_deduplicates_supported_models(self):
        self.assertEqual(setup.required_assets(), (encoding_asset_for_model("gpt-4.1-mini"),))

    def test_install_hash_permissions_reuse_and_no_partial_files(self):
        with patch.object(setup, "urlopen", return_value=io.BytesIO(self.data)) as network:
            setup.install_asset(self.directory, self.asset)
            network.assert_called_once_with(self.asset.url, timeout=setup.IO_TIMEOUT_SECONDS)
        destination = self.directory / self.asset.filename
        self.assertEqual(destination.read_bytes(), self.data)
        self.assertEqual(destination.stat().st_mode & 0o777, 0o644)
        with patch.object(setup, "urlopen", side_effect=AssertionError("network")):
            setup.install_asset(self.directory, self.asset)
        self.assertEqual(list(self.directory.iterdir()), [destination])

    def test_failed_download_preserves_old_file_and_cleans_staging(self):
        destination = self.directory / self.asset.filename
        destination.write_bytes(b"previous invalid asset")
        for error in (TimeoutError("private error"), URLError("private URL"), OSError("io")):
            with self.subTest(error=type(error).__name__):
                with patch.object(setup, "urlopen", side_effect=error):
                    with self.assertRaises(OSError):
                        setup.install_asset(self.directory, self.asset)
                self.assertEqual(destination.read_bytes(), b"previous invalid asset")
                self.assertEqual(list(self.directory.iterdir()), [destination])

    def test_truncated_oversized_and_wrong_hash_never_installed(self):
        for data in (b"short", self.data + b"extra", b"X" * len(self.data)):
            with self.subTest(size=len(data)):
                with patch.object(setup, "urlopen", return_value=io.BytesIO(data)):
                    with self.assertRaises(setup.AssetSetupError):
                        setup.install_asset(self.directory, self.asset)
                self.assertEqual(list(self.directory.iterdir()), [])

    def test_total_download_deadline(self):
        with patch.object(setup, "urlopen", return_value=io.BytesIO(self.data)), \
                patch.object(setup, "monotonic", side_effect=[0, setup.DOWNLOAD_DEADLINE_SECONDS + 1]):
            with self.assertRaisesRegex(setup.AssetSetupError, "deadline"):
                setup.install_asset(self.directory, self.asset)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_failed_replace_keeps_existing_file(self):
        destination = self.directory / self.asset.filename
        destination.write_bytes(b"old")
        with patch.object(setup, "urlopen", return_value=io.BytesIO(self.data)), \
                patch.object(setup.os, "replace", side_effect=OSError("failed")):
            with self.assertRaises(OSError):
                setup.install_asset(self.directory, self.asset)
        self.assertEqual(destination.read_bytes(), b"old")
        self.assertEqual(list(self.directory.iterdir()), [destination])

    def test_verify_missing_corrupt_and_directory_never_downloads(self):
        asset = setup.required_assets()[0]
        path = self.directory / asset.filename
        with patch.object(setup, "urlopen", side_effect=AssertionError("network")), \
                patch("sys.stderr", new_callable=io.StringIO) as output:
            self.assertEqual(setup.main(["--verify-only", "--asset-dir", str(self.directory)]), 1)
            path.write_bytes(b"corrupt")
            self.assertEqual(setup.main(["--verify-only", "--asset-dir", str(self.directory)]), 1)
            path.unlink()
            path.mkdir()
            self.assertEqual(setup.main(["--verify-only", "--asset-dir", str(self.directory)]), 1)
            self.assertIn("asset_missing_or_corrupt", output.getvalue())
            self.assertNotIn(str(self.directory), output.getvalue())

    def test_missing_dependency_or_wrong_mapping_fails_startup(self):
        with patch.object(setup, "verified_file", return_value=True), \
                patch.object(OfflineTokenCounter, "count", return_value=None):
            with self.assertRaisesRegex(setup.AssetSetupError, "runtime_tokenizer_unavailable"):
                setup.verify_assets(self.directory)

    def test_network_error_diagnostic_does_not_leak_url(self):
        with patch.object(setup, "install_asset", side_effect=URLError("private upstream URL")), \
                patch("sys.stderr", new_callable=io.StringIO) as output:
            self.assertEqual(setup.main(["--asset-dir", str(self.directory)]), 1)
        self.assertIn("URLError", output.getvalue())
        self.assertNotIn("private upstream URL", output.getvalue())

    def test_startup_does_not_launch_command_when_assets_missing(self):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.tokenizer_startup", sys.executable,
             "-c", "print('UNEXPECTED_CHILD')"], cwd=BACKEND,
            env={**os.environ, "DEEP_READER_TOKENIZER_ASSET_DIR": str(self.directory)},
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("UNEXPECTED_CHILD", result.stdout)
        self.assertIn("asset_missing_or_corrupt", result.stderr)

    def test_shell_syntax_and_local_setup_missing_python(self):
        for filename in ("start_local_dev.sh", "prepare_local_tokenizer.sh"):
            subprocess.run(["bash", "-n", str(ROOT / "scripts" / filename)], check=True)
        result = subprocess.run(["bash", str(ROOT / "scripts/prepare_local_tokenizer.sh")],
            env={**os.environ, "DEEP_READER_PYTHON": str(self.directory / "missing")},
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Install backend requirements", result.stderr)

    def test_local_dev_waits_for_api_verification_before_ui(self):
        scripts = self.directory / "scripts"
        scripts.mkdir()
        for name in ("start_local_dev.sh", "load_local_env.sh"):
            (scripts / name).write_bytes((ROOT / "scripts" / name).read_bytes())
        (self.directory / "Deep_Reader_UI").mkdir()
        migration = self.directory / "Deep_Reflective_Reader/db/migrations"
        migration.mkdir(parents=True)
        (migration / "003_interaction_version_guards.sql").write_text("SELECT 1;")
        (self.directory / ".env.local").write_text(
            "DEEP_READER_UI_PORT=5173\nDEEP_READER_API_PORT=8000\nDEEP_READER_POSTGRES_PORT=5433\n")
        bin_dir = self.directory / "bin"
        bin_dir.mkdir()
        log = self.directory / "calls.jsonl"
        for command in ("docker", "npm", "lsof"):
            path = bin_dir / command
            path.write_text(f"#!{sys.executable}\n" + """
import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
if name == 'lsof':
    sys.exit(1)
with open(os.environ['CALL_LOG'], 'a') as stream:
    stream.write(json.dumps([name, *sys.argv[1:]]) + '\\n')
if name == 'docker' and sys.argv[-1] == 'api' and os.environ['FAIL_API'] == '1':
    sys.exit(1)
""")
            path.chmod(0o755)
        for fail in ("0", "1"):
            log.unlink(missing_ok=True)
            result = subprocess.run(["bash", str(scripts / "start_local_dev.sh")],
                env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
                     "CALL_LOG": str(log), "FAIL_API": fail},
                capture_output=True, text=True, timeout=10)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            api = next(call for call in calls if call[0] == "docker" and call[-1] == "api")
            self.assertEqual(api[-5:], ["up", "-d", "--build", "--wait", "api"])
            self.assertEqual(result.returncode, int(fail), result.stderr)
            self.assertEqual(any(call[0] == "npm" for call in calls), fail == "0")

    @unittest.skipUnless(os.environ.get("DEEP_READER_TEST_TOKENIZER_ASSETS"), "real assets opt-in")
    def test_real_local_setup_and_offline_startup_exec_exit_status(self):
        directory = Path(os.environ["DEEP_READER_TEST_TOKENIZER_ASSETS"])
        with patch.object(setup, "urlopen", side_effect=AssertionError("network")):
            self.assertEqual(setup.main(["--asset-dir", str(directory)]), 0)
        env = {**os.environ, "DEEP_READER_TOKENIZER_ASSET_DIR": str(directory),
               "DEEP_READER_PYTHON": sys.executable, "TIKTOKEN_CACHE_DIR": str(self.directory)}
        subprocess.run(["bash", str(ROOT / "scripts/prepare_local_tokenizer.sh"), "--verify-only"],
                       env=env, check=True, capture_output=True, timeout=10)
        # Deny every socket before executing the startup module in a fresh process.
        script = ("import runpy,socket,sys; "
                  "socket.socket.connect=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('network')); "
                  "sys.argv=['startup',sys.executable,'-c','raise SystemExit(7)']; "
                  "runpy.run_module('scripts.tokenizer_startup',run_name='__main__')")
        result = subprocess.run([sys.executable, "-c", script], cwd=BACKEND, env=env,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("Tokenizer assets verified", result.stdout)
        self.assertEqual(list(self.directory.iterdir()), [])


@unittest.skipUnless(os.environ.get("DEEP_READER_TEST_TOKENIZER_IMAGE"), "rebuilt image opt-in")
class TokenizerContainerTests(unittest.TestCase):
    def run_container(self, command, options=()):
        return subprocess.run(
            ["docker", "run", "--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp",
             *options, os.environ["DEEP_READER_TEST_TOKENIZER_IMAGE"], *command],
            capture_output=True, text=True, timeout=30)

    def test_fresh_and_recreated_container_nonroot_offline(self):
        script = """
import os
from pathlib import Path
from llm.offline_token_counter import TOKEN_COUNTER, SUPPORTED_MODELS
assert not os.environ.get('OPENAI_API_KEY')
assert not os.environ.get('DEEP_READER_LLM_API_KEY')
assert not Path('data').exists()
assert not Path('.venv').exists()
assert not Path('.env.local').exists()
assert not Path('/tmp/data-gym-cache').exists()
for model in SUPPORTED_MODELS:
    cold = TOKEN_COUNTER.count(model, 'Offline startup test.')
    assert cold > 0 and cold == TOKEN_COUNTER.count(model, 'Offline startup test.')
assert not Path('/tmp/data-gym-cache').exists()
print('offline-recreation-ok')
"""
        for _ in range(2):
            result = self.run_container(["python", "-c", script], ["--user", "65534:65534"])
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("offline-recreation-ok", result.stdout)
            self.assertIn("Tokenizer assets verified", result.stdout)

    def test_missing_asset_blocks_child_without_network(self):
        result = self.run_container(["python", "-c", "print('UNEXPECTED_CHILD')"],
                                    ["--tmpfs", "/opt/deep-reader/tokenizer-assets"])
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("asset_missing_or_corrupt", result.stderr)
        self.assertNotIn("UNEXPECTED_CHILD", result.stdout)

    def test_corrupt_asset_blocks_child_without_network(self):
        script = """
import os, runpy, sys
from pathlib import Path
from llm.offline_token_counter import encoding_asset_for_model
asset = encoding_asset_for_model('gpt-4.1-mini')
(Path(os.environ['DEEP_READER_TOKENIZER_ASSET_DIR']) / asset.filename).write_bytes(b'corrupt')
sys.argv = ['startup', 'python', '-c', "print('UNEXPECTED_CHILD')"]
runpy.run_module('scripts.tokenizer_startup', run_name='__main__')
"""
        result = self.run_container(["-c", script], ["--tmpfs", "/opt/deep-reader/tokenizer-assets",
                                                   "--entrypoint", "python"])
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("asset_missing_or_corrupt", result.stderr)
        self.assertNotIn("UNEXPECTED_CHILD", result.stdout)


if __name__ == "__main__":
    unittest.main()
