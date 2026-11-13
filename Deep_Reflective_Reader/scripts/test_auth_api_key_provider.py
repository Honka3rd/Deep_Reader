"""AUTH-01: synthetic credentials and fresh-process lazy DI regressions."""

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from auth.openai_api_key_provider import OpenAIAPIKeyProvider


class OpenAIAPIKeyProviderTests(unittest.TestCase):
    sources = (("OPENAI_API_KEY", {}), ("AUTH_TEST_KEY", {"env_var": "AUTH_TEST_KEY"}))

    def test_missing_and_empty_selected_values_are_rejected(self):
        for name, kwargs in self.sources:
            for value in (None, ""):
                with self.subTest(source=name, value=value):
                    env = {} if value is None else {name: value}
                    with patch.dict(os.environ, env, clear=True):
                        with self.assertRaises(RuntimeError) as error:
                            OpenAIAPIKeyProvider(**kwargs)
                    self.assertEqual(str(error.exception), f"{name} environment variable not found.")

    def test_whitespace_only_selected_values_are_rejected(self):
        for name, kwargs in self.sources:
            for value in ("   ", "\t\r\n ", "\u2003\u00a0\u3000"):
                with self.subTest(source=name, whitespace=repr(value)):
                    with patch.dict(os.environ, {name: value}, clear=True):
                        with self.assertRaises(RuntimeError) as error:
                            OpenAIAPIKeyProvider(**kwargs)
                    self.assertEqual(str(error.exception), f"{name} environment variable not found.")

    def test_valid_values_are_trimmed_without_format_validation(self):
        for name, kwargs in self.sources:
            for value in ("synthetic-auth-key", "synthetic internal space"):
                with self.subTest(source=name, value=value):
                    with patch.dict(os.environ, {name: f"\u2003 \t{value}\r\n\u3000"}, clear=True):
                        self.assertEqual(OpenAIAPIKeyProvider(**kwargs).get(), value)

    def test_only_the_selected_environment_variable_is_used(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": " synthetic-default-key ",
            "AUTH_TEST_KEY": " synthetic-custom-key ",
        }, clear=True):
            self.assertEqual(OpenAIAPIKeyProvider().get(), "synthetic-default-key")
            self.assertEqual(OpenAIAPIKeyProvider(env_var="AUTH_TEST_KEY").get(), "synthetic-custom-key")

    def test_invalid_selected_source_never_falls_back_or_exposes_other_key(self):
        for name, kwargs in self.sources:
            other = "AUTH_TEST_KEY" if name == "OPENAI_API_KEY" else "OPENAI_API_KEY"
            for value in (None, "", "\t \n"):
                with self.subTest(source=name, value=value):
                    env = {other: "synthetic-do-not-expose"}
                    if value is not None:
                        env[name] = value
                    with patch.dict(os.environ, env, clear=True):
                        with self.assertRaises(RuntimeError) as error:
                            OpenAIAPIKeyProvider(**kwargs)
                    self.assertEqual(str(error.exception), f"{name} environment variable not found.")
                    self.assertNotIn("synthetic-do-not-expose", str(error.exception))

    def test_existing_instance_keeps_snapshot_after_environment_changes(self):
        for name, kwargs in self.sources:
            with self.subTest(source=name), patch.dict(os.environ, {name: " synthetic-first "}, clear=True):
                first = OpenAIAPIKeyProvider(**kwargs)
                os.environ[name] = " synthetic-second "
                second = OpenAIAPIKeyProvider(**kwargs)
                self.assertEqual(first.get(), "synthetic-first")
                self.assertEqual(second.get(), "synthetic-second")
                del os.environ[name]
                self.assertEqual(first.get(), "synthetic-first")
                self.assertEqual(second.get(), "synthetic-second")
                with self.assertRaises(RuntimeError):
                    OpenAIAPIKeyProvider(**kwargs)


class AuthLazyDITests(unittest.TestCase):
    def test_missing_or_blank_key_allows_resource_startup_and_persisted_reads(self):
        code = '''
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
with patch('socket.socket.connect', side_effect=AssertionError('unexpected network')):
    from auth.openai_api_key_provider import OpenAIAPIKeyProvider
    from config.app_DI_config import AppDIConfig
    from config.container import ApplicationLookupContainer
    from dependency_injector import providers
    from scripts.test_learning_file_repository import configure, fixture
    with patch.object(OpenAIAPIKeyProvider, '__init__', side_effect=AssertionError('eager key validation')) as initialize:
        container = ApplicationLookupContainer.build(AppDIConfig(structured_storage_backend='file'))
        container.document_profile_store.override(providers.Object(
            SimpleNamespace(load_source=lambda config: SimpleNamespace(document_language_code='en'))))
        try:
            container.init_resources()
            container.reading_preferences_service().read_workspace()
            assert not list(Path('.').rglob('*')), 'startup wrote files'
            docs = container.structured_document_artifact_repository()
            document = fixture()
            docs.save_reparsed_document(document, 'book')
            configure(container.reading_preferences_repository())
            before = {p: p.read_bytes() for p in Path('.').rglob('*') if p.is_file()}
            assert docs.load_document('book').raw_text == document.raw_text
            block = container.learning_repository().read_sources('book').blocks[0]
            result = container.learning_service().read('book', (block.source.content_block_id,))
            assert [component.view.state for component in result.blocks[0].components] == ['missing', 'missing']
            assert container.reading_preferences_service().read_book('book').book.settings.active_mode == 'language_learning'
            after = {p: p.read_bytes() for p in Path('.').rglob('*') if p.is_file()}
            assert before == after, 'reads changed persisted files'
            initialize.assert_not_called()
        finally:
            container.shutdown_resources()
    try:
        container.api_key_provider()
    except RuntimeError as error:
        assert str(error) == 'OPENAI_API_KEY environment variable not found.'
    else:
        raise AssertionError('explicit provider resolution accepted an invalid key')
'''
        for value in (None, "\t \n"):
            with self.subTest(key_missing=value is None), TemporaryDirectory() as root:
                env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]),
                           PYTHONDONTWRITEBYTECODE="1")
                env.pop("OPENAI_API_KEY", None)
                if value is not None:
                    env["OPENAI_API_KEY"] = value
                result = subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
