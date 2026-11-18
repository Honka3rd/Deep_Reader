"""Build recipe contracts and real registry projection; no Docker daemon required.

These tests do not substitute for approved cold/warm image and deployment acceptance.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent


def instructions():
    source = (ROOT / 'Dockerfile').read_text()
    return [line.strip() for line in source.replace('\\\n', ' ').splitlines()
            if line.strip() and not line.lstrip().startswith('#')]


def stages():
    result = {}
    current = None
    for line in instructions():
        if line.startswith('FROM '):
            parts = line.split()
            current = parts[3] if len(parts) == 4 else 'unnamed'
            result[current] = {'parent': parts[1], 'instructions': []}
        else:
            result[current]['instructions'].append(line)
    return result


class DockerBuildCacheTests(unittest.TestCase):
    def stage(self, name):
        graph = stages()
        self.assertIn(name, graph, 'Missing isolated build stage: ' + name)
        return graph[name]

    def test_python_dependencies_do_not_consume_registry_or_application(self):
        stage = self.stage('python-dependencies')
        self.assertEqual(stage['parent'], 'base')
        copies = [line for line in stage['instructions'] if line.startswith(('COPY ', 'ADD '))]
        self.assertEqual(len(copies), 1)
        self.assertIn('Deep_Reflective_Reader/requirements.txt', copies[0])
        joined = '\n'.join(stage['instructions'])
        self.assertIn('python -m venv /opt/venv', joined)
        self.assertIn('/opt/venv/bin/python -m pip install -r', joined)

    def test_pip_download_cache_survives_dependency_layer_changes(self):
        recipe = '\n'.join(instructions())
        self.assertNotIn('PIP_NO_CACHE_DIR', recipe)
        stage = '\n'.join(self.stage('python-dependencies')['instructions'])
        self.assertRegex(stage, r'--mount=type=cache,[^ ]*target=/root/.cache/pip')
        self.assertIn('${TARGETARCH}', stage)
        self.assertIn('sharing=locked', stage)
        self.assertNotIn('--no-cache-dir', stage)

    def test_ocr_install_consumes_only_generated_manifest(self):
        manifest = '\n'.join(self.stage('ocr-manifest')['instructions'])
        self.assertIn('doc_loaders.pdf_ocr_language_policy --debian-packages', manifest)
        self.assertIn('sort -u', manifest)
        stage = self.stage('ocr-runtime')
        self.assertEqual(stage['parent'], 'base')
        recipe = '\n'.join(stage['instructions'])
        self.assertIn('COPY --from=ocr-manifest /ocr-packages', recipe)
        self.assertNotIn('language_profile_registry.py', recipe)
        self.assertNotIn('requirements.txt', recipe)
        self.assertIn('$(cat /tmp/ocr-packages)', recipe)

    def test_apt_downloads_and_lists_use_locked_architecture_caches(self):
        recipe = '\n'.join(self.stage('ocr-runtime')['instructions'])
        for directory in ('/var/cache/apt', '/var/lib/apt'):
            mount = re.search(r'--mount=type=cache,[^ ]*target=' + re.escape(directory) + r'(?=,|\s)[^ ]*', recipe)
            self.assertIsNotNone(mount, directory)
            self.assertIn('sharing=locked', mount.group())
            self.assertIn('${TARGETARCH}', mount.group())
        self.assertIn('rm -f /etc/apt/apt.conf.d/docker-clean', recipe)
        self.assertIn('Keep-Downloaded-Packages', recipe)
        self.assertNotIn('rm -rf /var/lib/apt/lists', recipe)

    def test_tokenizer_stage_does_not_consume_ocr_or_application(self):
        stage = self.stage('tokenizer-assets')
        self.assertEqual(stage['parent'], 'python-dependencies')
        recipe = '\n'.join(stage['instructions'])
        self.assertIn('python -m scripts.prepare_tokenizer_assets', recipe)
        self.assertNotIn('--from=ocr', recipe)
        self.assertNotIn('COPY Deep_Reflective_Reader/ ./', recipe)
        self.assertNotIn('language_profile_registry', recipe)

    def test_final_image_contains_dependencies_and_offline_verified_assets(self):
        stage = self.stage('runtime')
        self.assertEqual(stage['parent'], 'ocr-runtime')
        lines = stage['instructions']
        recipe = '\n'.join(lines)
        self.assertIn('COPY --from=python-dependencies /opt/venv /opt/venv', recipe)
        self.assertIn('PATH="/opt/venv/bin:$PATH"', recipe)
        self.assertIn('COPY --from=tokenizer-assets /opt/deep-reader/tokenizer-assets', recipe)
        app_copy = lines.index('COPY Deep_Reflective_Reader/ ./')
        final = '\n'.join(lines[app_copy + 1:])
        self.assertIn('scripts.prepare_tokenizer_assets --verify-only', final)
        self.assertIn('scripts.verify_ocr_assets --load-models', final)
        self.assertNotIn('pip install', final)
        self.assertNotIn('apt-get', final)
        self.assertIn('ENTRYPOINT ["python", "-m", "scripts.tokenizer_startup"]', final)

    def test_vertical_models_keep_pinned_sources_and_integrity_checks(self):
        from doc_loaders.pdf_ocr_language_policy import VERTICAL_MODEL_HASHES
        recipe = '\n'.join(self.stage('ocr-runtime')['instructions'])
        for model, digest in VERTICAL_MODEL_HASHES.items():
            self.assertIn('/87416418657359cb625c412a48b6e1d6d41c29bd/' + model + '.traineddata', recipe)
            self.assertIn(digest + '  /usr/share/tesseract-ocr/5/tessdata/' + model + '.traineddata', recipe)
        self.assertEqual(recipe.count('sha256sum -c -'), len(VERTICAL_MODEL_HASHES))


class RegistryManifestTests(unittest.TestCase):
    def project(self, extra=''):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ('language/language_code.py', 'language/language_profile_registry.py',
                             'doc_loaders/pdf_ocr_language_policy.py'):
                path = root / relative
                path.parent.mkdir(exist_ok=True)
                shutil.copyfile(BACKEND / relative, path)
            registry = root / 'language/language_profile_registry.py'
            registry.write_text(registry.read_text() + '\n' + extra)
            environment = {**os.environ, 'PYTHONPATH': str(root), 'PYTHONDONTWRITEBYTECODE': '1'}
            result = subprocess.run([sys.executable, '-m', 'doc_loaders.pdf_ocr_language_policy',
                                     '--debian-packages'], cwd=root, env=environment,
                                    capture_output=True, text=True, timeout=15)
            manifest = ''.join(package + '\n' for package in sorted(set(result.stdout.split())))
            return result, manifest

    def test_manifest_preserves_full_registered_model_set(self):
        from doc_loaders.pdf_ocr_language_policy import registered_ocr_models
        result, manifest = self.project()
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = {'tesseract-ocr-' + name.replace('_', '-') for name in (*registered_ocr_models(), 'osd')}
        self.assertEqual(set(manifest.splitlines()), expected)
        self.assertIn('tesseract-ocr-chi-sim', expected)
        self.assertIn('tesseract-ocr-chi-tra', expected)

    def test_non_resource_edit_preserves_bytes_but_language_change_does_not(self):
        result, original = self.project()
        self.assertEqual(result.returncode, 0)
        result, unrelated = self.project('LanguageProfileRegistry.build_comment = "unrelated cue revision"')
        self.assertEqual(result.returncode, 0)
        self.assertEqual(original, unrelated)
        result, changed = self.project('LanguageProfileRegistry._PROFILES.pop(LanguageCode.EN)')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(original.splitlines()) - set(changed.splitlines()), {'tesseract-ocr-eng'})
        self.assertNotEqual(original, changed)

    def test_unmapped_language_fails_without_publishing_package_output(self):
        result, manifest = self.project(
            'LanguageProfileRegistry.get_supported_language_codes = classmethod(lambda cls: ["unmapped"])')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(manifest, '')
        self.assertIn('ocr_registry_mapping_missing', result.stderr)


if __name__ == '__main__':
    unittest.main()
