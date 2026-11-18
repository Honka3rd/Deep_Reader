"""Build recipe contracts and real registry projection; no Docker daemon required.

These tests do not substitute for approved cold/warm image and deployment acceptance.
"""
import json
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


class AptBootstrapTests(unittest.TestCase):
    """Execute the production RUN shell; apt/dpkg ports cannot change this host.

    The small version port covers release ordering, not Debian version syntax.
    Real Debian comparator/package execution is separate container acceptance.
    """

    def run_recipe(self, **settings):
        recipe = next(line for line in stages()['ocr-runtime']['instructions']
                      if line.startswith('RUN ') and 'apt-get' in line)
        recipe = re.sub(r'--mount=\S+\s*', '', recipe[4:])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'ocr-packages'
            manifest.write_text('tesseract-ocr-eng\ntesseract-ocr-chi-tra\n')
            recipe = recipe.replace('/tmp/ocr-packages', str(manifest))
            command = root / 'command'
            command.write_text('#!' + sys.executable + '\n' + r'''
import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
root = Path(os.environ['APT_TEST_ROOT'])
with (root / 'calls').open('a') as stream:
    stream.write(json.dumps([name, *args]) + '\n')
if name == 'apt-get':
    bootstrap = '--only-upgrade' in args
    updating = 'update' in args
    if bootstrap or updating:
        if os.environ.get('FAIL_STAGE') == ('upgrade' if bootstrap else 'update'):
            sys.exit(100)
    if bootstrap:
        (root / 'upgraded').touch()
    elif not updating:
        (root / 'ocr-called').touch()
elif name == 'dpkg-query':
    if os.environ.get('QUERY_FAIL') == args[-1]:
        sys.exit(1)
    version = os.environ.get('APT_VERSION' if args[-1] == 'apt' else 'LIB_VERSION', '3.0.4')
    print(version if (root / 'upgraded').exists() else '3.0.3')
elif name == 'dpkg':
    def release(value):
        return tuple(int(n) for n in value.split('.'))
    sys.exit(0 if release(args[1]) >= release(args[3]) else 1)
''')
            command.chmod(0o755)
            for name in ('apt-get', 'dpkg-query', 'dpkg'):
                (root / name).symlink_to(command)
            result = subprocess.run(['sh', '-c', recipe], capture_output=True, text=True,
                                    env={**os.environ, **settings, 'APT_TEST_ROOT': str(root),
                                         'PATH': str(root) + os.pathsep + os.environ['PATH']}, timeout=10)
            calls = [json.loads(line) for line in (root / 'calls').read_text().splitlines()]
            return result, calls, (root / 'ocr-called').exists(), manifest.exists()

    def test_upgrade_and_both_version_checks_precede_ocr(self):
        result, calls, installed, manifest = self.run_recipe()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(installed)
        self.assertFalse(manifest)
        apt_calls = [call for call in calls if call[0] == 'apt-get']
        self.assertEqual(len(apt_calls), 3)
        self.assertIn('update', apt_calls[0])
        self.assertIn('Acquire::Retries=0', apt_calls[0])
        self.assertIn('--only-upgrade', apt_calls[1])
        self.assertIn('Acquire::Retries=0', apt_calls[1])
        self.assertIn('apt', apt_calls[1])
        self.assertIn('libapt-pkg7.0', apt_calls[1])
        self.assertIn('tesseract-ocr-chi-tra', apt_calls[2])
        self.assertNotIn('Acquire::Retries=0', apt_calls[2])
        self.assertEqual([call[-1] for call in calls if call[0] == 'dpkg-query'],
                         ['apt', 'libapt-pkg7.0'])
        self.assertEqual(sum(call[0] == 'dpkg' for call in calls), 2)
        self.assertLess(max(i for i, call in enumerate(calls) if call[0] == 'dpkg'),
                        calls.index(apt_calls[2]))

    def test_later_patch_versions_are_accepted(self):
        result, _, installed, _ = self.run_recipe(APT_VERSION='3.0.5', LIB_VERSION='3.0.5')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(installed)

    def test_each_old_version_stops_before_ocr(self):
        for key in ('APT_VERSION', 'LIB_VERSION'):
            with self.subTest(package=key):
                result, _, installed, manifest = self.run_recipe(**{key: '3.0.3'})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('apt_bootstrap_version_unsupported', result.stderr)
                self.assertFalse(installed)
                self.assertTrue(manifest)

    def test_missing_package_version_stops_before_ocr(self):
        for package in ('apt', 'libapt-pkg7.0'):
            with self.subTest(package=package):
                result, _, installed, manifest = self.run_recipe(QUERY_FAIL=package)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('apt_bootstrap_version_unsupported', result.stderr)
                self.assertFalse(installed)
                self.assertTrue(manifest)

    def test_update_failure_stops_without_upgrade_or_ocr(self):
        result, calls, installed, manifest = self.run_recipe(FAIL_STAGE='update')
        self.assertEqual(result.returncode, 100)
        self.assertEqual(len(calls), 1)
        self.assertFalse(installed)
        self.assertTrue(manifest)

    def test_upgrade_failure_stops_without_checks_or_ocr(self):
        result, calls, installed, manifest = self.run_recipe(FAIL_STAGE='upgrade')
        self.assertEqual(result.returncode, 100)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call[0] == 'apt-get' for call in calls))
        self.assertFalse(installed)
        self.assertTrue(manifest)


if __name__ == '__main__':
    unittest.main()
