"""Backed-up real sources, temporary storage and deterministic provider; zero paid calls."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from learning import LearningService
from app.translation_action import TranslationAction
from language.language_proficiency_registry import LanguageProficiencyRegistry
from reading_preferences.models import BookSettings, LanguageLearningSettings
from reading_preferences.profile_source import ExistingProfileLanguageReader
from reading_preferences.service import ReadingPreferencesService
from scripts.private_book_corpus import load_book
from scripts.accept_learning_translation_books import excerpt_document
from scripts.test_learning_file_repository import repositories
from scripts.test_learning_translation_service import Provider
from section_tasks.learning_hints_service import LearningHintsService
from section_tasks.learning_translation_service import LearningTranslationService


def main():
    results = []
    for case, source, target, anchor, separator in (
        ('fa794acef055f235', 'en', 'zh', 'We were in class', '\n\n'),
        ('085747b65094d52f', 'zh', 'en', '2017', '\n\u3000')):
        root, snapshot, _ = load_book(case)
        manifest = json.loads((root/'manifest.json').read_text())
        hashes = {root/name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                  for name in (*manifest['files'], 'manifest.json')}
        start = snapshot.canonical_text.index(anchor)
        text = snapshot.canonical_text[start:snapshot.canonical_text.index(separator, start)]
        with TemporaryDirectory(prefix='shared-translation-source-') as temporary:
            docs, prefs, repo = repositories(Path(temporary))
            docs.save_reparsed_document(excerpt_document(text, source), 'excerpt')
            profile = SimpleNamespace(document_language_code=source, document_language=source)
            preferences = ReadingPreferencesService(prefs, ExistingProfileLanguageReader(lambda *_: profile))
            provider = Provider()
            def respond(prompt, budget):
                provider.calls.append((prompt, budget))
                payload = json.loads(prompt.split('INPUT_JSON:\n', 1)[1])
                return json.dumps({'status':'completed','rows':[{'sentence_id':row['sentence_id'],
                    'translated_text':'這是離線測試譯文。' if target == 'zh' else 'This is an offline test translation.'}
                    for row in payload['sentences']]})
            provider.complete_text_with_output_budget = respond
            service = LearningService(repo, preferences, LearningTranslationService(provider, provider_name='offline'),
                LearningHintsService(provider, provider_name='offline'), lease_ms=60000)
            try:
                action = TranslationAction(docs, SimpleNamespace(load=lambda _: profile), shared_translation=service)
                block, = repo.read_sources('excerpt').blocks
                ordinary = action.generate(doc_name='excerpt', content_block_id=block.source.content_block_id,
                    source_language=source, target_language=target, mutation_id='ordinary')
                assert len(provider.calls) == 1
                workspace = prefs.save_workspace(target, expected_revision=0, mutation_id='native')
                scheme = LanguageProficiencyRegistry.get_scheme(source)
                prefs.save_book('excerpt', BookSettings('language_learning', source, (LanguageLearningSettings(source,
                    scheme.scheme_id, scheme.version, scheme.levels[0].level_id),)), expected_revision=0,
                    expected_workspace_revision=workspace.revision, resolved_source_language=source, mutation_id='learning-mode')
                settings = preferences.read_book('excerpt')
                submission = service.request('excerpt', block.source.content_block_id, 'translation', command_id='learning',
                    expected_source=block.source, expected_book_revision=settings.book.revision,
                    expected_workspace_revision=workspace.revision)
                assert submission.outcome == 'reuse' and len(provider.calls) == 1
                page = service.read('excerpt', (block.source.content_block_id,))
                saved = page.blocks[0].components[0].view.saved
                assert saved is not None and saved.rows
                assert action.read(doc_name='excerpt', content_block_id=block.source.content_block_id)['translations'] == [ordinary.record]
                assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in hashes.items())
                results.append(dict(case=case, sentences=len(saved.rows), provider_calls=1, reuse=True, backups_unchanged=True))
            finally:
                service.shutdown()
    print(json.dumps(dict(pass_all=True, provider='offline fixture', semantic_quality='not assessed', cases=results)))

if __name__ == '__main__': main()
