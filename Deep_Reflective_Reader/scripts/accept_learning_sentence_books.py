"""LLRN-04 offline acceptance on hash-verified immutable book backups.

Run from backend: python -m scripts.accept_learning_sentence_books --report PATH
Only bounded effective blocks are segmented. No reparse, provider, SQL or live
Library access. The private report contains text and must remain outside git.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from document_structure.reading_content import original_blocks, project_document, state_for
from scripts.private_book_corpus import load_book, preserve
from section_tasks.learning_sentence_segmenter import SEGMENTATION_VERSION, segment_learning_block
from shared.learning_source import LearningSourceBinding, SentenceResource, text_hash


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = {'version': SEGMENTATION_VERSION, 'status': 'failed', 'cases': [],
              'scope': 'offline bounded segmentation; not generation/lifecycle/API acceptance'}
    backups = {}
    try:
        for case, language, anchor in (('fa794acef055f235', 'en', 'We were in class'),
                                       ('085747b65094d52f', 'zh', '2017年9月5日')):
            root, snapshot, document = load_book(case)
            manifest = json.loads((root / 'manifest.json').read_text())
            for name in (*manifest['files'], 'manifest.json'):
                path = root / name
                backups[path] = hashlib.sha256(path.read_bytes()).hexdigest()
            before = json.dumps(document.to_dict(), ensure_ascii=False, sort_keys=True)
            projected = project_document(document)
            candidates = []
            found_anchor = False
            for c, s, u, b in original_blocks(projected):
                found_anchor |= anchor in b.content
                if found_anchor and 150 <= len(b.content) <= 2000 and any(t in b.content for t in '.。!?！？'):
                    candidates.append((c, s, u, b))
                    if len(candidates) == 6:
                        break
            if len(candidates) != 6:
                raise AssertionError('bounded_book_fixture_missing')
            entries = []
            started = perf_counter()
            for chapter, section, unit, block in candidates:
                source = LearningSourceBinding(snapshot.namespace, document.document_id,
                    chapter.chapter_id, section.section_id, unit.unit_id, block.block_id,
                    document.parse_provenance.get('storage_structure_version'),
                    document.parse_provenance.get('interaction_epoch'),
                    state_for(document).get('block_content_epochs', {}).get(block.block_id), text_hash(block.content))
                result = segment_learning_block(source=source, effective_text=block.content, source_language=language)
                assert ''.join(p.text for p in result.project(block.content)) == block.content
                assert result == segment_learning_block(source=source, effective_text=block.content, source_language=language)
                assert SentenceResource.from_dict(result.to_dict()) == result
                result.validate_source(source, block.content)
                entries.append({'block_id': block.block_id, 'characters': len(block.content),
                    'source_hash': source.effective_content_hash, 'resource': result.to_dict(),
                    'source_text': block.content,
                    'sentences': [p.text for p in result.project(block.content) if p.sentence_id is not None]})
            assert json.dumps(document.to_dict(), ensure_ascii=False, sort_keys=True) == before
            report['cases'].append({'case': case, 'language': language, 'blocks': entries,
                                    'elapsed_seconds': perf_counter() - started, 'document_unchanged': True})
        report['status'] = 'passed'
    finally:
        report['backups_unchanged'] = all(hashlib.sha256(p.read_bytes()).hexdigest() == sha for p, sha in backups.items())
        preserve(args.report, json.dumps(report, ensure_ascii=False, indent=2).encode())
        assert report['backups_unchanged']
    print(json.dumps({'status': report['status'], 'books': len(report['cases']),
        'blocks': sum(len(c['blocks']) for c in report['cases']),
        'sentences': sum(len(b['sentences']) for c in report['cases'] for b in c['blocks']),
        'backups_unchanged': report['backups_unchanged'], 'report': str(args.report)}))


if __name__ == '__main__': run()
