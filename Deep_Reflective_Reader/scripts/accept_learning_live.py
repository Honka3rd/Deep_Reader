"""LLRN-22: bounded live provider / production HTTP / disposable SQL acceptance.

Requires --live (or explicit offline --replay-report) and a disposable DSN.
Never uses default application storage. Evidence contains private excerpts: place
--report-dir outside git or under ignored data/test-books. This runner accepts
system behavior only; it never declares model semantic quality accepted.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
from threading import Thread
from time import monotonic, sleep
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from dependency_injector import providers
from fastapi import FastAPI
import httpx
import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
import uvicorn

from config.app_DI_config import AppDIConfig
from config.container import ApplicationLookupContainer
from language.language_proficiency_registry import LanguageProficiencyRegistry
from learning_routes import router, learning_lifespan
from scripts.accept_learning_translation_books import excerpt_document
from scripts.private_book_corpus import load_book, preserve

BASE = '/documents/reading-interactions/learning/'
BOOK = '/documents/reading-preferences/'
WORKSPACE = '/reading-preferences/workspace'
MIGRATIONS = Path(__file__).resolve().parents[1] / 'db/migrations'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    default=str).encode()).hexdigest()


def record_returned_text(record, value):
    """Raw output is evidence; optional decoding must not change provider outcome."""
    record.update(status='returned', response_text=value)
    try:
        record['response'] = json.loads(value)
    except (ValueError, TypeError, RecursionError) as error:
        record['response_decode_error'] = type(error).__name__


def remap_sentence_references(raw, prompt):
    """Keep payload syntax intact, including duplicate keys rejected downstream."""
    try:
        value = json.loads(raw)
    except (ValueError, TypeError, RecursionError):
        return raw
    if not isinstance(value, dict):
        return raw
    old = value.get('rows', value.get('sentences'))
    rows = json.loads(prompt.split('INPUT_JSON:\n', 1)[1])['sentences']
    if (not isinstance(old, list) or len(old) != len(rows)
            or any(not isinstance(item, dict) or not isinstance(item.get('sentence_id'), str)
                   for item in old)):
        return raw
    mapping = {a['sentence_id']: b['sentence_id'] for a, b in zip(old, rows)}
    # Scan complete string tokens so embedded quoted text and escaped keys are
    # handled correctly. Never parse/dump the whole payload to perform a remap.
    tokens = list(re.finditer(r'"(?:[^"\\]|\\.)*"', raw))
    edits = []
    for key, token in zip(tokens, tokens[1:]):
        if raw[key.end():token.start()].strip() != ':' or json.loads(key.group()) != 'sentence_id':
            continue
        original = json.loads(token.group())
        if original in mapping:
            edits.append((token.start(), token.end(), json.dumps(mapping[original])))
    for start, end, replacement in reversed(edits):
        raw = raw[:start] + replacement + raw[end:]
    return raw


class RecordingProvider:
    """Observe real SDK receipts without replacing requests or changing retry policy."""
    def __init__(self, provider):
        self.provider, self.calls = provider, []

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def complete_text_once_with_output_budget(self, prompt, budget):
        # Acceptance-only spending bound, NOT a production text/token limit.
        if len(self.calls) >= 8:
            raise RuntimeError('acceptance_call_bound_exceeded')
        record = {'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                  'output_limit': budget, 'status': 'started'}
        self.calls.append(record)
        start = monotonic()
        import llm.openai_llm_provider as module
        original = module.OpenAIClient
        class Client:
            def __init__(self, **kwargs):
                assert kwargs['max_retries'] == 0
                self.inner = original(**kwargs)
            def __enter__(self):
                client = self.inner.__enter__()
                endpoint = (client.responses if self_outer.provider.get_model_capabilities().endpoint_kind == 'responses'
                            else client.chat.completions)
                create = endpoint.create
                def observed(**kwargs):
                    response = create(**kwargs)
                    record.update(request_id=response._request_id, response_id=response.id,
                                  model=response.model,
                                  usage=response.usage.model_dump() if response.usage else None)
                    return response
                endpoint.create = observed
                return client
            def __exit__(self, *args):
                return self.inner.__exit__(*args)
        self_outer = self
        try:
            # Runs are serial; this scoped observation is never installed in the app.
            with patch.object(module, 'OpenAIClient', Client):
                value = self.provider.complete_text_once_with_output_budget(prompt, budget)
            record_returned_text(record, value)
            return value
        except Exception as error:
            record.update(status='failed_or_unknown', error_type=type(error).__name__)
            raise
        finally:
            record['seconds'] = round(monotonic()-start, 3)


class ReplayProvider:
    """Replay private captured output for harness/UI regressions; never a live gate."""
    def __init__(self, report):
        from llm.llm_model_capabilities import LLMModelCapabilities
        self.report, self.calls = report, []
        self.caps = LLMModelCapabilities(report['model'], 'responses', 1047576, 32768)
        self.tokenizer_identity = 'acceptance-replay-character-counter'

    def get_model_capabilities(self):
        return self.caps

    def count_input_tokens(self, text):
        return len(text)  # deterministic harness capacity, not a production estimate

    def complete_text_once_with_output_budget(self, prompt, budget):
        previous = self.report['provider_calls'][len(self.calls)]
        if 'response_text' in previous:
            raw, source = previous['response_text'], 'raw_text'
        elif 'response' in previous:
            raw, source = json.dumps(previous['response'], ensure_ascii=False), 'legacy_json'
        else:
            raise ValueError('acceptance_replay_response_missing')
        value = remap_sentence_references(raw, prompt)
        record = {'replayed': True,
                  'original_request_id': previous.get('request_id') or previous.get('original_request_id'),
                  'response_source': source, 'output_limit': budget}
        record_returned_text(record, value)
        self.calls.append(record)
        return value


@contextmanager
def serve(container, fixture):
    app = FastAPI(lifespan=learning_lifespan)
    app.state.learning_container = container
    app.include_router(router)
    @app.get('/fixture')
    def get_fixture():
        return fixture
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level='warning'))
    thread = Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    try:
        deadline = monotonic()+20
        while not server.started:
            if not thread.is_alive() or monotonic() > deadline:
                raise RuntimeError('acceptance_server_start_failed')
            sleep(.05)
        yield f'http://127.0.0.1:{port}'
    finally:
        server.should_exit = True
        thread.join(330)
        sock.close()
        if thread.is_alive():
            raise RuntimeError('acceptance_worker_not_drained')


class Journey:
    def __init__(self, client, container, provider, report):
        self.client, self.container, self.provider, self.report = client, container, provider, report

    def request(self, method, path, body=None):
        response = self.client.request(method, path, **({'json': body} if body is not None else {}))
        value = response.json()
        self.report['http'].append({'method': method, 'path': path, 'body': body,
                                    'status': response.status_code, 'response': value})
        if response.status_code != 200:
            raise RuntimeError('acceptance_http_'+str(response.status_code))
        assert response.headers['cache-control'] == 'no-store'
        return value

    def select(self, case, level):
        workspace = self.request('GET', WORKSPACE)
        if workspace['native_language'] != case['target']:
            workspace = self.request('PUT', WORKSPACE, {'native_language': case['target'],
                'expected_revision': workspace['revision'], 'mutation_id': uuid4().hex})
        pref = self.container.reading_preferences_repository().read_book(case['doc_name'])
        scheme = LanguageProficiencyRegistry.get_scheme(case['source'])
        self.request('POST', BOOK+'apply', {'doc_name': case['doc_name'],
            'expected_revision': pref.revision, 'expected_workspace_revision': workspace['revision'],
            'mutation_id': uuid4().hex, 'source_language_override': case['source'],
            'selection': {'mode_id': 'language_learning', 'version': '1', 'language_code': case['source'],
                          'scheme_id': scheme.scheme_id, 'scheme_version': scheme.version, 'level_id': level}})

    def read(self, case):
        return self.request('POST', BASE+'read', {'doc_name': case['doc_name'],
            'content_block_ids': [case['block_id']]})

    def generate(self, case, component, expected='claimed'):
        page = self.read(case)
        prefs = self.container.reading_preferences_repository()
        body = {'doc_name': case['doc_name'], 'content_block_id': case['block_id'],
            'component': component, 'command_id': uuid4().hex,
            'expected_source_token': page['blocks'][0]['source_token'],
            'expected_workspace_revision': prefs.read_workspace().revision,
            'expected_book_revision': prefs.read_book(case['doc_name']).revision}
        receipt = self.request('POST', BASE+'generate', body)
        assert receipt['outcome'] == expected, receipt
        # Command replay must return the original receipt, including after commit.
        replay = self.request('POST', BASE+'generate', body)
        assert replay == {**receipt, 'replay': True}
        if expected == 'claimed':
            deadline = monotonic()+320
            while True:
                state = self.request('POST', BASE+'operations/read', {'handle': receipt['operation']})
                if state['state'] in ('succeeded', 'failed', 'unknown', 'stale'):
                    assert state['state'] == 'succeeded', state
                    break
                if monotonic() > deadline:
                    raise RuntimeError('acceptance_operation_timeout')
                sleep(.5)
        return self.read(case)


def run(args):
    report = {'status': 'failed', 'semantic_review': 'outside_system_scope', 'cases': [], 'http': [],
              'cleanup': {}, 'scope': 'bounded real excerpts; not whole books or all registry languages'}
    hashes, provider, schema, dsn = {}, None, None, None
    schema_created = False
    replay = json.loads(args.replay_report.read_text()) if args.replay_report else None
    report['live_provider'] = replay is None
    if replay:
        report['replay_report_sha256'] = hashlib.sha256(args.replay_report.read_bytes()).hexdigest()
        report['scope'] = 'recorded output replay for API/SQL/Reader regression; no live provider or quality acceptance'
    directory = args.report_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    try:
        dsn = os.environ['DEEP_READER_TEST_POSTGRES_DSN']
        cases = []
        for case_id, source, target, anchor, separator, levels in (
            ('fa794acef055f235', 'en', 'zh', 'We were in class', '\n\n', ['A1', 'C2']),
            ('085747b65094d52f', 'zh', 'en', '2017', '\n\u3000', ['1', '6'])):
            root, snapshot, _ = load_book(case_id)
            manifest = json.loads((root/'manifest.json').read_text())
            for name in (*manifest['files'], 'manifest.json'):
                path = root/name
                hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
            start = snapshot.canonical_text.index(anchor)
            end = snapshot.canonical_text.index(separator, start)
            cases.append({'doc_name': 'acceptance-'+case_id, 'case_id': case_id, 'source': source,
                          'target': target, 'text': snapshot.canonical_text[start:end], 'levels': levels})
        cases.append({'doc_name': 'acceptance-mixed', 'case_id': 'mixed-control', 'source': 'en',
            'target': 'zh', 'text': 'She said «bonjour, mon ami», then reluctantly departed.', 'levels': ['A1']})
        schema = 'llrn22_'+uuid4().hex
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
        schema_created = True
        scoped = make_conninfo(dsn, options=f'-c search_path={schema}')
        with psycopg.connect(scoped) as conn:
            report['migrations'] = []
            for path in sorted(MIGRATIONS.glob('*.sql')):
                conn.execute(path.read_text())
                report['migrations'].append({'name': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        namespace = uuid4().hex
        def container():
            result = ApplicationLookupContainer.build(AppDIConfig(structured_storage_backend='postgres',
                postgres_dsn=scoped, postgres_namespace=namespace))
            result.document_profile_store.override(providers.Object(SimpleNamespace(
                load_source=lambda config: SimpleNamespace(document_language_code=next(
                    c['source'] for c in cases if c['doc_name'] == config.doc_name)))))
            if provider is not None:
                result.llm_provider.override(providers.Object(provider))
            return result
        c = container()
        provider = ReplayProvider(replay) if replay else RecordingProvider(c.llm_provider())
        c.llm_provider.override(providers.Object(provider))
        report['model'] = provider.get_model_capabilities().model_name
        docs, repo = c.structured_document_artifact_repository(), c.learning_repository()
        for case in cases:
            docs.save_reparsed_document(excerpt_document(case['text'], case['source']), case['doc_name'])
            blocks = repo.read_sources(case['doc_name']).blocks
            assert len(blocks) == 1
            # The import/storage owner normalizes reading text. Feed Reader that
            # same effective source, while retaining the immutable input excerpt.
            case['backup_excerpt'] = case['text']
            case['text'] = blocks[0].effective_text
            if replay:
                previous = next(v for v in replay['cases'] if v['case_id'] == case['case_id'])
                assert all(case[k] == previous[k] for k in ('text', 'source', 'target', 'levels'))
            case['block_id'] = blocks[0].source.content_block_id
            case['source_hash'] = digest(asdict(repo.read_sources(case['doc_name'])))
        with serve(c, cases) as url, httpx.Client(base_url=url, timeout=330) as client:
            journey = Journey(client, c, provider, report)
            for case in cases:
                journey.select(case, case['levels'][0])
                assert all(v['state'] == 'missing' for v in journey.read(case)['blocks'][0]['components'])
                translation = journey.generate(case, 'translation')['blocks'][0]['components'][0]['saved']
                case['outputs'] = []
                for level in case['levels']:
                    journey.select(case, level)
                    page = journey.generate(case, 'hints')
                    assert page['blocks'][0]['components'][0]['saved'] == translation
                    sentences = page['blocks'][0]['sentences']
                    assert ''.join(s['text'] for s in sentences) == case['text']
                    assert [s['sentence_id'] for s in sentences if s['sentence_id']] == [
                        row['sentence_id'] for row in translation['rows']]
                    case['outputs'].append({'level': level, 'page': page})
                journey.select(case, case['levels'][0])
                before = len(provider.calls)
                for component in ('translation', 'hints'):
                    journey.generate(case, component, expected='reuse')
                assert len(provider.calls) == before
                assert digest(asdict(repo.read_sources(case['doc_name']))) == case['source_hash']
                report['cases'].append(case)
                print('PASS '+('replay' if replay else 'live')+'/API/SQL '+case['case_id'], flush=True)
        # New DI/lifespan: persisted SQL results, not runtime memory.
        c = container()
        with serve(c, cases) as url, httpx.Client(base_url=url, timeout=330) as client:
            journey = Journey(client, c, provider, report)
            before = len(provider.calls)
            for case in cases:
                journey.select(case, case['levels'][0])
                page = journey.read(case)
                assert page['blocks'][0]['components'] == case['outputs'][0]['page']['blocks'][0]['components']
                if case['case_id'] != 'mixed-control':
                    env = {**os.environ, 'LEARNING_LIVE_API': url, 'LEARNING_LIVE_BOOK': case['doc_name'],
                           'LEARNING_LIVE_REPORT_DIR': str(directory), 'LEARNING_LIVE_REPLAY': '1' if replay else '0'}
                    result = subprocess.run(['node', str(Path(__file__).with_name('accept_learning_live_reader.mjs'))],
                        env=env, capture_output=True, text=True, timeout=120)
                    preserve(directory/(case['case_id']+'-browser.log'), (result.stdout+result.stderr).encode())
                    assert result.returncode == 0, 'acceptance_browser_failed'
            assert len(provider.calls) == before == 8
            if not replay:
                assert all(call.get('request_id') and call.get('usage') and call['status'] == 'returned'
                           for call in provider.calls)
            for case in cases:
                assert digest(asdict(c.learning_repository().read_sources(case['doc_name']))) == case['source_hash']
            report['restart_and_browser_reuse_calls'] = 0
        report['status'] = 'replay_checks_passed' if replay else 'technical_checks_passed'
    except Exception as error:
        report['error_type'] = type(error).__name__
        # Safe diagnostic codes only; never dump connection strings/provider errors.
        print('FAIL acceptance: '+type(error).__name__, flush=True)
        import traceback
        report['trace_frames'] = [{'file': Path(f.filename).name, 'line': f.lineno, 'function': f.name}
                                  for f in traceback.extract_tb(error.__traceback__)]
    finally:
        report['provider_calls'] = provider.calls if provider else []
        report['live_provider_invocations'] = 0 if replay else len(report['provider_calls'])
        report['backup_hashes'] = {str(p): h for p, h in hashes.items()}
        report['backups_unchanged'] = all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in hashes.items())
        if schema_created:
            try:
                with psycopg.connect(dsn, autocommit=True) as conn:
                    conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
                    report['cleanup']['schema_removed'] = conn.execute(
                        'SELECT 1 FROM pg_namespace WHERE nspname=%s', (schema,)).fetchone() is None
            except Exception as error:
                report['status'] = 'failed'
                report['cleanup'].update(schema_removed=False, error_type=type(error).__name__)
        preserve(directory/'report.json', json.dumps(report, ensure_ascii=False, indent=2).encode())
        print('Evidence: '+str(directory/'report.json'), flush=True)
    return 0 if report['status'] in ('technical_checks_passed', 'replay_checks_passed') and report['backups_unchanged'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--live', action='store_true')
    mode.add_argument('--replay-report', type=Path)
    parser.add_argument('--report-dir', required=True, type=Path)
    raise SystemExit(run(parser.parse_args()))
