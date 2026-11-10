"""Explicit two-book live-provider acceptance on immutable backed-up excerpts.

Uses temporary document/preferences/learning repositories, never the live Library.
Private outputs are saved under ignored data/test-books. No automatic retries.
Source scripts/load_local_env.sh before running with --live.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from time import time
from uuid import uuid4

from document_structure.learning_repository import OperationWrite
from document_structure.structured_document import StructuredChapter, StructuredDocument, StructuredSection
from language.language_proficiency_registry import LanguageProficiencyRegistry
from reading_preferences.models import BookSettings, LanguageLearningSettings
from scripts.private_book_corpus import CORPUS_ROOT, load_book, preserve
from scripts.test_learning_file_repository import repositories
from section_tasks.learning_generation_policy import GenerationBinding, GenerationCommand
from section_tasks.learning_sentence_segmenter import segment_learning_block
from section_tasks.learning_hints_service import LearningHintsError, LearningHintsService
from shared.learning_results import LearningProficiencyRef
from shared.task_unit_model import TaskUnit


def now(): return int(time()*1000)


def excerpt_document(text, language):
    unit=TaskUnit(unit_id='unit',title='Excerpt',container_title=None,content=text,
        source_section_ids=['section'],is_fallback_generated=False,parent_section_id='section')
    section=StructuredSection('section',0,'Excerpt',2,text,0,len(text),parent_chapter_id='chapter',task_units=[unit])
    return StructuredDocument('excerpt','Acceptance excerpt',None,language,text,
        chapters=[StructuredChapter('chapter','Excerpt',1,None,[section])],
        parse_provenance={'storage_structure_version':1})


class RecordingProvider:
    """Keep model evidence only in the ignored private acceptance report."""
    def __init__(self, provider):
        self.provider, self.calls = provider, []

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def complete_text_with_output_budget(self, prompt, budget):
        record = {'output_limit': budget}
        self.calls.append(record)
        response = self.provider.complete_text_with_output_budget(prompt, budget)
        record['response'] = response
        return response


def run():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true',required=True)
    parser.parse_args()
    from auth.openai_api_key_provider import OpenAIAPIKeyProvider
    from config.app_DI_config import AppDIConfig
    from llm.openai_llm_provider import OpenAILLMProvider
    config=AppDIConfig()
    provider=OpenAILLMProvider(OpenAIAPIKeyProvider(),model=config.llm_model,
                              target_max_output_tokens=config.target_max_output_tokens)
    provider.llm.max_retries=0  # acceptance must not invisibly repeat paid work
    provider=RecordingProvider(provider)
    service=LearningHintsService(provider,provider_name='openai')
    directory=CORPUS_ROOT/'learning-hints'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8])
    report={'status':'failed','cases':[],'scope':'live two-book excerpts at two levels plus mixed-language control; temporary file persistence, not API/UI/SQL acceptance',
            'semantic_review':'required'}
    hashes={}
    try:
        for case,source,target,anchor,separator,level in (
            ('fa794acef055f235','en','zh','We were in class','\n\n','A1'),
            ('fa794acef055f235','en','zh','We were in class','\n\n','C2'),
            ('085747b65094d52f','zh','en','2017','\n\u3000','1'),
            ('085747b65094d52f','zh','en','2017','\n\u3000','6'),
            ('mixed-control','en','zh',None,None,'A1')):
            if case == 'mixed-control':
                text='She said «bonjour, mon ami», then reluctantly departed.'
            else:
                root,snapshot,_=load_book(case)
                manifest=json.loads((root/'manifest.json').read_text())
                for name in (*manifest['files'],'manifest.json'):
                    path=root/name; hashes[path]=hashlib.sha256(path.read_bytes()).hexdigest()
                start=snapshot.canonical_text.index(anchor)
                end=snapshot.canonical_text.index(separator,start)
                text=snapshot.canonical_text[start:end]
            call_start=len(provider.calls)
            with tempfile.TemporaryDirectory() as temporary:
                docs,prefs,repo=repositories(Path(temporary))
                docs.save_document(excerpt_document(text,source))
                workspace=prefs.save_workspace(target,expected_revision=0,mutation_id='native')
                scheme=LanguageProficiencyRegistry.get_scheme(source)
                book=prefs.save_book('excerpt',BookSettings('language_learning',source,(
                    LanguageLearningSettings(source,scheme.scheme_id,scheme.version,level),)),
                    expected_revision=0,expected_workspace_revision=workspace.revision,
                    resolved_source_language=source,mutation_id='mode')
                blocks=repo.read_sources('excerpt').blocks
                if len(blocks)!=1: raise ValueError('acceptance_excerpt_not_single_block')
                block=blocks[0]
                sentences=segment_learning_block(source=block.source,effective_text=block.effective_text,source_language=source)
                prepared=service.prepare(sentences=sentences,effective_text=block.effective_text,
                                         source_language=source,explanation_language=target,
                                         proficiency=LearningProficiencyRef(scheme.scheme_id,scheme.version,level))
                binding=GenerationBinding(prepared.key,workspace.revision,book.revision)
                entry=repo.claim('excerpt',GenerationCommand('generate',binding),sentences=sentences,
                                 plan=prepared.plan,now_ms=now(),lease_until_ms=now()+600000).next_entry
                def write(action,**kw):
                    return repo.write_operation('excerpt',OperationWrite(entry.operation.operation_id,
                        entry.revision,entry.last_fence,now(),action,**kw))
                try:
                    for index in range(len(prepared.plan.groups)):
                        entry=write('dispatch')
                        batch=service.generate_batch(prepared,entry.operation,now_ms=now())
                        entry=write('checkpoint',batch_index=index,batch=batch)
                    result=service.assemble(prepared,entry.operation,
                        repo.read_checkpoints('excerpt',entry.operation.operation_id))
                    entry=write('commit',result=result)
                except LearningHintsError as error:
                    entry=write('fail',reason='invalid_output' if error.outcome_known else 'provider_outcome_unknown',
                                outcome_known=error.outcome_known)
                    raise
                assert entry.saved==repo.read_batch('excerpt',(prepared.key,))[0].view.saved
                reuse=repo.claim('excerpt',GenerationCommand('reuse',binding),sentences=sentences,
                    plan=prepared.plan,now_ms=now(),lease_until_ms=now()+600000)
                assert reuse.outcome=='reuse'
                if case=='mixed-control':
                    # The entire mixed sentence is excluded, including English words.
                    assert not result.items
                report['cases'].append({'case':case,'level':level,'reuse':reuse.outcome,
                    'provider_calls':provider.calls[call_start:],'source_text':text,'source_language':source,'target_language':target,
                    'sentences':sentences.to_dict(),'result':result.to_dict(),'batches':len(prepared.plan.groups),
                    'operation_state':entry.operation.state})
                print(f'PASS {case}: {len(result.items)} hints at level {level}, {len(prepared.plan.groups)} batches',flush=True)
        report['status']='checks_passed'
    except Exception as error:
        report['error_code']=error.code if isinstance(error,LearningHintsError) else type(error).__name__
        print('Acceptance failed: '+report['error_code'],flush=True)
    finally:
        report['provider_call_count']=len(provider.calls)
        report['provider_evidence']=provider.calls
        report['backups_unchanged']=all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in hashes.items())
        preserve(directory/'report.json',json.dumps(report,ensure_ascii=False,indent=2).encode())
        print('Evidence: '+str(directory/'report.json'),flush=True)
    if report['status']!='checks_passed' or not report['backups_unchanged']:
        raise SystemExit(1)


if __name__=='__main__': run()
