"""LLRN-14 strict JSON/OpenAPI and pure domain-to-transport contract tests."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

import api_schemas as public
import api_learning_schemas as api
import api_learning_projection as projection
from learning import LearningService, LearningBlockView, LearningPage, LearningStatus, LearningSubmission
from document_structure.learning_repository import LearningBlock, SavedLearning, OperationSnapshot
from language.language_profile_registry import LanguageProfileRegistry
from language.language_proficiency_registry import LanguageProficiencyRegistry
from reading_preferences.effective import project_preferences
from reading_preferences.models import (WorkspaceScope, WorkspacePreferences, BookPreferences, BookSettings,
    LanguageLearningSettings, PreservedModeSettings)
from reading_preferences.profile_source import SourceLanguage, ExistingProfileLanguageReader
from reading_preferences.service import ReadingPreferencesService
from scripts import test_learning_contracts as fixtures
from scripts.test_learning_file_repository import repositories, fixture, configure
from scripts.test_learning_translation_service import Provider as TranslationProvider
from scripts.test_learning_hints_service import Provider as HintsProvider
from section_tasks.learning_cache import TranslationKey, HintsKey
from section_tasks.learning_generation_policy import (ReadView, BatchPlan, GenerationBinding,
    GenerationCommand, GenerationOperation, CommandReceipt)
from section_tasks.learning_translation_service import LearningTranslationService
from section_tasks.learning_hints_service import LearningHintsService
from shared.learning_source import LearningContractError


def selection(code='en'):
    scheme=LanguageProficiencyRegistry.get_scheme(code)
    return dict(mode_id='language_learning',version='1',language_code=code,
                scheme_id=scheme.scheme_id,scheme_version=scheme.version,level_id=scheme.levels[0].level_id)


def settings():
    scope=WorkspaceScope('default')
    workspace=WorkspacePreferences(scope,2,'zh')
    book=BookPreferences(scope,'book',3,BookSettings('language_learning',None,
        (LanguageLearningSettings('en','cefr_reading','1','B1'),)))
    return project_preferences(workspace,book,SourceLanguage('en','profile'))


def page(*,ready=True,empty_hints=False,refresh=None):
    sentences=fixtures.resource();trans=fixtures.translation(sentences);hints=fixtures.hints(sentences)
    if empty_hints:hints=replace(hints,items=())
    key=TranslationKey(sentences.reference,'en','zh','translation-v1','prompt-v1')
    hkey=HintsKey(sentences.reference,'en','zh','hints-v1','prompt-v1',hints.proficiency)
    return LearningPage(settings(),(LearningBlockView(LearningBlock(sentences.source,fixtures.TEXT),sentences,(
        SavedLearning(key,ReadView('ready' if ready else 'missing',trans if ready else None,refresh,'refresh' if refresh else None),sentences),
        SavedLearning(hkey,ReadView('ready' if ready else 'missing',hints if ready else None,None,None),sentences))),))


def generate():
    return dict(doc_name='book',content_block_id='unit:content:0',component='translation',command_id='new-command',
        expected_source_token=api.learning_source_token(fixtures.binding()),expected_workspace_revision=2,expected_book_revision=3)


def apply():
    return dict(doc_name='book',expected_revision=3,expected_workspace_revision=2,mutation_id='apply',selection=selection())


def status(state='claimed',**kw):
    key=page().blocks[0].components[0].key
    op=GenerationOperation('op',GenerationBinding(key,2,3),1,0,
        BatchPlan((tuple(s.sentence_id for s in fixtures.resource().sentences),)),state,1,1,100,
        outcome_uncertain=state in ('running','unknown'),**kw)
    return LearningStatus(OperationSnapshot(op,1),state,True,True)


class LearningTransportTests(unittest.TestCase):
    def reject(self,cls,payload):
        with self.assertRaises(ValidationError):cls.model_validate(payload)
        with self.assertRaises(ValidationError):cls.model_validate_json(json.dumps(payload))

    def roundtrip(self,value):
        self.assertEqual(type(value).model_validate_json(value.model_dump_json()),value)
        return value

    def test_public_module_exports_without_routes_or_runtime(self):
        self.assertIs(public.LearningReadRequest,api.LearningReadRequest)
        self.assertIs(public.ReadingBookApplyRequest,api.ReadingBookApplyRequest)
        self.roundtrip(api.LearningGenerateRequest.model_validate(generate()))

    def test_all_requests_reject_forbidden_authority_and_model_fields(self):
        cases=[(api.LearningGenerateRequest,generate()),(api.ReadingBookApplyRequest,apply()),
            (api.LearningReadRequest,dict(doc_name='book',content_block_ids=['block'])),
            (api.ReadingWorkspaceSaveRequest,dict(native_language='zh',expected_revision=0,mutation_id='save')),
            (api.LearningOperationRequest,dict(handle=dict(doc_name='book',operation_id='op')))]
        for cls,payload in cases:
            for field in ('text','source','source_language','target_language','chapter_id','parent_plan','user_id',
                          'workspace_id','namespace','model','max_output_tokens','temperature'):
                with self.subTest(cls=cls.__name__,field=field):self.reject(cls,{**payload,field:'client-authority'})

    def test_strict_revisions_bools_and_ids_do_not_coerce(self):
        for field,values in {'expected_workspace_revision':[True,'2',2.0,-1],
            'expected_book_revision':[False,'3',None], 'command_id':[1,False,'',' x','x\n']}.items():
            for value in values:self.reject(api.LearningGenerateRequest,{**generate(),field:value})
        for value in ('true',1,None):
            self.reject(api.LearningGenerateRequest,{**generate(),'acknowledge_possible_duplicate_cost':value})

    def test_exact_document_names_reject_paths_controls_and_normalization(self):
        for value in ('..','.','/tmp/book','a/b','a\\b',' book','book ','a\n','\ud800'):
            self.reject(api.ReadingBookReadRequest,{'doc_name':value})
        self.roundtrip(api.ReadingBookReadRequest(doc_name='中國文化【新版】'))

    def test_read_bounds_and_duplicates(self):
        for ids in ([],['x']*2,['x'+str(i) for i in range(51)],'x'):
            self.reject(api.LearningReadRequest,dict(doc_name='book',content_block_ids=ids))
        self.assertEqual(len(api.LearningReadRequest(doc_name='book',content_block_ids=[str(i) for i in range(50)]).content_block_ids),50)

    def test_registry_all_languages_and_proficiency_schemes(self):
        for code in LanguageProfileRegistry.get_supported_language_codes():
            if code=='unknown':continue
            self.roundtrip(api.LearningModeSelection(**selection(code)))
        options=self.roundtrip(projection.reading_options_response())
        self.assertEqual(set(options.language_codes),{s.language_code for s in options.proficiency_schemes})
        self.assertNotIn('unknown',options.language_codes)

    def test_registry_language_unknown_aliases_not_accepted(self):
        for code in ('unknown','xx','EN','en-US','中文',' en'):
            self.reject(api.ReadingWorkspaceSaveRequest,dict(native_language=code,expected_revision=0,mutation_id='save'))
        self.roundtrip(api.ReadingWorkspaceSaveRequest(native_language=None,expected_revision=0,mutation_id='save'))

    def test_registry_changes_are_consulted_at_validation_time(self):
        with patch.object(LanguageProfileRegistry,'get_supported_language_codes',return_value=('zh',)):
            self.reject(api.ReadingWorkspaceSaveRequest,dict(native_language='en',expected_revision=0,mutation_id='save'))

    def test_wrong_scheme_level_version_and_extra_settings_rejected(self):
        for field,value in [('scheme_id','jlpt_reading'),('scheme_version','99'),('level_id','N5'),
                            ('version','2'),('mode_id','unknown'),('model','x')]:
            self.reject(api.ReadingBookApplyRequest,{**apply(),'selection':{**selection(),field:value}})

    def test_normal_mode_needs_no_learning_fields(self):
        request=api.ReadingBookApplyRequest(**{**apply(),'selection':dict(mode_id='normal',version='1')})
        self.assertEqual(request.to_settings(settings().book).active_mode,'normal')
        self.reject(api.ReadingBookApplyRequest,{**apply(),'selection':dict(mode_id='normal',version='1',level_id='B1')})

    def test_preference_mutation_ids_match_domain_identifier_contract(self):
        for value in ('..','a/b','a\\b'):
            self.reject(api.ReadingBookApplyRequest,{**apply(),'mutation_id':value})
            self.reject(api.ReadingWorkspaceSaveRequest,dict(native_language='zh',expected_revision=0,mutation_id=value))

    def test_apply_preserves_other_modes_and_omitted_override(self):
        previous=replace(settings().book,settings=BookSettings('normal','en',(
            PreservedModeSettings('future','2','{"private":"data"}'),settings().book.settings.modes[0])))
        request=api.ReadingBookApplyRequest(**apply())
        saved=request.to_settings(previous)
        self.assertEqual(saved.modes[0],previous.settings.modes[0]);self.assertEqual(saved.source_language_override,'en')
        self.assertEqual(saved.modes[1].level_id,'A1')
        self.assertEqual(previous.settings.modes[1].level_id,'B1')
        cleared=api.ReadingBookApplyRequest(**{**apply(),'source_language_override':None}).to_settings(previous)
        self.assertIsNone(cleared.source_language_override)

    def test_apply_merge_requires_current_book_revision(self):
        request=api.ReadingBookApplyRequest(**apply())
        for previous in (replace(settings().book,document_id='other'),replace(settings().book,revision=4)):
            with self.assertRaises(LearningContractError):request.to_settings(previous)

    def test_override_selection_language_must_match(self):
        self.reject(api.ReadingBookApplyRequest,{**apply(),'source_language_override':'ja'})

    def test_same_language_is_ineligible_using_authoritative_workspace(self):
        current=settings();same=project_preferences(replace(current.workspace,native_language='en'),current.book,current.source)
        response=self.roundtrip(projection.book_response(same))
        self.assertIsNone(response.effective_mode)
        self.assertIn('same_language',[r.code for r in response.active_reasons])
        self.reject(api.ReadingBookResponse,{**response.model_dump(),'effective_mode':'language_learning','active_reasons':[]})

    def test_unknown_persisted_mode_is_readable_and_has_no_arbitrary_payload(self):
        current=settings();book=replace(current.book,settings=BookSettings('future',None,(
            PreservedModeSettings('future','2','{"secret":"not-for-client"}'),)))
        result=self.roundtrip(projection.book_response(project_preferences(current.workspace,book,current.source)))
        self.assertIsNone(result.effective_mode);self.assertFalse(result.saved_settings[0].supported)
        self.assertNotIn('secret',result.model_dump_json())
        self.assertEqual(result.active_mode,'future')

    def test_stale_known_settings_are_described_without_silent_repair(self):
        current=settings();old=replace(current.book.settings.modes[0],scheme_version='old')
        book=replace(current.book,settings=replace(current.book.settings,modes=(old,)))
        response=projection.book_response(project_preferences(current.workspace,book,current.source))
        self.assertFalse(response.saved_settings[0].supported)
        self.assertEqual(book.settings.modes[0].scheme_version,'old')

    def test_unknown_mode_and_arbitrary_settings_cannot_be_submitted(self):
        for selection_value in (dict(mode_id='future',version='2'),dict(mode_id='normal',version='1',payload={}),{}):
            self.reject(api.ReadingBookApplyRequest,{**apply(),'selection':selection_value})
        self.reject(api.ReadingBookApplyRequest,{**apply(),'saved_settings':[dict(mode_id='future',payload={})]})

    def test_source_token_is_deterministic_and_binds_every_source_field(self):
        binding=fixtures.binding();expected=api.learning_source_token(binding)
        self.assertEqual(len(expected),64)
        changes=dict(namespace='other',document_id='other',chapter_id='other',section_id='other',task_unit_id='other',
            content_block_id='other',structure_version=2,document_epoch='new',content_epoch='new',effective_content_hash='0'*64)
        request=api.LearningGenerateRequest(**generate());request.verify_source(binding)
        for field,value in changes.items():
            with self.subTest(field=field):
                other=replace(binding,**{field:value});self.assertNotEqual(api.learning_source_token(other),expected)
                with self.assertRaises(LearningContractError):request.verify_source(other)

    def test_source_expectation_does_not_accept_internal_binding(self):
        self.reject(api.LearningGenerateRequest,{**generate(),'expected_source':fixtures.binding().to_dict()})
        for token in ('x'*64,'A'*64,'0'*63):
            self.reject(api.LearningGenerateRequest,{**generate(),'expected_source_token':token})

    def test_retry_requires_matching_book_new_command_and_explicit_target(self):
        payload={**generate(),'intent':'retry','retry_operation':dict(doc_name='book',operation_id='old')}
        self.roundtrip(api.LearningGenerateRequest(**payload))
        for handle in (None,dict(doc_name='other',operation_id='old'),dict(doc_name='book',operation_id='new-command')):
            self.reject(api.LearningGenerateRequest,{**payload,'retry_operation':handle})
        self.reject(api.LearningGenerateRequest,{**generate(),'retry_operation':payload['retry_operation']})
        self.reject(api.LearningGenerateRequest,{**generate(),'acknowledge_possible_duplicate_cost':True})

    def test_operation_handle_forbids_client_namespace_and_cross_book_projection(self):
        self.reject(api.LearningOperationRequest,dict(handle=dict(doc_name='book',operation_id='op',namespace='other')))
        with self.assertRaises(LearningContractError):
            projection.operation_response(api.LearningOperationHandle(doc_name='other',operation_id='op'),status())

    def test_partial_pagination_is_bounded_and_strict(self):
        payload=dict(handle=dict(doc_name='book',operation_id='op'))
        for field,value in [('limit',0),('limit',101),('offset',-1),('offset',True),('limit','50')]:
            self.reject(api.LearningPartialReadRequest,{**payload,field:value})
        self.roundtrip(api.LearningPartialReadRequest(**payload,offset=100,limit=100))

    def test_page_projection_roundtrip_preserves_unicode_and_whitespace(self):
        result=self.roundtrip(projection.learning_read_response(page()))
        parts=result.blocks[0].sentences
        self.assertEqual(''.join(s.text for s in parts),fixtures.TEXT)
        self.assertEqual(parts[1].end-parts[1].start,8)
        self.assertIsNone(parts[0].sentence_id)
        self.assertNotIn('namespace',result.model_dump_json());self.assertNotIn('chapter_id',result.model_dump_json())

    def test_empty_saved_hints_are_ready_not_missing(self):
        result=projection.learning_read_response(page(empty_hints=True))
        hints=result.blocks[0].components[1]
        self.assertEqual(hints.state,'ready');self.assertEqual(hints.saved.items,[])

    def test_missing_and_saved_refresh_states_are_independent(self):
        self.assertEqual(projection.learning_read_response(page(ready=False)).blocks[0].components[0].state,'missing')
        for state in ('claimed','running','partial','unknown','failed','stale'):
            view=projection.learning_read_response(page(refresh=state)).blocks[0].components[0]
            self.assertEqual(view.state,'ready');self.assertEqual(view.operation_state,state)
            self.assertEqual(view.saved.result_revision,1)

    def test_pending_unknown_partial_failed_without_saved_are_distinct(self):
        for raw,wanted in (('claimed','pending'),('running','pending'),('partial','partial'),('failed','failed'),('unknown','unknown'),('stale','stale')):
            data=page(ready=False);block=data.blocks[0];saved=block.components[0]
            changed=replace(saved,view=ReadView(raw,None,raw,'op'))
            result=projection.learning_read_response(replace(data,blocks=(replace(block,components=(changed,block.components[1])),)))
            self.assertEqual(result.blocks[0].components[0].state,wanted)

    def test_ready_requires_saved_and_nonready_cannot_claim_saved(self):
        payload=projection.learning_read_response(page()).blocks[0].components[0].model_dump()
        self.reject(api.LearningComponentView,{**payload,'saved':None})
        self.reject(api.LearningComponentView,{**payload,'state':'missing'})
        self.reject(api.LearningComponentView,{**payload,'component':'hints'})

    def test_misaligned_duplicate_or_fabricated_translations_rejected(self):
        value=projection.learning_read_response(page()).model_dump()
        for rows in ([],value['blocks'][0]['components'][0]['saved']['rows'][::-1]):
            bad=copy.deepcopy(value);bad['blocks'][0]['components'][0]['saved']['rows']=rows
            self.reject(api.LearningReadResponse,bad)

    def test_hint_span_and_source_form_are_verified(self):
        value=projection.learning_read_response(page()).model_dump()
        for field,new in [('source_form','wrong'),('span',dict(start=1,end=7)),('sentence_id','unknown')]:
            bad=copy.deepcopy(value);bad['blocks'][0]['components'][1]['saved']['items'][0][field]=new
            self.reject(api.LearningReadResponse,bad)

    def test_cross_book_and_language_results_rejected(self):
        value=projection.learning_read_response(page(refresh='running')).model_dump()
        bad=copy.deepcopy(value);bad['blocks'][0]['components'][0]['operation']['doc_name']='other'
        self.reject(api.LearningReadResponse,bad)
        bad=copy.deepcopy(value);bad['blocks'][0]['components'][0]['saved']['explanation_language']='ja'
        self.reject(api.LearningReadResponse,bad)

    def test_hints_for_another_level_cannot_be_current_results(self):
        bad=projection.learning_read_response(page()).model_dump()
        bad['blocks'][0]['components'][1]['saved']['proficiency']['level_id']='C2'
        self.reject(api.LearningReadResponse,bad)

    def test_effective_mode_requires_consistent_eligibility_and_saved_selection(self):
        value=projection.book_response(settings()).model_dump()
        self.reject(api.ReadingBookResponse,{**value,'saved_settings':[]})
        bad=copy.deepcopy(value);bad['modes'][1]['eligible']=False
        bad['modes'][1]['reasons']=[dict(code='proficiency_required',field='proficiency',message='Choose a level.')]
        self.reject(api.ReadingBookResponse,bad)

    def test_normal_mode_returns_source_handles_without_learning_projection(self):
        data=page();current=data.settings
        normal=project_preferences(current.workspace,replace(current.book,settings=BookSettings()),current.source)
        result=projection.learning_read_response(LearningPage(normal,(replace(data.blocks[0],sentences=None,components=()),)))
        self.roundtrip(result);self.assertIsNone(result.blocks[0].sentences)

    def test_operation_status_missing_running_unknown_stale_and_superseded(self):
        handle=api.LearningOperationHandle(doc_name='book',operation_id='op')
        self.assertEqual(projection.operation_response(handle,LearningStatus(None,'missing',False,False)).state,'missing')
        for state in ('claimed','running','unknown','failed','stale'):
            response=self.roundtrip(projection.operation_response(handle,status(state)))
            self.assertEqual(response.state,state)
        stale=replace(status('running'),state='stale',settings_current=False)
        result=projection.operation_response(handle,stale)
        self.assertEqual(result.durable_state,'running');self.assertTrue(result.outcome_uncertain)
        superseded=replace(stale,snapshot=replace(stale.snapshot,entry_revision=None))
        self.assertFalse(projection.operation_response(handle,superseded).owns_current_slot)

    def test_submission_cannot_claim_committed_revision_before_commit(self):
        payload=dict(doc_name='book',command_id='op',outcome='claimed',operation=dict(doc_name='book',operation_id='op'))
        self.roundtrip(api.LearningSubmissionResponse(**payload))
        self.reject(api.LearningSubmissionResponse,{**payload,'result_revision':1})
        self.reject(api.LearningSubmissionResponse,{**payload,'outcome':'succeeded'})
        self.reject(api.LearningSubmissionResponse,{**payload,'operation':None})

    def test_operation_cannot_hide_unknown_cost_or_claim_incomplete_success(self):
        handle=api.LearningOperationHandle(doc_name='book',operation_id='op')
        value=projection.operation_response(handle,status('running')).model_dump()
        self.reject(api.LearningOperationResponse,{**value,'outcome_uncertain':False})
        self.reject(api.LearningOperationResponse,{**value,'completed_batches':1})
        succeeded={**value,'state':'succeeded','durable_state':'succeeded','outcome_uncertain':False,'result_revision':1}
        self.reject(api.LearningOperationResponse,succeeded)
        self.roundtrip(api.LearningOperationResponse(**{**succeeded,'completed_batches':1}))

    def test_denied_and_reuse_submissions_have_distinct_evidence(self):
        base=dict(doc_name='book',command_id='op',operation=None)
        self.roundtrip(api.LearningSubmissionResponse(**base,outcome='denied',reason='future_restriction'))
        self.reject(api.LearningSubmissionResponse,{**base,'outcome':'denied'})
        self.reject(api.LearningSubmissionResponse,{**base,'outcome':'reuse'})
        self.roundtrip(api.LearningSubmissionResponse(**base,outcome='reuse',result_revision=2))

    def test_receipt_projection_reuse_has_original_revision_and_no_fake_operation(self):
        binding=status().snapshot.operation.binding
        receipt=CommandReceipt(GenerationCommand('reuse',binding),'reuse',None,4)
        response=projection.command_lookup_response('book','reuse',receipt)
        self.assertEqual(response.receipt.result_revision,4);self.assertIsNone(response.receipt.operation)
        self.roundtrip(response)
        with self.assertRaises(LearningContractError):projection.command_lookup_response('other','reuse',receipt)
        self.assertFalse(projection.command_lookup_response('book','absent',None).found)

    def test_unknown_submission_requires_lookup_not_fake_success(self):
        result=self.roundtrip(projection.submission_response('book',LearningSubmission('unknown','op',None)))
        self.assertIsNone(result.result_revision);self.assertIsNone(result.operation)

    def test_error_union_is_discriminated_content_free_and_strict(self):
        for kind,code in [('invalid_request','learning_request_invalid'),('conflict','learning_source_changed'),
                          ('unavailable','learning_read_failed'),('not_found','learning_block_missing')]:
            self.roundtrip(api.LearningErrorResponse(detail=dict(kind=kind,code=code)))
            self.reject(api.LearningErrorResponse,dict(detail=dict(kind=kind,code=code,body='private provider text')))
        self.reject(api.LearningErrorResponse,dict(detail=dict(kind='conflict',code='made_up')))
        self.reject(api.LearningErrorResponse,dict(detail=dict(kind='unknown',code='learning_commit_unknown',recovery='read_preferences')))
        self.roundtrip(api.LearningErrorResponse(detail=dict(kind='unknown',code='learning_commit_unknown',recovery='lookup_command')))

    def test_eligibility_errors_keep_field_reasons_and_reject_unknown_reason_codes(self):
        detail=dict(kind='ineligible',code='learning_mode_unavailable',reasons=[
            dict(code='same_language',field='source_language',message='Choose a different original language.')])
        self.roundtrip(api.LearningErrorResponse(detail=detail))
        self.reject(api.LearningErrorResponse,dict(detail={**detail,'reasons':[]}))
        detail['reasons'][0]['code']='provider_body'
        self.reject(api.LearningErrorResponse,dict(detail=detail))


CONTRACTS = (
    ('GET','/reading-preferences/options',None,api.ReadingOptionsResponse),
    ('GET','/reading-preferences/workspace',None,api.ReadingWorkspaceResponse),
    ('PUT','/reading-preferences/workspace',api.ReadingWorkspaceSaveRequest,api.ReadingWorkspaceResponse),
    ('POST','/documents/reading-preferences/read',api.ReadingBookReadRequest,api.ReadingBookResponse),
    ('POST','/documents/reading-preferences/apply',api.ReadingBookApplyRequest,api.ReadingBookResponse),
    ('POST','/documents/reading-interactions/learning/read',api.LearningReadRequest,api.LearningReadResponse),
    ('POST','/documents/reading-interactions/learning/generate',api.LearningGenerateRequest,api.LearningSubmissionResponse),
    ('POST','/documents/reading-interactions/learning/operations/read',api.LearningOperationRequest,api.LearningOperationResponse),
    ('POST','/documents/reading-interactions/learning/operations/reconcile',api.LearningOperationRequest,api.LearningSubmissionResponse),
    ('POST','/documents/reading-interactions/learning/operations/partial',api.LearningPartialReadRequest,api.LearningPartialReadResponse),
    ('POST','/documents/reading-interactions/learning/commands/read',api.LearningCommandLookupRequest,api.LearningCommandLookupResponse),
)


def contract_app():
    # Test-only shape harness. This does not install production routes or DI.
    app=FastAPI();calls=[]
    @app.exception_handler(RequestValidationError)
    async def invalid_request(request,error):
        code=('preference_request_invalid' if 'reading-preferences' in request.url.path else 'learning_request_invalid')
        response=api.LearningErrorResponse(detail=dict(kind='invalid_request',code=code))
        return JSONResponse(status_code=422,content=response.model_dump(mode='json'))
    for method,path,request,response in CONTRACTS:
        def endpoint(body):
            calls.append(body)
            return projection.book_response(settings())
        def empty_endpoint(): return projection.reading_options_response()
        if request is not None:
            endpoint.__annotations__={'body':request};fn=endpoint
        else:fn=empty_endpoint
        app.add_api_route(path,fn,methods=[method],response_model=response,
            responses={code:{'model':api.LearningErrorResponse} for code in (400,404,409,422,503)})
    return app,calls


class LearningOpenApiTests(unittest.TestCase):
    def setUp(self):self.app,self.calls=contract_app();self.schema=self.app.openapi()

    def test_all_eleven_operations_have_typed_bodies_and_responses(self):
        self.assertEqual(sum(len(p) for p in self.schema['paths'].values()),11)
        for method,path,request,response in CONTRACTS:
            operation=self.schema['paths'][path][method.lower()]
            if request:
                self.assertEqual(operation['requestBody']['content']['application/json']['schema']['$ref'],
                                 '#/components/schemas/'+request.__name__)
            else:self.assertNotIn('requestBody',operation)
            self.assertEqual(operation['responses']['200']['content']['application/json']['schema']['$ref'],
                             '#/components/schemas/'+response.__name__)

    def test_request_schemas_expose_bounds_strict_extras_and_discriminators(self):
        schemas=self.schema['components']['schemas']
        self.assertFalse(schemas['LearningGenerateRequest']['additionalProperties'])
        self.assertEqual(schemas['LearningReadRequest']['properties']['content_block_ids']['maxItems'],50)
        self.assertEqual(schemas['ReadingBookApplyRequest']['properties']['selection']['discriminator']['propertyName'],'mode_id')
        self.assertEqual(schemas['LearningErrorResponse']['properties']['detail']['discriminator']['propertyName'],'kind')
        for field in ('text','namespace','chapter_id','model','workspace_id','source_language'):
            self.assertNotIn(field,schemas['LearningGenerateRequest']['properties'])

    def test_http_json_invalid_requests_do_not_reach_handler(self):
        with TestClient(self.app) as client:
            route='/documents/reading-interactions/learning/generate'
            for payload in ({**generate(),'model':'x'},{**generate(),'expected_book_revision':True},
                            {**generate(),'intent':'retry','retry_operation':dict(doc_name='other',operation_id='old')}):
                response=client.post(route,json=payload)
                self.assertEqual(response.status_code,422)
                self.assertEqual(api.LearningErrorResponse.model_validate(response.json()).detail.kind,'invalid_request')
                self.assertNotIn('input',response.json()['detail'])
        self.assertEqual(self.calls,[])

    def test_http_normal_mode_needs_no_language_fields(self):
        with TestClient(self.app) as client:
            result=client.post('/documents/reading-preferences/apply',json={**apply(),
                'selection':dict(mode_id='normal',version='1')})
        self.assertEqual(result.status_code,200);self.assertEqual(len(self.calls),1)
        self.assertEqual(self.calls[0].selection.mode_id,'normal')

    def test_real_options_endpoint_roundtrips_registry_catalog(self):
        with TestClient(self.app) as client:result=client.get('/reading-preferences/options')
        self.assertEqual(result.status_code,200)
        self.assertEqual(api.ReadingOptionsResponse.model_validate(result.json()),projection.reading_options_response())


class LearningSchemaServiceTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.docs,self.prefs,self.repo=repositories(self.root);self.docs.save_document(fixture());configure(self.prefs)
        self.preferences=ReadingPreferencesService(self.prefs,ExistingProfileLanguageReader(
            lambda scope,doc:SimpleNamespace(document_language_code='en')))
        self.tp,self.hp=TranslationProvider(),HintsProvider()
        self.service=LearningService(self.repo,self.preferences,LearningTranslationService(self.tp,provider_name='test'),
            LearningHintsService(self.hp,provider_name='test'),lease_ms=60000)
        self.addCleanup(self.service.shutdown)
        self.block=self.repo.read_sources('book').blocks[0]

    def request(self,component='translation',command_id='generate',intent='get_or_generate'):
        current=self.preferences.read_book('book')
        request=api.LearningGenerateRequest.model_validate_json(json.dumps(dict(doc_name='book',
            content_block_id=self.block.source.content_block_id,component=component,command_id=command_id,intent=intent,
            expected_source_token=api.learning_source_token(self.block.source),
            expected_workspace_revision=current.workspace.revision,expected_book_revision=current.book.revision)))
        source,=self.repo.resolve_blocks(request.doc_name,(request.content_block_id,));request.verify_source(source.source)
        return self.service.request(request.doc_name,request.content_block_id,request.component,command_id=request.command_id,
            expected_source=source.source,expected_workspace_revision=request.expected_workspace_revision,
            expected_book_revision=request.expected_book_revision,intent=request.intent)

    def read(self):
        request=api.LearningReadRequest(doc_name='book',content_block_ids=[self.block.source.content_block_id])
        return projection.learning_read_response(self.service.read(request.doc_name,tuple(request.content_block_ids)))

    def snapshot(self):return {str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}

    def test_read_projection_performs_no_writes_or_generation(self):
        before=self.snapshot();response=self.read()
        self.assertEqual(response.blocks[0].components[0].state,'missing')
        self.assertEqual(self.snapshot(),before);self.assertFalse(self.tp.calls);self.assertFalse(self.hp.calls)

    def test_generate_persist_read_reuse_and_command_lookup_contract(self):
        result=self.request();submitted=projection.submission_response('book',result)
        self.assertEqual(submitted.outcome,'claimed');self.assertIsNone(submitted.result_revision)
        self.request('hints','hints');self.service.shutdown()
        result=self.read();self.assertEqual([v.state for v in result.blocks[0].components],['ready','ready'])
        self.assertEqual(result.blocks[0].components[1].saved.items,[])
        reuse=self.request(command_id='reuse');receipt=self.service.lookup_command('book','reuse')
        response=projection.submission_response('book',reuse,receipt=receipt)
        self.assertEqual(response.result_revision,1)
        self.assertEqual(projection.command_lookup_response('book','reuse',receipt).receipt.result_revision,1)
        self.assertEqual(len(self.tp.calls),1);self.assertEqual(len(self.hp.calls),1)
        self.assertEqual(api.LearningReadResponse.model_validate_json(result.model_dump_json()),result)

    def test_reconcile_succeeded_requires_durable_commit_proof(self):
        self.request();self.service.shutdown()
        result=self.service.reconcile('book','generate')
        response=projection.submission_response('book',result,status=self.service.status('book','generate'))
        self.assertEqual(response.outcome,'succeeded');self.assertEqual(response.result_revision,1)
        with self.assertRaises(LearningContractError):projection.submission_response('book',result)

    def test_unknown_provider_operation_is_not_serialized_as_missing(self):
        self.tp.failure=TimeoutError('private provider body');self.request();self.service.shutdown()
        response=self.read();self.assertEqual(response.blocks[0].components[0].state,'unknown')
        handle=api.LearningOperationHandle(doc_name='book',operation_id='generate')
        operation=projection.operation_response(handle,self.service.status('book','generate'))
        self.assertTrue(operation.outcome_uncertain);self.assertNotIn('private provider body',operation.model_dump_json())

    def test_translation_partial_checkpoint_is_separate_from_saved_result(self):
        from document_structure.learning_repository import LearningCommitUnknown
        write=self.repo.write_operation
        def lose_ack(doc,command,**kwargs):
            result=write(doc,command,**kwargs)
            if command.action=='checkpoint':raise LearningCommitUnknown()
            return result
        with patch.object(self.repo,'write_operation',side_effect=lose_ack):self.request();self.service.shutdown()
        handle=api.LearningOperationHandle(doc_name='book',operation_id='generate')
        response=projection.partial_response(handle,self.service.status('book','generate'),
            self.service.read_partial('book','generate'))
        self.assertEqual(response.completed_batches,1);self.assertIsNone(response.next_offset)
        self.assertNotIn('result_revision',response.model_dump_json())
        self.assertEqual(self.read().blocks[0].components[0].state,'partial')
        self.assertEqual(api.LearningPartialReadResponse.model_validate_json(response.model_dump_json()),response)
        bad=response.model_dump();bad['batches'][0]['batch_index']=1
        with self.assertRaises(ValidationError):api.LearningPartialReadResponse.model_validate(bad)

    def test_empty_hints_checkpoint_has_explicit_batch_and_is_not_missing(self):
        self.request('hints','hints');self.service.shutdown()
        handle=api.LearningOperationHandle(doc_name='book',operation_id='hints')
        response=projection.partial_response(handle,self.service.status('book','hints'),self.service.read_partial('book','hints'))
        self.assertEqual(response.batches[0].items,[]);self.assertEqual(response.completed_batches,1)

    def test_partial_projection_rejects_wrong_book_or_stale_binding(self):
        self.request();self.service.shutdown();state=self.service.status('book','generate')
        for handle,current in ((api.LearningOperationHandle(doc_name='other',operation_id='generate'),state),
            (api.LearningOperationHandle(doc_name='book',operation_id='generate'),replace(state,source_current=False))):
            with self.assertRaises(LearningContractError):projection.partial_response(handle,current,())

    def test_normal_apply_preserves_saved_learning_settings_and_does_not_generate(self):
        current=self.preferences.read_book('book')
        request=api.ReadingBookApplyRequest(doc_name='book',expected_revision=current.book.revision,
            expected_workspace_revision=current.workspace.revision,mutation_id='normal',
            selection=api.NormalModeSelection(mode_id='normal',version='1'))
        result=self.preferences.save_book(request.doc_name,request.to_settings(current.book),
            expected_revision=request.expected_revision,expected_workspace_revision=request.expected_workspace_revision,
            mutation_id=request.mutation_id)
        response=projection.book_response(result)
        self.assertEqual(response.effective_mode,'normal');self.assertTrue(response.saved_settings[0].supported)
        self.assertFalse(self.tp.calls);self.assertFalse(self.hp.calls)

    def test_apply_same_language_is_rejected_by_authoritative_service(self):
        from reading_preferences.effective import PreferenceValidationError
        workspace=self.preferences.read_workspace()
        self.preferences.save_workspace('en',expected_revision=workspace.revision,mutation_id='native-en')
        current=self.preferences.read_book('book')
        request=api.ReadingBookApplyRequest(doc_name='book',expected_revision=current.book.revision,
            expected_workspace_revision=current.workspace.revision,mutation_id='apply',selection=selection())
        before=self.snapshot()
        with self.assertRaises(PreferenceValidationError):
            self.preferences.save_book(request.doc_name,request.to_settings(current.book),
                expected_revision=request.expected_revision,expected_workspace_revision=request.expected_workspace_revision,
                mutation_id=request.mutation_id)
        self.assertEqual(self.snapshot(),before);self.assertFalse(self.tp.calls)


if __name__=='__main__':unittest.main()
