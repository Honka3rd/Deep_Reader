"""LLRN-03 source identity, Unicode coverage and result reference contracts."""
from dataclasses import FrozenInstanceError, replace
import json
import unittest

from shared.artifact_target_model import ArtifactTargetLevel
from shared.learning_source import (LearningContractError, LearningSourceBinding,
    SentenceResource, SourceSpan, text_hash)
from shared.learning_results import (AlignedTranslation, GenerationProvenance, LearningHints,
    LearningProficiencyRef, LearningResultContext, TranslatedSentence, VocabularyHint)
from shared.task_unit_model import TaskUnit

TEXT = ' \tHello 😀!\r\nCafe\u0301 世界。\n'


def binding(text=TEXT):
    return LearningSourceBinding('default', 'book', 'chapter', 'section', 'unit',
        'unit:content:0', 1, 'document-epoch', 'content-epoch', text_hash(text))


def resource(text=TEXT, source=None, version='sentences-v1', language='en', spans=None):
    if spans is None:
        spans = (SourceSpan(2, 10), SourceSpan(12, len(text) - 1))
    return SentenceResource.from_spans(source=source or binding(text), effective_text=text,
        segmentation_version=version, segmentation_language=language, spans=spans)


def context(sentences):
    return LearningResultContext(sentences.reference, 'en', 'zh', 'translation-v1', 1,
        GenerationProvenance('test-provider', 'test-model', 'prompt-v1'))


def translation(sentences):
    return AlignedTranslation(context(sentences), tuple(
        TranslatedSentence(s.sentence_id, f'譯文 {i}') for i, s in enumerate(sentences.sentences)))


def hints(sentences):
    hint = VocabularyHint(sentences.sentences[0].sentence_id, SourceSpan(2, 7),
        'Hello', 'en', 'word', '你好', 'hello', 'interjection')
    return LearningHints(replace(context(sentences), policy_version='hints-v1'),
        LearningProficiencyRef('cefr_reading', '1', 'B1'), (hint,))


class LearningContractsTests(unittest.TestCase):
    def test_unicode_projection_round_trip_preserves_all_source(self):
        saved = resource()
        decoded = SentenceResource.from_dict(json.loads(json.dumps(saved.to_dict())))
        self.assertEqual(decoded, saved)
        parts = decoded.project(TEXT)
        self.assertEqual(''.join(p.text for p in parts), TEXT)
        self.assertEqual([p.text for p in parts], [' \t', 'Hello 😀!', '\r\n', 'Cafe\u0301 世界。', '\n'])
        self.assertEqual(parts[1].span, SourceSpan(2, 10))
        self.assertEqual(len(parts[1].text), 8)
        self.assertEqual(len(parts[1].text.encode('utf-16-le')) // 2, 9)
        decoded.validate_source(binding(), TEXT)

    def test_multiscript_and_combining_text_never_normalized(self):
        for body in ('e\u0301.', 'é.', '👩🏽\u200d💻!', '日本語。', '한국어!', 'العربية؟', 'ไทย。', '\r\n'):
            with self.subTest(body=body):
                spans = () if body.isspace() else (SourceSpan(0, len(body)),)
                saved = resource(body, spans=spans)
                self.assertEqual(''.join(p.text for p in saved.project(body)), body)
        self.assertNotEqual(text_hash('e\u0301'), text_hash('é'))
        self.assertNotEqual(text_hash('\r\n'), text_hash('\n'))

    def test_empty_and_whitespace_only_source_have_no_synthetic_sentence(self):
        for body in ('', '\n \t'):
            saved = resource(body, spans=())
            self.assertEqual(saved.sentences, ())
            self.assertEqual(''.join(p.text for p in saved.project(body)), body)
            translation(saved).validate_against(saved)
        with self.assertRaisesRegex(LearningContractError, 'sentence_empty'):
            resource(' ', spans=(SourceSpan(0, 1),))

    def test_ids_and_reference_are_deterministic_after_serialization(self):
        saved = resource()
        self.assertEqual(saved, resource())
        self.assertEqual(saved.reference, SentenceResource.from_dict(saved.to_dict()).reference)
        self.assertEqual(len({s.sentence_id for s in saved.sentences}), len(saved.sentences))

    def test_every_source_identity_dimension_fences_same_text(self):
        saved = resource()
        alternatives = dict(namespace='other', document_id='other', chapter_id='other', section_id='other',
            task_unit_id='other', content_block_id='other', structure_version=2,
            document_epoch='new-document', content_epoch='undo-epoch')
        for field, value in alternatives.items():
            with self.subTest(field=field):
                changed = replace(saved.source, **{field: value})
                next_resource = resource(source=changed)
                self.assertNotEqual(saved.reference, next_resource.reference)
                self.assertNotEqual(saved.sentences[0].sentence_id, next_resource.sentences[0].sentence_id)
                with self.assertRaisesRegex(LearningContractError, 'source_changed'):
                    saved.validate_source(changed, TEXT)
                with self.assertRaisesRegex(LearningContractError, 'reference_mismatch'):
                    translation(saved).validate_against(next_resource)

    def test_legacy_none_versions_are_explicit_not_generated(self):
        old = replace(binding(), structure_version=None, document_epoch=None, content_epoch=None)
        saved = resource(source=old)
        self.assertEqual(SentenceResource.from_dict(saved.to_dict()), saved)
        self.assertIsNone(saved.source.content_epoch)
        self.assertNotEqual(saved.reference, resource().reference)

    def test_segmentation_version_language_and_spans_change_reference(self):
        saved = resource()
        for changed in (resource(version='v2'), resource(language='fr'), resource(language=None),
                        resource(spans=(SourceSpan(2, len(TEXT) - 1),))):
            self.assertNotEqual(saved.reference, changed.reference)
            with self.assertRaises(LearningContractError):
                translation(saved).validate_against(changed)

    def test_hash_mismatch_or_normalized_text_rejected(self):
        for text in (TEXT.replace('Hello', 'Hallo'), TEXT.strip(), TEXT.replace('\r\n', '\n')):
            with self.assertRaisesRegex(LearningContractError, 'source_changed'):
                resource().project(text)
        with self.assertRaises(LearningContractError):
            resource(source=replace(binding(), effective_content_hash='0' * 64))

    def test_invalid_spans_overlap_reverse_and_bounds_rejected(self):
        for start, end in ((-1, 1), (1, 1), (2, 1), (False, 2), (0, 1.5), ('0', 2)):
            with self.subTest(start=start, end=end), self.assertRaises(LearningContractError):
                SourceSpan(start, end)
        for spans in ((SourceSpan(2, 11), SourceSpan(10, 20)),
                      (SourceSpan(12, 20), SourceSpan(2, 10)), (SourceSpan(0, 999),)):
            with self.assertRaises(LearningContractError):
                resource(spans=spans)

    def test_uncovered_letters_and_punctuation_are_not_silent_gaps(self):
        for spans in ((), (SourceSpan(3, 10), SourceSpan(12, len(TEXT) - 1)),
                      (SourceSpan(2, 9), SourceSpan(12, len(TEXT) - 1)), (SourceSpan(2, 10),)):
            with self.assertRaisesRegex(LearningContractError, 'uncovered_source'):
                resource(spans=spans)

    def test_forged_sentence_id_rejected_on_decode(self):
        data = resource().to_dict()
        data['sentences'][0]['sentence_id'] = 'provider-supplied-id'
        with self.assertRaisesRegex(LearningContractError, 'sentence_id_invalid'):
            SentenceResource.from_dict(data)

    def test_source_payload_and_numeric_types_are_strict(self):
        for change in ({'namespace': ''}, {'document_id': 1}, {'chapter_id': None},
                       {'structure_version': True}, {'structure_version': 0},
                       {'content_epoch': 1}, {'effective_content_hash': 'x'}, {'scope_kind': 'user'}):
            with self.subTest(change=change), self.assertRaises(LearningContractError):
                replace(binding(), **change)
        for change in ({'source_length': True}, {'source_length': -1}, {'schema_version': 2},
                       {'schema_version': True}, {'segmentation_version': ''}):
            with self.assertRaises(LearningContractError):
                replace(resource(), **change)

    def test_schema_unknown_missing_extra_fields_and_array_types_rejected(self):
        for obj in (resource(), translation(resource()), hints(resource())):
            for operation in ('version', 'missing', 'extra'):
                data = obj.to_dict()
                if operation == 'version': data['schema_version'] = 2
                elif operation == 'missing': del data['schema_version']
                else: data['unknown'] = 'not ignored'
                with self.subTest(cls=type(obj), operation=operation), self.assertRaises(LearningContractError):
                    type(obj).from_dict(data)
        data = resource().to_dict()
        data['sentences'] = {}
        with self.assertRaises(LearningContractError):
            SentenceResource.from_dict(data)

    def test_dtos_immutable_and_serialization_detached(self):
        saved = resource()
        with self.assertRaises(FrozenInstanceError): saved.source_length = 3
        with self.assertRaises(FrozenInstanceError): saved.source.document_id = 'changed'
        data = saved.to_dict()
        data['source']['namespace'] = 'changed'
        data['sentences'].clear()
        self.assertEqual(saved, resource())
        with self.assertRaises(LearningContractError):
            replace(saved, sentences=list(saved.sentences))

    def test_aligned_translation_round_trip_and_exact_order(self):
        saved = resource()
        result = translation(saved)
        decoded = AlignedTranslation.from_dict(json.loads(json.dumps(result.to_dict(), ensure_ascii=False)))
        decoded.validate_against(saved)
        self.assertEqual(decoded, result)
        self.assertNotIn('source_form', json.dumps(result.to_dict()))
        self.assertNotIn('proficiency', result.to_dict())

    def test_alignment_missing_extra_reordered_or_foreign_ids_rejected(self):
        saved = resource()
        result = translation(saved)
        for rows in (result.rows[:1], result.rows[::-1], (),
                     result.rows + (TranslatedSentence('foreign', '額外'),),
                     (TranslatedSentence('foreign', '錯誤'), result.rows[1])):
            with self.assertRaisesRegex(LearningContractError, 'alignment_invalid'):
                replace(result, rows=rows).validate_against(saved)
        with self.assertRaisesRegex(LearningContractError, 'alignment_duplicate'):
            replace(result, rows=(result.rows[0], result.rows[0]))

    def test_result_reference_spoof_and_context_types_rejected(self):
        saved = resource()
        result = translation(saved)
        other_ref = replace(saved.reference, resource_id='0' * 64)
        with self.assertRaisesRegex(LearningContractError, 'reference_mismatch'):
            replace(result, context=replace(result.context, resource=other_ref)).validate_against(saved)
        for change in ({'result_revision': 0}, {'result_revision': True}, {'provenance': {}},
                       {'explanation_language': 'en'}, {'policy_version': ''}):
            with self.assertRaises(LearningContractError):
                replace(result.context, **change)

    def test_output_text_types_and_surrogates_rejected_without_echo(self):
        for value in ('', ' \n', None, {}, 1, '\ud800'):
            with self.assertRaises(LearningContractError) as raised:
                TranslatedSentence('sentence', value)
            self.assertTrue(str(raised.exception).startswith('learning_'))
        with self.assertRaisesRegex(LearningContractError, 'unicode_invalid'):
            text_hash('\udfff')

    def test_hint_source_span_and_json_round_trip(self):
        saved = resource()
        result = hints(saved)
        decoded = LearningHints.from_dict(json.loads(json.dumps(result.to_dict())))
        self.assertEqual(decoded, result)
        decoded.validate_against(saved, TEXT)
        emoji = replace(result.items[0], span=SourceSpan(8, 9), source_form='😀', lemma=None)
        replace(result, items=(emoji,)).validate_against(saved, TEXT)

    def test_empty_hints_are_valid_saved_result(self):
        saved = resource()
        result = replace(hints(saved), items=())
        LearningHints.from_dict(result.to_dict()).validate_against(saved, TEXT)
        self.assertEqual(result.to_dict()['items'], [])

    def test_hint_wrong_source_sentence_and_cross_sentence_span_rejected(self):
        saved = resource()
        result = hints(saved)
        for item in (replace(result.items[0], source_form='Hallo'),
                     replace(result.items[0], sentence_id='foreign'),
                     replace(result.items[0], span=SourceSpan(2, 13)),
                     replace(result.items[0], span=SourceSpan(0, 2), source_form=' \tX')):
            with self.assertRaises(LearningContractError):
                replace(result, items=(item,)).validate_against(saved, TEXT)
        with self.assertRaises(LearningContractError):
            result.validate_against(saved, TEXT.replace('Hello', 'Hallo'))

    def test_hint_duplicate_and_declared_foreign_language_rejected(self):
        result = hints(resource())
        with self.assertRaisesRegex(LearningContractError, 'duplicate'):
            replace(result, items=(result.items[0], result.items[0]))
        with self.assertRaisesRegex(LearningContractError, 'language_mismatch'):
            replace(result, items=(replace(result.items[0], language_code='fr'),))
        with self.assertRaises(LearningContractError):
            replace(result.items[0], kind='grammar_lesson')

    def test_repeated_word_uses_explicit_occurrence_not_search(self):
        body = 'Go go.'
        saved = resource(body, spans=(SourceSpan(0, len(body)),))
        result = hints(saved)
        first = replace(result.items[0], span=SourceSpan(0, 2), source_form='Go')
        second = replace(first, span=SourceSpan(3, 5), source_form='go')
        replace(result, items=(first, second)).validate_against(saved, body)

    def test_hint_level_and_translation_refresh_do_not_change_sentence_resource(self):
        saved = resource()
        result = hints(saved)
        advanced = replace(result, proficiency=replace(result.proficiency, level_id='C1'))
        refreshed = replace(translation(saved), context=replace(context(saved), result_revision=2))
        self.assertEqual(advanced.context.resource, result.context.resource)
        self.assertNotEqual(advanced.proficiency, result.proficiency)
        self.assertEqual(refreshed.context.resource, result.context.resource)
        result.validate_against(saved, TEXT)

    def test_legacy_task_unit_payload_and_artifact_levels_unchanged(self):
        unit = TaskUnit.from_dict({'unit_id': 'unit', 'content': TEXT, 'source_section_ids': ['section']})
        before = unit.to_dict()
        saved = resource()
        translation(saved).validate_against(saved)
        self.assertEqual(unit.to_dict(), before)
        self.assertNotIn('sentences', before)
        self.assertNotIn('content_blocks', before)
        self.assertEqual(unit.to_content_blocks()[0].content, TEXT)
        self.assertEqual({v.value for v in ArtifactTargetLevel},
            {'document', 'chapter', 'section', 'task_unit', 'content_block'})


if __name__ == '__main__':
    unittest.main()
