"""LLRN-04 multilingual boundaries and lossless source invariants."""
from dataclasses import replace
import random
import unittest
from unittest.mock import patch

from language.language_profile_registry import LanguageProfileRegistry
from language.learning_sentence_rules import sentence_rules
from section_tasks.learning_sentence_segmenter import SEGMENTATION_VERSION, segment_learning_block
from shared.learning_source import LearningSourceBinding, SentenceResource, text_hash


def binding(text):
    return LearningSourceBinding('test', 'book', 'chapter', 'section', 'unit', 'block',
                                 1, 'document-epoch', 'content-epoch', text_hash(text))


def segment(text, language='en', source=None):
    return segment_learning_block(source=source or binding(text), effective_text=text,
                                  source_language=language)


def bodies(result, text):
    return [part.text for part in result.project(text) if part.sentence_id is not None]


# One explicit expected boundary fixture per current registry language. These
# examples certify this baseline, not every orthographic convention in a language.
FIXTURES = {
    'en': ('Dr. Smith paid 3.14 dollars. He left!', ['Dr. Smith paid 3.14 dollars.', 'He left!']),
    'zh': ('他說：「你好！」她微笑。再見！', ['他說：「你好！」她微笑。', '再見！']),
    'ja': ('「こんにちは！」彼は言った。次へ。', ['「こんにちは！」彼は言った。', '次へ。']),
    'ko': ('안녕하세요. 다음 문장입니다!', ['안녕하세요.', '다음 문장입니다!']),
    'es': ('El Sr. Pérez llegó. ¿Todo bien? ¡Sí!', ['El Sr. Pérez llegó.', '¿Todo bien?', '¡Sí!']),
    'pt': ('A Sra. Silva chegou. Tudo bem?', ['A Sra. Silva chegou.', 'Tudo bem?']),
    'fr': ('M. Dupont dit « Bonjour ! » Puis il part.', ['M. Dupont dit « Bonjour ! »', 'Puis il part.']),
    'de': ('Dr. Müller kommt. Das kostet 3.14 Euro.', ['Dr. Müller kommt.', 'Das kostet 3.14 Euro.']),
    'it': ('Il dott. Rossi arriva. Tutto bene?', ['Il dott. Rossi arriva.', 'Tutto bene?']),
    'ru': ('Он живёт на ул. Ленина. Всё хорошо!', ['Он живёт на ул. Ленина.', 'Всё хорошо!']),
    'ar': ('جاء د. أحمد. هل أنت بخير؟ نعم!', ['جاء د. أحمد.', 'هل أنت بخير؟', 'نعم!']),
    'hi': ('डॉ. शर्मा आए। वह खुश हैं॥', ['डॉ. शर्मा आए।', 'वह खुश हैं॥']),
    'tr': ('Dr. Kaya geldi. Her şey iyi mi?', ['Dr. Kaya geldi.', 'Her şey iyi mi?']),
    'nl': ('Dhr. Jansen komt. Alles goed?', ['Dhr. Jansen komt.', 'Alles goed?']),
    'pl': ('Prof. Kowalski przyszedł. Wszystko dobrze?', ['Prof. Kowalski przyszedł.', 'Wszystko dobrze?']),
    'uk': ('Він на вул. Шевченка. Усе добре!', ['Він на вул. Шевченка.', 'Усе добре!']),
    'id': ('Dr. Budi datang. Apa kabar?', ['Dr. Budi datang.', 'Apa kabar?']),
    'vi': ('TS. An đã đến. Bạn khỏe không?', ['TS. An đã đến.', 'Bạn khỏe không?']),
    'th': ('ดร. สมชายมาแล้ว! คุณสบายดีไหม?', ['ดร. สมชายมาแล้ว!', 'คุณสบายดีไหม?']),
}


class LearningSentenceSegmenterTests(unittest.TestCase):
    def check(self, text, expected, language='en'):
        result = segment(text, language)
        self.assertEqual(bodies(result, text), expected)
        self.assertEqual(''.join(part.text for part in result.project(text)), text)
        self.assertEqual(SentenceResource.from_dict(result.to_dict()), result)
        result.validate_source(binding(text), text)
        return result

    def test_all_registry_languages_have_explicit_boundary_fixture(self):
        self.assertEqual(set(FIXTURES), set(LanguageProfileRegistry.get_supported_language_codes()) - {'unknown'})
        for language, (text, expected) in FIXTURES.items():
            with self.subTest(language=language): self.check(text, expected, language)

    def test_exact_language_codes_only(self):
        for language in (None, '', 'unknown', 'EN', 'en-US', '中文', 1, {}, True):
            with self.subTest(language=language), self.assertRaisesRegex(ValueError, 'language_invalid'):
                segment('Hello.', language)

    def test_registry_expansion_removal_or_duplicate_requires_rules(self):
        current = LanguageProfileRegistry.get_supported_language_codes()
        for codes in (list(current) + ['new'], list(current) + ['en'], [c for c in current if c != 'en']):
            with patch.object(LanguageProfileRegistry, 'get_supported_language_codes', return_value=codes):
                with self.assertRaisesRegex(ValueError, 'coverage_invalid'): sentence_rules('en')

    def test_decimals_versions_domains_email_and_urls(self):
        self.check('Value 3.14, v1.2.3 and example.com. Visit https://example.org/a.b?x=2 now. Done.',
                   ['Value 3.14, v1.2.3 and example.com.', 'Visit https://example.org/a.b?x=2 now.', 'Done.'])
        self.check('Mail a.b@example.org. Next.', ['Mail a.b@example.org.', 'Next.'])
        self.check('値は３.１４です。次。', ['値は３.１４です。', '次。'], 'ja')

    def test_initials_acronyms_and_abbreviations(self):
        self.check('J. R. Tolkien met Dr. Jones in the U.S. Army. Next.',
                   ['J. R. Tolkien met Dr. Jones in the U.S. Army.', 'Next.'])
        self.check('Use e.g. apples, etc. Then leave.', ['Use e.g. apples, etc.', 'Then leave.'])
        self.check('Das ist z.B. gut. Weiter.', ['Das ist z.B. gut.', 'Weiter.'], 'de')

    def test_dialogue_closers_and_attribution(self):
        self.check('“Go!” she said. “Why?” He left.', ['“Go!” she said.', '“Why?”', 'He left.'])
        self.check('« Bonjour ! » dit-il. Ensuite.', ['« Bonjour ! » dit-il.', 'Ensuite.'], 'fr')
        self.check('「去！」「再見。」', ['「去！」', '「再見。」'], 'zh')

    def test_nested_parenthetical_punctuation(self):
        self.check('It works (really? yes!) today. Next.', ['It works (really? yes!) today.', 'Next.'])
        self.check('Done (yes!). Next.', ['Done (yes!).', 'Next.'])
        self.check('Text (unfinished. Still here\n\nNew paragraph.',
                   ['Text (unfinished. Still here', 'New paragraph.'])

    def test_ellipsis_and_punctuation_clusters(self):
        self.check('Wait... maybe later. Done?! Next… Yes.', ['Wait... maybe later.', 'Done?!', 'Next…', 'Yes.'])
        self.check('等等……再說。好了！', ['等等……再說。', '好了！'], 'zh')

    def test_mixed_script_quotes_and_no_space_terminals(self):
        self.check('Hello!你好。日本語です。هل هذا صحيح؟हाँ।',
                   ['Hello!', '你好。', '日本語です。', 'هل هذا صحيح؟', 'हाँ।'])
        self.check('He said «Bonjour.» Then left. 再見！', ['He said «Bonjour.»', 'Then left.', '再見！'])

    def test_semicolons_and_weak_punctuation_do_not_end_sentences(self):
        self.check('A; B؛ C； D: E، Finish. Next.', ['A; B؛ C； D: E، Finish.', 'Next.'])

    def test_plain_soft_wrapping_and_crlf_are_not_paragraphs(self):
        for newline in ('\n', '\r\n', '\r', '\u2028'):
            text = f'The first{newline}wrapped line. Next.'
            self.check(text, [f'The first{newline}wrapped line.', 'Next.'])

    def test_blank_lines_and_unicode_paragraph_separator(self):
        for separator in ('\n\n', '\r\n \t\r\n', '\r\r', '\u2029'):
            text = f'Heading{separator}Text without terminal'
            self.check(text, ['Heading', 'Text without terminal'])

    def test_lists_keep_markers_and_continuation_lines(self):
        self.check('Intro\n1. First item\n   continued\n2. Second. More.\n\nEnd',
                   ['Intro', '1. First item\n   continued', '2. Second.', 'More.', 'End'])
        self.check('• First\n• Second', ['• First', '• Second'])
        self.check('一、第一項\n二、第二項', ['一、第一項', '二、第二項'], 'zh')

    def test_empty_whitespace_and_unpunctuated_text(self):
        for text in ('', ' \t\r\n  '): self.check(text, [])
        for language, text in [('th', 'ภาษาไทยไม่มีจุดจบ ประโยคยังอยู่ด้วยกัน'), ('zh', '沒有標點的原文'),
                               ('en', 'Heading or unfinished sentence')]:
            self.check(text, [text], language)

    def test_codepoints_emoji_combining_and_original_whitespace(self):
        text = ' \tHi 👩🏽\u200d💻!\r\nCafe\u0301.\u00a0Next!\n'
        self.check(text, ['Hi 👩🏽\u200d💻!', 'Cafe\u0301.', 'Next!'])
        self.check('Hi!\u0301 Next.', ['Hi!\u0301', 'Next.'])

    def test_source_hash_mismatch_never_rewritten(self):
        with self.assertRaisesRegex(ValueError, 'source_changed'):
            segment('Changed.', source=binding('Original.'))
        with self.assertRaises(ValueError): segment('\ud800')

    def test_source_epoch_and_version_rebuild_ids(self):
        text = 'One. Two.'
        result = segment(text)
        self.assertEqual(result, segment(text))
        self.assertEqual(result.segmentation_version, SEGMENTATION_VERSION)
        for change in ({'content_epoch': 'undo'}, {'structure_version': 2}, {'namespace': 'other'}):
            rebuilt = segment(text, source=replace(binding(text), **change))
            self.assertNotEqual(result.reference, rebuilt.reference)
            self.assertNotEqual(result.sentences[0].sentence_id, rebuilt.sentences[0].sentence_id)
        with patch('section_tasks.learning_sentence_segmenter.SEGMENTATION_VERSION', 'next-version'):
            self.assertNotEqual(result.reference, segment(text).reference)

    def test_seeded_unicode_reconstruction_and_order(self):
        rng = random.Random(104)
        alphabet = 'abcXYZ0123 \t\n\r。.!?…（）「」“”[]،؟।॥😀\u0301\u200d你好ไทย'
        for case in range(150):
            text = ''.join(rng.choice(alphabet) for _ in range(rng.randrange(0, 400)))
            with self.subTest(case=case):
                result = segment(text, list(FIXTURES)[case % len(FIXTURES)])
                self.assertEqual(''.join(p.text for p in result.project(text)), text)
                self.assertEqual(result, segment(text, result.segmentation_language))

    def test_large_unpunctuated_block_and_many_list_items(self):
        self.check('字' * 50000, ['字' * 50000], 'zh')
        text = 'word ' * 10000
        self.check(text, [text.rstrip()])
        text = '\n'.join(f'{i}. item' for i in range(1000))
        self.check(text, text.splitlines())


if __name__ == '__main__': unittest.main()
