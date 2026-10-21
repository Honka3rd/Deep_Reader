import unittest

from section_tasks.summary_note_contracts import (
    NOTE_MAX_CHARS, NotePayload, SummaryPayload, validate_note_text,
)


class SummaryNoteContractTests(unittest.TestCase):
    def test_summary_roundtrip(self):
        value = {"headline": "A meeting", "body": "Charles meets Emma.", "key_points": ["They meet."]}
        self.assertEqual(SummaryPayload.from_dict(value).to_dict(), value)

    def test_summary_rejects_insight_and_invalid_shapes(self):
        for value in ({"summary": "insight", "reasoning": "reason"}, [],
                      {"headline": " ", "body": "body"},
                      {"headline": "h", "body": "b", "key_points": "text"}):
            with self.assertRaises(ValueError):
                SummaryPayload.from_dict(value)

    def test_note_counts_codepoints_and_keeps_whitespace(self):
        text = "\U0001f4dd" * NOTE_MAX_CHARS
        self.assertEqual(validate_note_text(text), text)
        self.assertEqual(validate_note_text("  text\n"), "  text\n")
        for text in ("", " \n", "x" * (NOTE_MAX_CHARS + 1), "\ud800"):
            with self.assertRaises(ValueError):
                validate_note_text(text)

    def test_note_revision_required(self):
        note = NotePayload("server-id", "text", 1, "now", "now")
        self.assertEqual(note.to_dict()["revision"], 1)
        with self.assertRaises(ValueError):
            NotePayload("id", "text", 0, "now", "now")

    def test_wire_validation(self):
        from api_schemas import NoteSaveRequest, SummaryInteractionGenerateRequest
        target = {"doc_name": "book", "target_type": "document"}
        SummaryInteractionGenerateRequest(target=target)
        fields = dict(target=target, text="note", expected_revision=0,
                      mutation_id="retry-id", target_token="source-token")
        NoteSaveRequest(**fields)
        for changes in ({"text": " "}, {"text": "x" * 10001},
                        {"expected_revision": True}, {"prompt": "forbidden"}):
            with self.assertRaises(ValueError):
                NoteSaveRequest(**(fields | changes))


if __name__ == "__main__":
    unittest.main()
