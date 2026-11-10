import { Box, Typography } from "@mui/material";
import type { HintAssessment } from "../../services/LearningService";

const reasons: Record<string, string> = {
  invalid_candidate: "Incomplete vocabulary item", invalid_sentence_reference: "Unrecognized source sentence",
  sentence_unavailable: "Sentence could not be verified", invalid_difficulty: "Unrecognized proficiency level",
  invalid_eligibility: "Missing vocabulary selection information", invalid_source: "Expression could not be located in the source",
  ambiguous_source: "Expression location was ambiguous", invalid_kind: "Unrecognized expression type",
  invalid_meaning: "Missing or invalid meaning", wrong_meaning_language: "Meaning used the wrong language",
  duplicate_conflict: "Conflicting entries for the same expression", duplicate: "Repeated entry removed",
  known_level: "Within the selected proficiency level", proper_name: "Proper name", nonlexical: "Not a vocabulary expression",
  uncertain_difficulty: "Proficiency assessment uncertain", foreign_sentence: "Mixed or other-language sentence",
  invalid_annotation: "Invalid sentence annotation", missing_annotation: "Sentence annotation missing",
  duplicate_annotation: "Conflicting sentence annotations", invalid_sentence_language: "Sentence language could not be verified",
  script_mismatch: "Sentence text did not match its language annotation", optional_metadata: "Optional lexical detail discarded",
};

export function HintAssessmentView({ value }: { value: HintAssessment }) {
  const omitted = value.rejected_count > 0 || value.omitted_sentences > 0;
  return <Box className="learning-vocabulary-assessment" sx={{ mb: 1 }}>
    {omitted && <Typography className="learning-vocabulary-omissions" variant="body2" role="status">
      {value.accepted_count ? "Verified vocabulary is shown; some items could not be verified."
        : "No verified vocabulary hints could be retained. This does not mean no hints are needed."}
    </Typography>}
    <Box component="details" className="learning-vocabulary-selection-details">
      <Box component="summary" className="learning-vocabulary-selection-toggle" sx={{ cursor: "pointer", fontSize: 13 }}>
        Vocabulary selection: {value.accepted_count} kept, {value.filtered_count} filtered, {value.rejected_count} not verified
      </Box>
      {value.omitted_sentences > 0 && <Typography variant="caption" component="p">
        {value.omitted_sentences} sentence(s) could not be assessed.
      </Typography>}
      <Box component="ul" className="learning-vocabulary-selection-reasons" sx={{ m: 0, pl: 2.5, maxHeight: 160, overflowY: "auto" }}>
        {value.issues.map(issue => <Typography component="li" variant="caption" key={issue.code}>
          {reasons[issue.code] ?? "Other selection issue"}: {issue.count}
        </Typography>)}
      </Box>
    </Box>
  </Box>;
}
