export type ErrorType =
  | "None"
  | "Omission"
  | "Insertion"
  | "Mispronunciation"
  | "UnexpectedBreak"
  | "MissingBreak"
  | "Monotone";

export interface NBestPhoneme {
  Phoneme: string;
  Score: number;
}

export interface PhonemeResult {
  Phoneme: string;
  Offset: number;
  Duration: number;
  PronunciationAssessment: {
    AccuracyScore: number;
    NBestPhonemes?: NBestPhoneme[];
  };
}

export interface WordResult {
  Word: string;
  Offset: number;
  Duration: number;
  PronunciationAssessment: {
    AccuracyScore: number;
    ErrorType: ErrorType;
  };
  Phonemes?: PhonemeResult[];
}

export interface PronunciationScores {
  AccuracyScore: number;
  FluencyScore: number;
  CompletenessScore: number;
  ProsodyScore: number;
  PronScore: number;
}

export interface AssessmentDetailResult {
  RecognitionStatus: number;
  DisplayText: string;
  NBest: {
    Confidence: number;
    Lexical: string;
    Display: string;
    PronunciationAssessment: PronunciationScores;
    Words: WordResult[];
  }[];
}

export interface AssessmentResult {
  detailResult: AssessmentDetailResult;
  scores: PronunciationScores;
  words: WordResult[];
  recognizedText: string;
  durationMs: number;
}

export type AccuracyTier = "Excellent" | "Good" | "Fair";

export type AssessmentFilter =
  | "All"
  | ErrorType
  | `None:${AccuracyTier}`;

export function getAccuracyTier(score: number): AccuracyTier {
  if (score >= 90) return "Excellent";
  if (score >= 80) return "Good";
  return "Fair";
}

export function filterWords(words: WordResult[], filter: AssessmentFilter): WordResult[] {
  if (filter === "All") return words;
  if (filter.startsWith("None:")) {
    const tier = filter.slice(5) as AccuracyTier;
    return words.filter(
      (w) =>
        w.PronunciationAssessment.ErrorType === "None" &&
        getAccuracyTier(w.PronunciationAssessment.AccuracyScore) === tier
    );
  }
  return words.filter((w) => w.PronunciationAssessment.ErrorType === filter);
}

export type GranularityLevel = "FullText" | "Word" | "Phoneme";

export type RecordingMode = "auto" | "manual";

export type RecordingState = "idle" | "recording" | "processing" | "done";

export interface StressSyllable {
  text: string;
  durationMs: number;
  prominence: number;
}

export interface StressWord {
  word: string;
  syllables: StressSyllable[];
  expectedIndex: number;
  actualIndex: number;
  correct: boolean | null;
}

export interface SavedAssessment {
  id: string;
  referenceText: string;
  recognizedText: string;
  durationMs: number;
  accuracyScore: number;
  fluencyScore: number;
  completenessScore: number;
  prosodyScore: number | null;
  pronScore: number;
  words: WordResult[];
  phonemes: PhonemeResult[][] | null;
  audioPath: string | null;
  referenceAudioPath: string | null;
  feedback: { strengths: string[]; weaknesses: string[]; tips: string[] } | null;
  stress: StressWord[] | null;
  cost: number;
  expiresAt: string | null;
  createdAt: string;
}
