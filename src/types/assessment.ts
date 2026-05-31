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

export interface SyllableResult {
  Syllable: string;
  Offset: number;
  Duration: number;
  PronunciationAssessment: {
    AccuracyScore: number;
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
  Syllables?: SyllableResult[];
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

export type GranularityLevel = "FullText" | "Word" | "Phoneme";

export type RecordingMode = "auto" | "manual";

export type RecordingState = "idle" | "recording" | "processing" | "done";

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
  syllables: SyllableResult[][] | null;
  audioPath: string | null;
  cost: number;
  expiresAt: string | null;
  createdAt: string;
}
