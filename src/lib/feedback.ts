import { generateObject } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { z } from "zod";
import type { WordResult } from "@/types/assessment";

const FEEDBACK_INPUT_PRICE_PER_1M = Number(
  process.env.FEEDBACK_PRICE_INPUT_PER_1M_TOKENS ??
    process.env.VISION_PRICE_INPUT_PER_1M_TOKENS ??
    "0.40",
);
const FEEDBACK_OUTPUT_PRICE_PER_1M = Number(
  process.env.FEEDBACK_PRICE_OUTPUT_PER_1M_TOKENS ??
    process.env.VISION_PRICE_OUTPUT_PER_1M_TOKENS ??
    "1.60",
);
const USD_TO_HKD = 7.8;

export function estimateFeedbackCostHkd(
  promptTokens: number,
  completionTokens: number,
): number {
  const inputUsd = (promptTokens / 1_000_000) * FEEDBACK_INPUT_PRICE_PER_1M;
  const outputUsd = (completionTokens / 1_000_000) * FEEDBACK_OUTPUT_PRICE_PER_1M;
  return Math.max((inputUsd + outputUsd) * USD_TO_HKD, 0.01);
}

export const feedbackSchema = z.object({
  strengths: z.array(z.string()).min(1).max(5),
  weaknesses: z.array(z.string()).min(1).max(5),
  tips: z.array(z.string()).min(1).max(5),
});
export type AssessmentFeedback = z.infer<typeof feedbackSchema>;

interface PhonemeStat {
  phoneme: string;
  avgAccuracy: number;
  count: number;
}
interface ConfusionStat {
  expected: string;
  spoken: string;
  count: number;
}

export interface AssessmentSummary {
  scores: {
    accuracy: number;
    fluency: number;
    completeness: number;
    prosody: number | null;
    pron: number;
  };
  referenceText: string;
  recognizedText: string;
  durationSeconds: number;
  wordCount: number;
  errorCounts: Record<string, number>;
  worstPhonemes: PhonemeStat[];
  topConfusions: ConfusionStat[];
}

export function summarizeAssessment(
  words: WordResult[],
  scores: AssessmentSummary["scores"],
  referenceText: string,
  recognizedText: string,
  durationMs: number,
): AssessmentSummary {
  const errorCounts: Record<string, number> = {};
  const phonemeMap = new Map<string, { sum: number; count: number }>();
  const confusionMap = new Map<string, ConfusionStat>();

  for (const w of words) {
    const et = w.PronunciationAssessment.ErrorType;
    if (et !== "None") {
      errorCounts[et] = (errorCounts[et] ?? 0) + 1;
    }
    if (!w.Phonemes) continue;
    for (const p of w.Phonemes) {
      const raw = p.PronunciationAssessment.AccuracyScore;
      const acc = Number.isFinite(raw) ? raw : 0;
      let stat = phonemeMap.get(p.Phoneme);
      if (!stat) {
        stat = { sum: 0, count: 0 };
        phonemeMap.set(p.Phoneme, stat);
      }
      stat.sum += acc;
      stat.count++;

      const nbest = p.PronunciationAssessment.NBestPhonemes ?? [];
      const top = nbest[0];
      if (top && top.Phoneme !== p.Phoneme) {
        const key = `${p.Phoneme}>${top.Phoneme}`;
        let pair = confusionMap.get(key);
        if (!pair) {
          pair = { expected: p.Phoneme, spoken: top.Phoneme, count: 0 };
          confusionMap.set(key, pair);
        }
        pair.count++;
      }
    }
  }

  const worstPhonemes = Array.from(phonemeMap.entries())
    .map(([phoneme, s]) => ({
      phoneme,
      avgAccuracy: s.count > 0 ? Math.round((s.sum / s.count) * 10) / 10 : 0,
      count: s.count,
    }))
    .filter((p) => p.avgAccuracy < 80)
    .sort((a, b) => a.avgAccuracy - b.avgAccuracy)
    .slice(0, 8);

  const topConfusions = Array.from(confusionMap.values())
    .sort((a, b) => b.count - a.count)
    .slice(0, 8);

  return {
    scores,
    referenceText,
    recognizedText,
    durationSeconds: Math.round(durationMs / 1000),
    wordCount: words.length,
    errorCounts,
    worstPhonemes,
    topConfusions,
  };
}

function buildPrompt(summary: AssessmentSummary, locale: string): string {
  const languageInstruction =
    locale === "zh-TW"
      ? "Respond entirely in Traditional Chinese (繁體中文)."
      : "Respond entirely in English.";

  const prosody = summary.scores.prosody ?? 0;
  const errors = Object.entries(summary.errorCounts)
    .map(([k, v]) => `${k}: ${v}`)
    .join(", ");
  const worst = summary.worstPhonemes
    .map((p) => `/${p.phoneme}/ (${p.avgAccuracy}%, ×${p.count})`)
    .join(", ");
  const confusions = summary.topConfusions
    .map((c) => `/${c.expected}/→/${c.spoken}/ (×${c.count})`)
    .join(", ");

  return `You are an expert English pronunciation coach analyzing a student's speaking assessment. Provide personalized, specific, actionable feedback grounded in the data below — not generic advice.

${languageInstruction}

ASSESSMENT DATA:
- Overall Pronunciation Score: ${summary.scores.pron}/100
- Accuracy: ${summary.scores.accuracy}/100
- Fluency: ${summary.scores.fluency}/100
- Completeness: ${summary.scores.completeness}/100
- Prosody: ${prosody}/100
- Duration: ${summary.durationSeconds}s, ${summary.wordCount} words

ERROR DISTRIBUTION:
${errors || "None"}

WORST PHONEMES (avg accuracy < 80%):
${worst || "None"}

MOST COMMON CONFUSIONS (expected → spoken):
${confusions || "None"}

REFERENCE TEXT (what they should have said):
"${summary.referenceText.slice(0, 500)}"

RECOGNIZED TEXT (what they actually said):
"${summary.recognizedText.slice(0, 500)}"

Provide:
- "strengths": 1-5 concise points on what the student did well (be specific, reference scores/phonemes when relevant).
- "weaknesses": 1-5 concise points identifying specific areas needing improvement (reference actual error types, weak phonemes, or confusion patterns).
- "tips": 1-5 actionable, specific practice tips to address the weaknesses (e.g., specific phonemes to drill, techniques).`;
}

export async function generateFeedback(
  summary: AssessmentSummary,
  locale: string,
): Promise<{ feedback: AssessmentFeedback; cost: number }> {
  const apiKey =
    process.env.FEEDBACK_API_KEY ||
    process.env.VISION_API_KEY ||
    process.env.TTS_API_KEY;
  const baseURL = (
    process.env.FEEDBACK_BASE_URL ||
    process.env.VISION_BASE_URL ||
    process.env.TTS_BASE_URL ||
    ""
  ).replace(/\/$/, "");

  if (!apiKey) {
    throw new Error("FEEDBACK_API_KEY is not configured");
  }

  const openai = createOpenAI({ apiKey, baseURL });

  const { object, usage } = await generateObject({
    model: openai(
      process.env.FEEDBACK_MODEL ||
        process.env.VISION_MODEL ||
        "gpt-4.1-mini",
    ),
    schema: feedbackSchema,
    prompt: buildPrompt(summary, locale),
    temperature: 0.4,
  });

  const cost = estimateFeedbackCostHkd(usage.promptTokens, usage.completionTokens);

  return { feedback: object, cost };
}
