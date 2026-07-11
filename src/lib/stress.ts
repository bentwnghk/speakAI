import { dictionary as cmuDict } from "cmu-pronouncing-dictionary";
import type { StressWord } from "@/types/assessment";

const VOWEL_RE = /^(AA|AE|AH|AO|AW|AY|EH|ER|EY|IH|IY|OW|OY|UH|UW)([012])$/;

const UNCERTAINTY_THRESHOLD = 0.12;

interface StressPattern {
  syllableCount: number;
  stressIndex: number;
}

interface AzureSyllable {
  Syllable: string;
  Offset?: number;
  Duration?: number;
}

interface AzureWord {
  Word: string;
  Syllables?: AzureSyllable[];
}

const patternCache = new Map<string, StressPattern | null>();

function parsePattern(phonemes: string): StressPattern {
  const vowelStresses: number[] = [];
  for (const token of phonemes.split(/\s+/)) {
    const m = VOWEL_RE.exec(token);
    if (m) vowelStresses.push(Number(m[2]));
  }
  let stressIndex = vowelStresses.indexOf(1);
  if (stressIndex === -1) stressIndex = vowelStresses.indexOf(2);
  if (stressIndex === -1) stressIndex = 0;
  return { syllableCount: Math.max(vowelStresses.length, 1), stressIndex };
}

export function getStressPattern(word: string): StressPattern | null {
  const key = word.toLowerCase().replace(/[^a-z']/g, "");
  if (!key) return null;
  if (patternCache.has(key)) return patternCache.get(key) ?? null;
  const phonemes = cmuDict[key];
  const result = phonemes ? parsePattern(phonemes) : null;
  patternCache.set(key, result);
  return result;
}

export function analyzeStress(words: unknown): StressWord[] {
  const list = (words as AzureWord[]) ?? [];
  const results: StressWord[] = [];

  for (const w of list) {
    if (!w?.Syllables || w.Syllables.length < 2) continue;
    const pattern = getStressPattern(w.Word);
    if (!pattern || pattern.syllableCount < 2) continue;
    if (pattern.stressIndex >= w.Syllables.length) continue;

    const durations = w.Syllables.map((s) =>
      Number.isFinite(s.Duration) ? (s.Duration as number) : 0,
    );
    const total = durations.reduce((a, b) => a + b, 0);
    if (total <= 0) continue;

    const prominence = durations.map((d) => d / total);
    let actualIndex = 0;
    for (let i = 1; i < durations.length; i++) {
      if (durations[i] > durations[actualIndex]) actualIndex = i;
    }

    let correct: boolean | null;
    if (actualIndex === pattern.stressIndex) {
      correct = true;
    } else {
      const gap = prominence[actualIndex] - prominence[pattern.stressIndex];
      correct = gap >= UNCERTAINTY_THRESHOLD ? false : null;
    }

    results.push({
      word: w.Word,
      syllables: w.Syllables.map((s, i) => ({
        text: s.Syllable,
        durationMs: Math.round(
          (Number.isFinite(s.Duration) ? (s.Duration as number) : 0) / 10000,
        ),
        prominence: Math.round(prominence[i] * 1000) / 1000,
      })),
      expectedIndex: pattern.stressIndex,
      actualIndex,
      correct,
    });
  }

  return results;
}
