"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import type { WordResult, ErrorType, PhonemeResult, SyllableResult } from "@/types/assessment";
import { PlayWordButton } from "./play-word-button";

interface WordDetailProps {
  words: WordResult[];
  t: Record<string, string>;
  onCostUpdate?: (cost: number) => void;
}

const ERROR_COLORS: Record<Exclude<ErrorType, "None">, string> = {
  Mispronunciation: "text-red-600 dark:text-red-400",
  Omission: "text-muted-foreground",
  Insertion: "text-orange-600 dark:text-orange-400",
  UnexpectedBreak: "text-blue-600 dark:text-blue-400",
  MissingBreak: "text-blue-600 dark:text-blue-400",
  Monotone: "text-purple-600 dark:text-purple-400",
};

const ERROR_BG: Record<Exclude<ErrorType, "None">, string> = {
  Mispronunciation: "bg-red-500/10 border-red-500/20",
  Omission: "bg-muted/50 border-muted",
  Insertion: "bg-orange-500/10 border-orange-500/20",
  UnexpectedBreak: "bg-blue-500/10 border-blue-500/20",
  MissingBreak: "bg-blue-500/10 border-blue-500/20",
  Monotone: "bg-purple-500/10 border-purple-500/20",
};

function getAccuracyTextColor(score: number): string {
  if (score >= 90) return "text-green-600 dark:text-green-400";
  if (score >= 80) return "text-lime-600 dark:text-lime-400";
  return "text-yellow-600 dark:text-yellow-400";
}

function getAccuracyBg(score: number): string {
  if (score >= 90) return "bg-green-500/15 border-green-500/20";
  if (score >= 80) return "bg-lime-500/10 border-lime-500/20";
  return "bg-yellow-500/10 border-yellow-500/20";
}

export function WordDetail({ words, t, onCostUpdate }: WordDetailProps) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  return (
    <div className="space-y-1">
      {words.map((word, i) => {
        const errorType = word.PronunciationAssessment.ErrorType;
        const isError = errorType !== "None";
        const rawScore = word.PronunciationAssessment.AccuracyScore;
        const score = Number.isFinite(rawScore) ? Math.round(rawScore) : null;
        const isExpanded = expandedIdx === i;
        const hasPhonemes = word.Phonemes && word.Phonemes.length > 0;

        const resolvedScore = score ?? 0;

        return (
          <div key={i}>
            <button
              type="button"
              className={cn(
                "flex w-full items-center gap-3 rounded-lg border px-3 py-2 text-left text-sm transition-colors hover:bg-accent/50",
                isError ? ERROR_BG[errorType] : getAccuracyBg(resolvedScore)
              )}
              onClick={() => setExpandedIdx(isExpanded ? null : i)}
            >
              <span className="shrink-0 text-muted-foreground">
                {isExpanded ? (
                  <ChevronDown className="size-4" />
                ) : hasPhonemes ? (
                  <ChevronRight className="size-4" />
                ) : null}
              </span>

              <span className="font-medium">{word.Word}</span>

              <span
                className={cn("ml-auto tabular-nums font-semibold", isError ? ERROR_COLORS[errorType] : getAccuracyTextColor(resolvedScore))}
              >
                {score ?? "\u2013"}
              </span>

              <span
                className={cn(
                  "text-xs",
                  isError ? ERROR_COLORS[errorType] : getAccuracyTextColor(resolvedScore)
                )}
              >
                {isError
                  ? String(t[`error${errorType}`] ?? errorType)
                  : t.errorNone}
              </span>
            </button>

            {isExpanded && hasPhonemes && word.Phonemes && (
              <PhonemeBreakdown phonemes={word.Phonemes} wordText={word.Word} t={t} onCostUpdate={onCostUpdate} />
            )}
          </div>
        );
      })}
    </div>
  );
}

function PhonemeBreakdown({
  phonemes,
  wordText,
  t,
  onCostUpdate,
}: {
  phonemes: PhonemeResult[];
  wordText: string;
  t: Record<string, string>;
  onCostUpdate?: (cost: number) => void;
}) {
  return (
    <div className="ml-6 mt-1 space-y-2 rounded-lg border bg-card p-3">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <span className="font-medium">{t.phonemeIPA}:</span>
        <span className="font-mono text-foreground">
          /{phonemes.map((p) => p.Phoneme).join("")}/
        </span>
        <PlayWordButton word={wordText} label={t.playPronunciation} costLabel={t.wordPronunciationCost} onCostUpdate={onCostUpdate} />
      </div>

      <div className="grid gap-2">
        {phonemes.map((phoneme, j) => {
          const rawAcc = phoneme.PronunciationAssessment.AccuracyScore;
          const acc = Number.isFinite(rawAcc) ? Math.round(rawAcc) : 0;
          const displayAcc = Number.isFinite(rawAcc) ? Math.round(rawAcc) : null;
          const nbest = phoneme.PronunciationAssessment.NBestPhonemes;

          return (
            <div
              key={j}
              className="flex items-center gap-2 rounded border bg-background px-2 py-1.5 text-sm"
            >
              <span className="w-12 shrink-0 font-mono font-medium text-foreground">
                /{phoneme.Phoneme}/
              </span>

              <div className="flex-1">
                <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                  <div
                    className={cn(
                      "h-full rounded-full transition-all duration-500",
                      acc >= 80
                        ? "bg-green-500"
                        : acc >= 60
                          ? "bg-yellow-500"
                          : "bg-red-500"
                    )}
                    style={{ width: `${acc}%` }}
                  />
                </div>
              </div>

              <span
                className={cn(
                  "w-8 text-right tabular-nums font-semibold",
                  acc >= 80
                    ? "text-green-600 dark:text-green-400"
                    : acc >= 60
                      ? "text-yellow-600 dark:text-yellow-400"
                      : "text-red-600 dark:text-red-400"
                )}
              >
                {displayAcc ?? "\u2013"}
              </span>

              {nbest && nbest.length > 1 && (
                <NBestDetail nbest={nbest} expected={phoneme.Phoneme} t={t} />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function NBestDetail({
  nbest,
  expected,
  t,
}: {
  nbest: { Phoneme: string; Score: number }[];
  expected: string;
  t: Record<string, string>;
}) {
  const [show, setShow] = useState(false);
  const topSpoken = nbest[0];
  const isDifferent = topSpoken && topSpoken.Phoneme !== expected;

  return (
    <div className="group relative ml-1">
      <span
        className="cursor-help text-xs text-muted-foreground underline decoration-dotted select-none"
        onClick={() => setShow((s) => !s)}
      >
        {isDifferent ? `/ ${topSpoken.Phoneme}/` : ""}
      </span>
      {isDifferent && (
        <div
          className={cn(
            "absolute bottom-full right-0 z-10 mb-1 w-max rounded bg-popover px-2 py-1 text-xs shadow-lg",
            show ? "block" : "hidden group-hover:block"
          )}
        >
          <div className="font-medium">{t.phonemeSpoken}:</div>
          {nbest.slice(0, 3).map((nb, k) => (
            <div key={k} className="flex gap-2">
              <span className="font-mono">/{nb.Phoneme}/</span>
              <span className="tabular-nums text-muted-foreground">
                {Math.round(nb.Score)}%
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function SyllableView({ words, t, onCostUpdate }: { words: WordResult[]; t: Record<string, string>; onCostUpdate?: (cost: number) => void }) {
  const allSyllables: { word: string; syllable: SyllableResult }[] = [];
  for (const w of words) {
    if (w.Syllables) {
      for (const s of w.Syllables) {
        allSyllables.push({ word: w.Word, syllable: s });
      }
    }
  }

  if (allSyllables.length === 0) {
    return (
      <p className="py-6 text-center text-sm text-muted-foreground">
        No syllable data available.
      </p>
    );
  }

  return (
    <div className="space-y-1">
      {allSyllables.map((item, i) => {
        const rawAcc = item.syllable.PronunciationAssessment.AccuracyScore;
        const acc = Number.isFinite(rawAcc) ? Math.round(rawAcc) : 0;
        const displayAcc = Number.isFinite(rawAcc) ? Math.round(rawAcc) : null;

        return (
          <div
            key={i}
            className="flex items-center gap-3 rounded-lg border px-3 py-2 text-sm"
          >
            <span className="font-mono font-medium text-foreground w-16 shrink-0">
              {item.syllable.Syllable}
            </span>

            <span className="text-xs text-muted-foreground w-20 shrink-0 truncate">
              ({item.word})
            </span>

            <PlayWordButton word={item.word} label={t.playPronunciation} costLabel={t.wordPronunciationCost} onCostUpdate={onCostUpdate} />

            <div className="flex-1">
              <div className="h-2 overflow-hidden rounded-full bg-muted">
                <div
                  className={cn(
                    "h-full rounded-full transition-all duration-500",
                    acc >= 80
                      ? "bg-green-500"
                      : acc >= 60
                        ? "bg-yellow-500"
                        : "bg-red-500"
                  )}
                  style={{ width: `${acc}%` }}
                />
              </div>
            </div>

            <span
              className={cn(
                "w-8 text-right tabular-nums font-semibold",
                acc >= 80
                  ? "text-green-600 dark:text-green-400"
                  : acc >= 60
                    ? "text-yellow-600 dark:text-yellow-400"
                    : "text-red-600 dark:text-red-400"
              )}
            >
              {displayAcc ?? "\u2013"}
            </span>
          </div>
        );
      })}
    </div>
  );
}
