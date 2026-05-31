"use client";

import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import type { WordResult, ErrorType } from "@/types/assessment";

interface TranscriptViewProps {
  words: WordResult[];
  t: Record<string, string>;
}

const ERROR_STYLES: Record<Exclude<ErrorType, "None">, string> = {
  Mispronunciation:
    "text-red-600 dark:text-red-400 bg-red-500/15 font-semibold underline decoration-red-500/60 underline-offset-2",
  Omission:
    "text-muted-foreground bg-muted/50 line-through opacity-70",
  Insertion:
    "text-orange-600 dark:text-orange-400 bg-orange-500/15 italic",
  UnexpectedBreak:
    "text-blue-600 dark:text-blue-400 bg-blue-500/10 underline decoration-dotted decoration-blue-500/60 underline-offset-2",
  MissingBreak:
    "text-blue-600 dark:text-blue-400 bg-blue-500/10 underline decoration-wavy decoration-blue-500/60 underline-offset-2",
  Monotone:
    "text-purple-600 dark:text-purple-400 bg-purple-500/15",
};

const ERROR_ICONS: Record<Exclude<ErrorType, "None">, string> = {
  Mispronunciation: "\u2717",
  Omission: "\u00D8",
  Insertion: "+",
  UnexpectedBreak: "\u223F",
  MissingBreak: "\u2312",
  Monotone: "\u2248",
};

function getAccuracyStyle(score: number): string {
  if (score >= 90)
    return "text-green-600 dark:text-green-400 bg-green-500/15";
  if (score >= 80)
    return "text-lime-600 dark:text-lime-400 bg-lime-500/10";
  if (score >= 60)
    return "text-yellow-600 dark:text-yellow-400 bg-yellow-500/10";
  return "text-red-600 dark:text-red-400 bg-red-500/10";
}

function getAccuracyLabel(score: number, t: Record<string, string>): string {
  if (score >= 90) return t.accuracyExcellent ?? "Excellent";
  if (score >= 80) return t.accuracyGood ?? "Good";
  if (score >= 60) return t.accuracyFair ?? "Fair";
  return t.accuracyPoor ?? "Poor";
}

function getWordLabel(word: WordResult, t: Record<string, string>): string {
  const errorType = word.PronunciationAssessment.ErrorType;
  const rawScore = word.PronunciationAssessment.AccuracyScore;
  const score = Number.isFinite(rawScore) ? Math.round(rawScore) : null;
  const isError = errorType !== "None";
  const scoreStr = score !== null ? ` (${score}%)` : "";
  return isError
    ? `${String(t[`error${errorType}`] ?? errorType)}${scoreStr}`
    : `${getAccuracyLabel(score ?? 0, t)}${scoreStr}`;
}

function WordInfoBar({
  word,
  label,
  t,
}: {
  word: WordResult;
  label: string;
  t: Record<string, string>;
}) {
  const errorType = word.PronunciationAssessment.ErrorType;
  const isError = errorType !== "None";
  const phonemes = word.Phonemes?.map((p) => p.Phoneme).join("");

  return (
    <div className="flex items-center gap-2 rounded-md bg-muted/50 px-2 py-1.5 text-xs">
      <span className="font-semibold">{word.Word}</span>
      <span className="text-muted-foreground">{label}</span>
      {phonemes && (
        <span className="font-mono text-muted-foreground">/{phonemes}/</span>
      )}
      {isError && (
        <span className="text-destructive">
          {String(t[`error${errorType}`] ?? errorType)}
        </span>
      )}
    </div>
  );
}

export function TranscriptView({ words, t }: TranscriptViewProps) {
  const [tappedIdx, setTappedIdx] = useState<number | null>(null);
  const accuracyBuckets = { excellent: 0, good: 0, fair: 0, poor: 0 };

  useEffect(() => {
    if (tappedIdx === null) return;
    function dismiss() {
      setTappedIdx(null);
    }
    document.addEventListener("click", dismiss);
    document.addEventListener("touchstart", dismiss);
    return () => {
      document.removeEventListener("click", dismiss);
      document.removeEventListener("touchstart", dismiss);
    };
  }, [tappedIdx]);

  const tappedWord = tappedIdx !== null ? words[tappedIdx] : null;
  const tappedLabel = tappedWord ? getWordLabel(tappedWord, t) : null;

  return (
    <div className="space-y-3">
      <div className="rounded-lg border bg-card p-4">
        <p className="text-sm leading-relaxed">
          {words.map((word, i) => {
            const errorType = word.PronunciationAssessment.ErrorType;
            const isError = errorType !== "None";
            const rawScore = word.PronunciationAssessment.AccuracyScore;
            const score = Number.isFinite(rawScore) ? Math.round(rawScore) : 0;

            if (!isError) {
              if (score >= 90) accuracyBuckets.excellent++;
              else if (score >= 80) accuracyBuckets.good++;
              else if (score >= 60) accuracyBuckets.fair++;
              else accuracyBuckets.poor++;
            }

            const style = isError
              ? ERROR_STYLES[errorType] ?? ""
              : getAccuracyStyle(score);

            const label = getWordLabel(word, t);
            const isTapped = tappedIdx === i;

            return (
              <span
                key={i}
                role="button"
                tabIndex={0}
                title={label}
                aria-label={label}
                onClick={(e) => {
                  e.stopPropagation();
                  setTappedIdx(isTapped ? null : i);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    setTappedIdx(isTapped ? null : i);
                  }
                }}
                className={cn(
                  "inline-block cursor-pointer rounded px-1 py-0.5 text-sm transition-colors hover:opacity-80 active:opacity-70",
                  isTapped && "ring-2 ring-primary ring-offset-1",
                  style
                )}
              >
                {word.Word}
                {isError && (
                  <sup className="ml-0.5 text-[0.6em]">
                    {ERROR_ICONS[errorType]}
                  </sup>
                )}
              </span>
            );
          })}
        </p>
      </div>

      {tappedWord && tappedLabel && (
        <WordInfoBar word={tappedWord} label={tappedLabel} t={t} />
      )}

      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span className="flex items-center gap-1">
          <span className="inline-block rounded px-1 py-px text-green-600 dark:text-green-400 bg-green-500/15">
            {t.accuracyExcellent ?? "Excellent"} &ge;90
          </span>
          <span>{accuracyBuckets.excellent}</span>
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block rounded px-1 py-px text-lime-600 dark:text-lime-400 bg-lime-500/10">
            {t.accuracyGood ?? "Good"} 80&ndash;89
          </span>
          <span>{accuracyBuckets.good}</span>
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block rounded px-1 py-px text-yellow-600 dark:text-yellow-400 bg-yellow-500/10">
            {t.accuracyFair ?? "Fair"} 60&ndash;79
          </span>
          <span>{accuracyBuckets.fair}</span>
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block rounded px-1 py-px text-red-600 dark:text-red-400 bg-red-500/10">
            {t.accuracyPoor ?? "Poor"} &lt;60
          </span>
          <span>{accuracyBuckets.poor}</span>
        </span>

        {(Object.keys(ERROR_STYLES) as Exclude<ErrorType, "None">[]).map((type) => {
          const count = words.filter(
            (w) => w.PronunciationAssessment.ErrorType === type
          ).length;
          if (count === 0) return null;
          return (
            <span key={type} className="flex items-center gap-1">
              <span
                className={cn("inline-block rounded px-1 py-px", ERROR_STYLES[type])}
              >
                {ERROR_ICONS[type]}
              </span>
              <span>{String(t[`error${type}`] ?? type)}: {count}</span>
            </span>
          );
        })}
      </div>
    </div>
  );
}
