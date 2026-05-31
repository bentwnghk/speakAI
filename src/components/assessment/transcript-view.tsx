"use client";

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

export function TranscriptView({ words, t }: TranscriptViewProps) {
  const accuracyBuckets = { excellent: 0, good: 0, fair: 0, poor: 0 };

  return (
    <div className="space-y-3">
      <div className="rounded-lg border bg-card p-4">
        <p className="text-sm leading-relaxed">
          {words.map((word, i) => {
            const errorType = word.PronunciationAssessment.ErrorType;
            const isError = errorType !== "None";
            const score = Math.round(word.PronunciationAssessment.AccuracyScore);

            if (!isError) {
              if (score >= 90) accuracyBuckets.excellent++;
              else if (score >= 80) accuracyBuckets.good++;
              else if (score >= 60) accuracyBuckets.fair++;
              else accuracyBuckets.poor++;
            }

            const style = isError
              ? ERROR_STYLES[errorType] ?? ""
              : getAccuracyStyle(score);

            const label = isError
              ? `${String(t[`error${errorType}`] ?? errorType)} (${score}%)`
              : `${getAccuracyLabel(score, t)} (${score}%)`;

            return (
              <span
                key={i}
                title={label}
                className={cn(
                  "inline-block cursor-default rounded px-1 py-0.5 text-sm transition-colors hover:opacity-80",
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
