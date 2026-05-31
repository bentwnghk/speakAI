"use client";

import { cn } from "@/lib/utils";
import type { WordResult, ErrorType } from "@/types/assessment";

interface TranscriptViewProps {
  words: WordResult[];
  t: Record<string, string>;
}

const ERROR_STYLES: Record<ErrorType, string> = {
  None: "text-green-600 dark:text-green-400 bg-green-500/10",
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

const ERROR_ICONS: Record<ErrorType, string> = {
  None: "\u2713",
  Mispronunciation: "\u2717",
  Omission: "\u00D8",
  Insertion: "+",
  UnexpectedBreak: "\u223F",
  MissingBreak: "\u2312",
  Monotone: "\u2248",
};

export function TranscriptView({ words, t }: TranscriptViewProps) {
  return (
    <div className="space-y-3">
      <div className="rounded-lg border bg-card p-4">
        <p className="text-sm leading-relaxed">
          {words.map((word, i) => {
            const errorType = word.PronunciationAssessment.ErrorType;
            const isError = errorType !== "None";

            return (
              <span
                key={i}
                title={
                  isError
                    ? `${t.errorType}: ${String(t[`error${errorType}`] ?? errorType)} (${Math.round(word.PronunciationAssessment.AccuracyScore)}%)`
                    : `${t.errorNone} (${Math.round(word.PronunciationAssessment.AccuracyScore)}%)`
                }
                className={cn(
                  "inline-block cursor-default rounded px-1 py-0.5 text-sm transition-colors hover:opacity-80",
                  ERROR_STYLES[errorType] ?? ERROR_STYLES.None
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
      {(Object.keys(ERROR_STYLES) as ErrorType[]).map((type) => {
        const count = words.filter(
          (w) => w.PronunciationAssessment.ErrorType === type
        ).length;
        if (count === 0 && type !== "None") return null;
        return (
          <span
            key={type}
            className="flex items-center gap-1"
          >
            <span
              className={cn(
                "inline-block rounded px-1 py-px",
                ERROR_STYLES[type]
              )}
            >
              {type === "None" ? "OK" : ERROR_ICONS[type]}
            </span>
            <span>{String(t[`error${type}`] ?? type)}: {count}</span>
          </span>
        );
      })}
      </div>
    </div>
  );
}
