"use client";

import { cn } from "@/lib/utils";
import type { ErrorType } from "@/types/assessment";
import { type AccuracyTier, type AssessmentFilter, getAccuracyTier } from "@/types/assessment";
import type { WordResult } from "@/types/assessment";

interface ErrorSummaryProps {
  words: WordResult[];
  t: Record<string, string>;
  filter: AssessmentFilter;
  onFilterChange: (filter: AssessmentFilter) => void;
}

const ERROR_COLORS: Record<ErrorType, string> = {
  None: "bg-green-500/15 text-green-700 dark:text-green-400 hover:bg-green-500/25",
  Mispronunciation: "bg-red-500/15 text-red-700 dark:text-red-400 hover:bg-red-500/25",
  Omission: "bg-gray-500/15 text-gray-600 dark:text-gray-400 hover:bg-gray-500/25",
  Insertion: "bg-orange-500/15 text-orange-700 dark:text-orange-400 hover:bg-orange-500/25",
  UnexpectedBreak: "bg-blue-500/15 text-blue-700 dark:text-blue-400 hover:bg-blue-500/25",
  MissingBreak: "bg-blue-500/15 text-blue-700 dark:text-blue-400 hover:bg-blue-500/25",
  Monotone: "bg-purple-500/15 text-purple-700 dark:text-purple-400 hover:bg-purple-500/25",
};

const TIER_COLORS: Record<AccuracyTier, string> = {
  Excellent: "bg-green-500/15 text-green-600 dark:text-green-400 hover:bg-green-500/25",
  Good: "bg-lime-500/10 text-lime-600 dark:text-lime-400 hover:bg-lime-500/20",
  Fair: "bg-yellow-500/10 text-yellow-600 dark:text-yellow-400 hover:bg-yellow-500/20",
};

const TIERS: AccuracyTier[] = ["Excellent", "Good", "Fair"];

const ERROR_TYPES: ErrorType[] = [
  "Mispronunciation",
  "Omission",
  "Insertion",
  "UnexpectedBreak",
  "MissingBreak",
  "Monotone",
];

export function ErrorSummary({ words, t, filter, onFilterChange }: ErrorSummaryProps) {
  const tierCounts: Record<AccuracyTier, number> = { Excellent: 0, Good: 0, Fair: 0 };
  const errorCounts: Record<string, number> = {};

  for (const w of words) {
    const et = w.PronunciationAssessment.ErrorType;
    if (et === "None") {
      tierCounts[getAccuracyTier(w.PronunciationAssessment.AccuracyScore)]++;
    } else {
      errorCounts[et] = (errorCounts[et] || 0) + 1;
    }
  }

  const hasErrors = ERROR_TYPES.some((et) => (errorCounts[et] ?? 0) > 0);

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium text-muted-foreground">{t.accuracyLabel}</span>
        <button
          type="button"
          onClick={() => onFilterChange("All")}
          className={cn(
            "cursor-pointer rounded-full px-3 py-1 text-xs font-medium transition-colors",
            filter === "All"
              ? "bg-primary text-primary-foreground"
              : "bg-muted text-muted-foreground hover:bg-accent"
          )}
        >
          All ({words.length})
        </button>
        {TIERS.map((tier) => {
          const count = tierCounts[tier];
          if (count === 0) return null;
          const tierFilter = `None:${tier}` as AssessmentFilter;
          const key = `accuracy${tier}`;
          return (
            <button
              key={tier}
              type="button"
              onClick={() => onFilterChange(tierFilter)}
              className={cn(
                "cursor-pointer rounded-full px-3 py-1 text-xs font-medium transition-colors",
                TIER_COLORS[tier],
                filter === tierFilter && "ring-2 ring-primary ring-offset-1"
              )}
            >
              {String(t[key] ?? tier)} ({count})
            </button>
          );
        })}
      </div>
      {hasErrors && (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-medium text-muted-foreground">{t.errorLabel}</span>
          {ERROR_TYPES.map((type) => {
            const count = errorCounts[type] ?? 0;
            if (count === 0) return null;
            return (
              <button
                key={type}
                type="button"
                onClick={() => onFilterChange(type)}
                className={cn(
                  "cursor-pointer rounded-full px-3 py-1 text-xs font-medium transition-colors",
                  ERROR_COLORS[type],
                  filter === type && "ring-2 ring-primary ring-offset-1"
                )}
              >
                {String(t[`error${type}`] ?? type)} ({count})
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
