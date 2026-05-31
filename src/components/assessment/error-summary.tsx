"use client";

import { cn } from "@/lib/utils";
import type { WordResult, ErrorType } from "@/types/assessment";

interface ErrorSummaryProps {
  words: WordResult[];
  t: Record<string, string>;
  filter: ErrorType | "All";
  onFilterChange: (filter: ErrorType | "All") => void;
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

export function ErrorSummary({ words, t, filter, onFilterChange }: ErrorSummaryProps) {
  const counts: Record<string, number> = {};
  for (const w of words) {
    const et = w.PronunciationAssessment.ErrorType;
    counts[et] = (counts[et] || 0) + 1;
  }

  return (
    <div className="flex flex-wrap gap-2">
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
      {(Object.keys(ERROR_COLORS) as ErrorType[]).map((type) => {
        const count = counts[type] ?? 0;
        if (count === 0 && type !== "None") return null;
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
  );
}
