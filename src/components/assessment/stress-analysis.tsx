"use client";

import { cn } from "@/lib/utils";
import { CheckCircle2, XCircle, MinusCircle } from "lucide-react";
import type { StressWord } from "@/types/assessment";
import { PlayWordButton } from "./play-word-button";

interface StressAnalysisProps {
  stress: StressWord[] | null;
  t: Record<string, string>;
}

export function StressAnalysis({ stress, t }: StressAnalysisProps) {
  if (!stress || stress.length === 0) {
    return (
      <p className="py-6 text-center text-sm text-muted-foreground">
        {t.stressNoData ?? "No multi-syllable words to assess."}
      </p>
    );
  }

  const correctCount = stress.filter((w) => w.correct === true).length;
  const wrongCount = stress.filter((w) => w.correct === false).length;
  const uncertainCount = stress.filter((w) => w.correct === null).length;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <span className="font-medium text-muted-foreground">
          {(t.stressSummary ?? "{correct}/{total} correctly stressed")
            .replace("{correct}", String(correctCount))
            .replace("{total}", String(stress.length))}
        </span>
        {wrongCount > 0 && (
          <span className="text-red-600 dark:text-red-400">
            {t.stressWrong ?? "Wrong"}: {wrongCount}
          </span>
        )}
        {uncertainCount > 0 && (
          <span className="text-muted-foreground">
            {t.stressUncertain ?? "Uncertain"}: {uncertainCount}
          </span>
        )}
      </div>

      <div className="space-y-2">
        {stress.map((w, i) => (
          <StressRow key={i} word={w} t={t} />
        ))}
      </div>

      <p className="text-xs text-muted-foreground">
        {t.stressLegend ??
          "Underlined = expected stress. The bar shows your relative emphasis per syllable."}
      </p>
    </div>
  );
}

function StressRow({ word, t }: { word: StressWord; t: Record<string, string> }) {
  const isWrong = word.correct === false;
  const isUncertain = word.correct === null;

  return (
    <div
      className={cn(
        "rounded-lg border p-3",
        isWrong
          ? "border-red-500/30 bg-red-500/5"
          : isUncertain
            ? "border-muted bg-muted/20"
            : "border-green-500/20 bg-green-500/5",
      )}
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5">
          <span className="font-medium">{word.word}</span>
          <PlayWordButton
            word={word.word}
            label={t.playPronunciation}
            costLabel={t.wordPronunciationCost}
          />
        </span>
        <span
          className={cn(
            "inline-flex items-center gap-1 text-xs font-medium",
            isWrong
              ? "text-red-600 dark:text-red-400"
              : isUncertain
                ? "text-muted-foreground"
                : "text-green-600 dark:text-green-400",
          )}
        >
          {isWrong ? (
            <XCircle className="size-3.5" />
          ) : isUncertain ? (
            <MinusCircle className="size-3.5" />
          ) : (
            <CheckCircle2 className="size-3.5" />
          )}
          {isWrong
            ? (t.stressMisplaced ?? "Misplaced")
            : isUncertain
              ? (t.stressUncertainLabel ?? "Uncertain")
              : (t.stressCorrectLabel ?? "Correct")}
        </span>
      </div>

      <div className="flex flex-wrap items-end gap-2">
        {word.syllables.map((syl, idx) => {
          const isExpected = idx === word.expectedIndex;
          const isActual = idx === word.actualIndex;
          const showYours = isActual && idx !== word.expectedIndex;
          return (
            <span key={idx} className="inline-flex flex-col items-center gap-0.5">
              <span
                className={cn(
                  "rounded px-1.5 py-0.5 font-mono text-sm",
                  isExpected &&
                    "font-semibold underline decoration-2 underline-offset-2 text-primary",
                  showYours && "text-red-600 dark:text-red-400",
                  !isExpected && !showYours && "text-foreground",
                )}
              >
                {syl.text}
              </span>
              <span className="text-[0.6rem] leading-none text-muted-foreground">
                {isExpected ? (t.stressExpected ?? "stress") : showYours ? (t.stressYours ?? "you") : ""}
              </span>
            </span>
          );
        })}
      </div>

      <div className="mt-2 flex h-2 overflow-hidden rounded-full bg-muted">
        {word.syllables.map((syl, idx) => {
          const isExpected = idx === word.expectedIndex;
          const isActual = idx === word.actualIndex;
          const color = isWrong
            ? isActual
              ? "bg-red-500"
              : isExpected
                ? "bg-green-500/60"
                : "bg-muted-foreground/30"
            : isExpected
              ? "bg-green-500"
              : "bg-muted-foreground/30";
          return (
            <div
              key={idx}
              className={cn("h-full transition-all duration-500", color)}
              style={{ width: `${Math.round(syl.prominence * 100)}%` }}
            />
          );
        })}
      </div>
    </div>
  );
}
