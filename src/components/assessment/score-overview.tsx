"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { Info } from "lucide-react";
import { cn } from "@/lib/utils";
import type { PronunciationScores } from "@/types/assessment";

function InfoTip({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);

  const dismiss = useCallback((e: MouseEvent | TouchEvent) => {
    if (ref.current && e.target instanceof Node && ref.current.contains(e.target)) return;
    setOpen(false);
  }, []);

  useEffect(() => {
    if (!open) return;
    document.addEventListener("click", dismiss, true);
    document.addEventListener("touchstart", dismiss, true);
    return () => {
      document.removeEventListener("click", dismiss, true);
      document.removeEventListener("touchstart", dismiss, true);
    };
  }, [open, dismiss]);

  return (
    <span ref={ref} className="relative inline-flex">
      <Info
        className="size-3.5 cursor-pointer text-muted-foreground/60"
        onClick={(e) => { e.stopPropagation(); setOpen((v) => !v); }}
        onTouchStart={(e) => e.stopPropagation()}
        onPointerDown={(e) => e.stopPropagation()}
      />
      <span
        className={cn(
          "absolute bottom-full left-1/2 z-50 mb-2 w-56 -translate-x-1/2 rounded-md bg-popover px-2.5 py-1.5 text-xs font-normal text-popover-foreground shadow-lg ring-1 ring-border transition-opacity",
          open ? "opacity-100" : "pointer-events-none opacity-0",
        )}
      >
        {text}
      </span>
    </span>
  );
}

interface ScoreOverviewProps {
  scores: PronunciationScores;
  t: Record<string, string>;
}

function ScoreBar({
  label,
  score,
  color,
  description,
  na,
}: {
  label: string;
  score: number;
  color: string;
  description?: string;
  na?: boolean;
}) {
  const getColor = () => {
    if (na) return "bg-muted-foreground/30";
    if (score >= 80) return "bg-green-500 dark:bg-green-400";
    if (score >= 60) return "bg-yellow-500 dark:bg-yellow-400";
    if (score >= 40) return "bg-orange-500 dark:bg-orange-400";
    return "bg-red-500 dark:bg-red-400";
  };

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-sm">
        <span className="inline-flex items-center gap-1 font-medium">
          {label}
          {description && <InfoTip text={description} />}
        </span>
        <span className={cn("font-bold tabular-nums", na ? "text-muted-foreground" : color)}>
          {na ? "\u2013" : Math.round(score)}
        </span>
      </div>
      <div className="h-2.5 w-full overflow-hidden rounded-full bg-muted">
        <div
          className={cn("h-full rounded-full transition-all duration-700", getColor())}
          style={{ width: na ? "100%" : `${score}%`, opacity: na ? 0.3 : 1 }}
        />
      </div>
    </div>
  );
}

function OverallGauge({ score, label }: { score: number; label: string }) {
  const getGrade = () => {
    if (score >= 90) return { letter: "A", color: "text-green-600 dark:text-green-400" };
    if (score >= 80) return { letter: "B", color: "text-green-500 dark:text-green-400" };
    if (score >= 70) return { letter: "C", color: "text-yellow-500 dark:text-yellow-400" };
    if (score >= 60) return { letter: "D", color: "text-orange-500 dark:text-orange-400" };
    return { letter: "F", color: "text-red-500 dark:text-red-400" };
  };

  const grade = getGrade();
  const circumference = 2 * Math.PI * 54;
  const dashOffset = circumference - (score / 100) * circumference;

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative size-32">
        <svg className="size-full -rotate-90" viewBox="0 0 120 120">
          <circle
            cx="60" cy="60" r="54"
            fill="none" stroke="currentColor"
            className="text-muted/30"
            strokeWidth="8"
          />
          <circle
            cx="60" cy="60" r="54"
            fill="none"
            stroke="currentColor"
            className={score >= 80 ? "text-green-500 dark:text-green-400" : score >= 60 ? "text-yellow-500 dark:text-yellow-400" : "text-red-500 dark:text-red-400"}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            style={{ transition: "stroke-dashoffset 0.7s ease-out" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={cn("text-3xl font-bold tabular-nums", grade.color)}>
            {Math.round(score)}
          </span>
          <span className="text-xs text-muted-foreground">{label}</span>
        </div>
      </div>
      <span className={cn("text-2xl font-bold", grade.color)}>{grade.letter}</span>
    </div>
  );
}

export function ScoreOverview({ scores, t }: ScoreOverviewProps) {
  return (
    <div className="space-y-6">
      <div className="flex flex-col items-center gap-4 sm:flex-row sm:justify-center sm:gap-8">
        <OverallGauge score={scores.PronScore} label={t.overallScore} />

        <div className="grid w-full max-w-sm gap-4">
          <ScoreBar label={t.accuracy} score={scores.AccuracyScore} color="text-foreground" description={t.accuracyDesc} />
          <ScoreBar label={t.fluency} score={scores.FluencyScore} color="text-foreground" description={t.fluencyDesc} />
          <ScoreBar label={t.completeness} score={scores.CompletenessScore} color="text-foreground" description={t.completenessDesc} />
          <ScoreBar label={t.prosody} score={scores.ProsodyScore} color="text-foreground" na={!Number.isFinite(scores.ProsodyScore) || scores.ProsodyScore <= 0} description={t.prosodyDesc} />
        </div>
      </div>
    </div>
  );
}
