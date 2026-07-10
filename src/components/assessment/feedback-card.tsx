"use client";

import { useState } from "react";
import { toast } from "sonner";
import {
  Sparkles,
  CheckCircle2,
  AlertTriangle,
  Lightbulb,
  Loader2,
  Coins,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useUserSettings } from "@/hooks/use-settings";
import { useCredits } from "@/hooks/use-credits";
import type { AssessmentFeedback } from "@/lib/feedback";

interface FeedbackCardProps {
  assessmentId: string | null;
  initialFeedback?: AssessmentFeedback | null;
  t: Record<string, string>;
  onCostUpdate?: (cost: number) => void;
}

type Status = "idle" | "loading" | "done" | "error";

export function FeedbackCard({
  assessmentId,
  initialFeedback,
  t,
  onCostUpdate,
}: FeedbackCardProps) {
  const { locale } = useUserSettings();
  const { refreshBalance } = useCredits();
  const [feedback, setFeedback] = useState<AssessmentFeedback | null>(
    initialFeedback ?? null,
  );
  const [status, setStatus] = useState<Status>(
    initialFeedback ? "done" : "idle",
  );
  const [lastCost, setLastCost] = useState<number | null>(null);

  async function handleGenerate() {
    if (!assessmentId) {
      toast.error(t.feedbackNeedsSave ?? "Save the assessment first.");
      return;
    }

    setStatus("loading");
    try {
      const res = await fetch(`/api/assessment/${assessmentId}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ locale }),
      });

      if (res.status === 402) {
        setStatus("idle");
        toast.error(t.insufficientCredits);
        return;
      }

      if (res.status === 404) {
        setStatus("idle");
        toast.error(t.feedbackNotFound ?? "Assessment not found.");
        return;
      }

      if (!res.ok) {
        const data = (await res.json().catch(() => ({}))) as { error?: string };
        setStatus("error");
        toast.error(
          data.error ?? t.feedbackFailed ?? "Failed to generate feedback.",
        );
        return;
      }

      const data = (await res.json()) as {
        feedback: AssessmentFeedback;
        cost: number;
        cached: boolean;
      };

      setFeedback(data.feedback);
      setLastCost(data.cost);
      setStatus("done");
      void refreshBalance();
      if (data.cost > 0) {
        onCostUpdate?.(data.cost);
      }
      if (!data.cached && data.cost > 0) {
        toast.success(
          (t.feedbackCost ?? "Feedback generated — HK${cost} deducted").replace(
            "${cost}",
            data.cost.toFixed(2),
          ),
        );
      }
    } catch {
      setStatus("error");
      toast.error(t.feedbackFailed ?? "Failed to generate feedback.");
    }
  }

  if (status === "loading") {
    return (
      <div className="flex flex-col items-center justify-center gap-3 rounded-lg border p-8 text-center">
        <Loader2 className="size-8 animate-spin text-primary" />
        <p className="text-sm text-muted-foreground">
          {t.feedbackGenerating ?? "Analyzing your pronunciation..."}
        </p>
      </div>
    );
  }

  if (status === "done" && feedback) {
    return (
      <div className="space-y-4">
        {lastCost !== null && lastCost > 0 && (
          <p className="flex items-center gap-1 text-xs text-muted-foreground">
            <Coins className="size-3" />
            {t.feedbackCost?.replace("${cost}", lastCost.toFixed(2))}
          </p>
        )}
        <FeedbackSection
          icon={<CheckCircle2 className="size-4 text-green-600 dark:text-green-400" />}
          title={t.feedbackStrengths ?? "Strengths"}
          items={feedback.strengths}
          accent="border-green-500/20 bg-green-500/5"
        />
        <FeedbackSection
          icon={<AlertTriangle className="size-4 text-amber-600 dark:text-amber-400" />}
          title={t.feedbackWeaknesses ?? "Areas to Improve"}
          items={feedback.weaknesses}
          accent="border-amber-500/20 bg-amber-500/5"
        />
        <FeedbackSection
          icon={<Lightbulb className="size-4 text-yellow-500 dark:text-yellow-400" />}
          title={t.feedbackTips ?? "Practice Tips"}
          items={feedback.tips}
          accent="border-yellow-500/20 bg-yellow-500/5"
        />
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed p-8 text-center">
      <Sparkles className="size-8 text-primary" />
      <div className="space-y-1">
        <p className="text-sm font-medium">
          {t.feedbackTitle ?? "AI Pronunciation Coach"}
        </p>
        <p className="max-w-sm text-xs text-muted-foreground">
          {t.feedbackDesc ??
            "Get personalized feedback on your strengths, weaknesses, and tips to improve your pronunciation."}
        </p>
      </div>
      <Button onClick={() => void handleGenerate()} disabled={!assessmentId}>
        <Sparkles className="size-4" />
        {t.feedbackGenerate ?? "Get AI Feedback"}
      </Button>
    </div>
  );
}

function FeedbackSection({
  icon,
  title,
  items,
  accent,
}: {
  icon: React.ReactNode;
  title: string;
  items: string[];
  accent: string;
}) {
  return (
    <div className={`rounded-lg border p-4 ${accent}`}>
      <div className="mb-2 flex items-center gap-2">
        {icon}
        <h4 className="text-sm font-semibold">{title}</h4>
      </div>
      <ul className="space-y-1.5">
        {items.map((item, i) => (
          <li key={i} className="flex gap-2 text-sm leading-relaxed">
            <span className="mt-1 size-1.5 shrink-0 rounded-full bg-muted-foreground/50" />
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
