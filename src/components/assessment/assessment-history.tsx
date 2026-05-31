"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Trash2,
  Clock,
  Play,
  ArrowLeft,
  Download,
  Volume2,
  Timer,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { ScoreOverview } from "@/components/assessment/score-overview";
import { TranscriptView } from "@/components/assessment/transcript-view";
import { ErrorSummary } from "@/components/assessment/error-summary";
import {
  WordDetail,
  SyllableView,
} from "@/components/assessment/word-detail";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import type {
  SavedAssessment,
  WordResult,
  ErrorType,
  PronunciationScores,
} from "@/types/assessment";

interface AssessmentHistoryProps {
  t: Record<string, string>;
}

interface HistoryResponse {
  items: SavedAssessment[];
  total: number;
}

interface HistoryItem {
  id: string;
  referenceText: string;
  recognizedText: string;
  durationMs: number;
  pronScore: number;
  cost: number;
  hasAudio: boolean;
  expiresAt: string | null;
  createdAt: string;
}

function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleString("en-HK", {
    timeZone: "Asia/Hong_Kong",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function getDaysUntilExpiry(expiresAt: string | null): number | null {
  if (!expiresAt) return null;
  const diff = new Date(expiresAt).getTime() - Date.now();
  return Math.ceil(diff / (1000 * 60 * 60 * 24));
}

export function AssessmentHistory({ t }: AssessmentHistoryProps) {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<SavedAssessment | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [errorFilter, setErrorFilter] = useState<ErrorType | "All">("All");

  const loadHistory = useCallback(async () => {
    try {
      const res = await fetch("/api/assessment?limit=50");
      if (!res.ok) throw new Error("Failed");
      const data = (await res.json()) as HistoryResponse & {
        items: HistoryItem[];
      };
      setItems(data.items);
    } catch {
      toast.error(t.loadFailed);
    } finally {
      setLoading(false);
    }
  }, [t]);

  useEffect(() => {
    void loadHistory();
  }, [loadHistory]);

  async function loadDetail(id: string) {
    setDetailLoading(true);
    try {
      const res = await fetch(`/api/assessment/${id}`);
      if (!res.ok) throw new Error("Failed");
      const data = (await res.json()) as SavedAssessment;
      setDetail(data);
      setSelectedId(id);
      setErrorFilter("All");
    } catch {
      toast.error(t.loadFailed);
    } finally {
      setDetailLoading(false);
    }
  }

  async function handleDelete(id: string) {
    if (!confirm(t.deleteConfirm)) return;
    try {
      const res = await fetch(`/api/assessment/${id}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Failed");
      toast.success(t.deleted);
      setItems((prev) => prev.filter((a) => a.id !== id));
      if (selectedId === id) {
        setSelectedId(null);
        setDetail(null);
      }
    } catch {
      toast.error(t.deleteFailed);
    }
  }

  function downloadAudio(item: HistoryItem) {
    const a = document.createElement("a");
    a.href = `/api/assessment/${item.id}/audio`;
    a.download = `assessment-${item.id}.webm`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  if (loading) {
    return (
      <div className="space-y-2">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-16 animate-pulse rounded-lg bg-muted" />
        ))}
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <div className="py-8 text-center">
        <Volume2 className="mx-auto mb-2 size-8 opacity-30" />
        <p className="text-sm text-muted-foreground">{t.noHistory}</p>
      </div>
    );
  }

  if (selectedId && detail) {
    const scores: PronunciationScores = {
      AccuracyScore: detail.accuracyScore,
      FluencyScore: detail.fluencyScore,
      CompletenessScore: detail.completenessScore,
      ProsodyScore: detail.prosodyScore ?? 0,
      PronScore: detail.pronScore,
    };
    const words = detail.words;
    const filteredWords: WordResult[] =
      errorFilter === "All"
        ? words
        : words.filter((w) => w.PronunciationAssessment.ErrorType === errorFilter);
    const hasAudio = !!detail.id;
    const daysLeft = getDaysUntilExpiry(detail.expiresAt ?? null);
    const nearExpiry = daysLeft !== null && daysLeft <= 14;

    return (
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setSelectedId(null);
              setDetail(null);
            }}
          >
            <ArrowLeft className="size-4" />
            {t.backToList ?? "Back to list"}
          </Button>
          <h2 className="truncate text-lg font-semibold">
            {detail.referenceText.slice(0, 50)}
            {detail.referenceText.length > 50 ? "..." : ""}
          </h2>
          {daysLeft !== null && (
            <Badge
              variant={nearExpiry ? "destructive" : "outline"}
              className="text-xs shrink-0 gap-1"
            >
              <Timer className="size-3" />
              {t.expiresIn?.replace("{days}", String(daysLeft)) ?? `${daysLeft}d left`}
            </Badge>
          )}
        </div>

        <ScoreOverview scores={scores} t={t} />

        <Separator />

        {hasAudio && (
          <Card>
            <CardContent className="flex items-center gap-3 py-3">
              <audio controls className="w-full" preload="metadata">
                <source
                  src={`/api/assessment/${detail.id}/audio`}
                  type="audio/webm"
                />
              </audio>
              <Button
                variant="ghost"
                size="icon"
                className="shrink-0"
                onClick={() => downloadAudio({ id: detail.id } as HistoryItem)}
              >
                <Download className="size-4" />
              </Button>
            </CardContent>
          </Card>
        )}

        <Separator />

        <div className="space-y-3">
          <h3 className="text-sm font-semibold">{t.recognizedText}</h3>
          <TranscriptView words={words} t={t} />
        </div>

        <Separator />

        <div className="space-y-3">
          <h3 className="text-sm font-semibold">{t.errorSummary}</h3>
          <ErrorSummary
            words={words}
            t={t}
            filter={errorFilter}
            onFilterChange={setErrorFilter}
          />
        </div>

        <Separator />

        <Tabs defaultValue="word">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold">{t.granularity}</h3>
            <TabsList>
              <TabsTrigger value="fulltext">{t.granFullText}</TabsTrigger>
              <TabsTrigger value="word">{t.granWord}</TabsTrigger>
              <TabsTrigger value="syllable">{t.granSyllable}</TabsTrigger>
              <TabsTrigger value="phoneme">{t.granPhoneme}</TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="fulltext">
            <div className="rounded-lg border p-4">
              <ScoreOverview scores={scores} t={t} />
            </div>
          </TabsContent>

          <TabsContent value="word">
            <div className="max-h-96 overflow-y-auto">
              <WordDetail words={filteredWords} t={t} />
            </div>
          </TabsContent>

          <TabsContent value="syllable">
            <div className="max-h-96 overflow-y-auto">
              <SyllableView words={filteredWords} t={t} />
            </div>
          </TabsContent>

          <TabsContent value="phoneme">
            <div className="max-h-96 overflow-y-auto">
              <WordDetail words={filteredWords} t={t} expandAll />
            </div>
          </TabsContent>
        </Tabs>

        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <Clock className="size-3" />
            {formatDate(detail.createdAt)}
          </span>
          <Badge variant="outline">{t.score.replace("{score}", String(Math.round(detail.pronScore)))}</Badge>
          <Badge variant="outline">
            {t.duration.replace("{seconds}", String(Math.round(detail.durationMs / 1000)))}
          </Badge>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {detailLoading && (
        <div className="space-y-2">
          {[1, 2].map((i) => (
            <div key={i} className="h-16 animate-pulse rounded-lg bg-muted" />
          ))}
        </div>
      )}

      {!detailLoading &&
        items.map((item) => {
          const daysLeft = getDaysUntilExpiry(item.expiresAt);
          const nearExpiry = daysLeft !== null && daysLeft <= 14;

          return (
          <Card key={item.id} className={`transition-colors hover:bg-accent/30${nearExpiry ? " border-red-500 border-2" : ""}`}>
            <CardContent className="flex items-center gap-3 py-3">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <p className="truncate text-sm font-medium">
                    {item.referenceText.slice(0, 60)}
                    {item.referenceText.length > 60 ? "..." : ""}
                  </p>
                  {daysLeft !== null && (
                    <Badge
                      variant={nearExpiry ? "destructive" : "outline"}
                      className="text-xs shrink-0 gap-1"
                    >
                      <Timer className="size-3" />
                      {t.expiresIn?.replace("{days}", String(daysLeft)) ?? `${daysLeft}d left`}
                    </Badge>
                  )}
                </div>
                <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
                  <Badge variant="secondary" className="text-xs">
                    {Math.round(item.pronScore)}/100
                  </Badge>
                  <span className="flex items-center gap-1">
                    <Clock className="size-3" />
                    {formatDate(item.createdAt)}
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-1 shrink-0">
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-8"
                  onClick={() => void loadDetail(item.id)}
                  title={t.viewDetails ?? "View details"}
                >
                  <Play className="size-4" />
                </Button>
                {item.hasAudio && (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="size-8"
                    onClick={() => downloadAudio(item)}
                    title={t.download ?? "Download"}
                  >
                    <Download className="size-4" />
                  </Button>
                )}
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-8 text-destructive"
                  onClick={() => void handleDelete(item.id)}
                >
                  <Trash2 className="size-4" />
                </Button>
              </div>
             </CardContent>
           </Card>
        );
        })}
    </div>
  );
}
