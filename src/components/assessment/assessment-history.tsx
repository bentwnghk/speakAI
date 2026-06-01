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
  Coins,
  ChevronLeft,
  ChevronRight,
  Star,
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import type {
  SavedAssessment,
  PronunciationScores,
} from "@/types/assessment";
import { type AssessmentFilter, filterWords } from "@/types/assessment";

interface AssessmentHistoryProps {
  t: Record<string, string>;
  ht: Record<string, string>;
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

const PAGE_SIZE_OPTIONS = [10, 20, 30, 50] as const;

export function AssessmentHistory({ t, ht }: AssessmentHistoryProps) {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<SavedAssessment | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [errorFilter, setErrorFilter] = useState<AssessmentFilter>("All");
  const [page, setPage] = useState(1);
  const [limit, setLimit] = useState(10);
  const [total, setTotal] = useState(0);

  const totalPages = Math.ceil(total / limit);

  const loadHistory = useCallback(async () => {
    try {
      const params = new URLSearchParams({
        page: String(page),
        limit: String(limit),
      });
      const res = await fetch(`/api/assessment?${params}`);
      if (!res.ok) throw new Error("Failed");
      const data = (await res.json()) as {
        items: HistoryItem[];
        total: number;
      };
      setItems(data.items);
      setTotal(data.total);
    } catch {
      toast.error(t.loadFailed);
    } finally {
      setLoading(false);
    }
  }, [t, page, limit]);

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
      setTotal((prev) => Math.max(0, prev - 1));
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
    a.download = `assessment-${item.id}.mp4`;
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

  if (items.length === 0 && page === 1) {
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
    const filteredWords = filterWords(words, errorFilter);
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
            {ht.backToList ?? "Back to list"}
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
          <div className="space-y-3">
            <h3 className="text-sm font-semibold">{t.yourRecording}</h3>
            <Card>
              <CardContent className="flex items-center gap-3 py-3">
                <audio controls className="w-full" preload="metadata">
                  <source
                    src={`/api/assessment/${detail.id}/audio`}
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
          </div>
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
          <span className="flex items-center gap-1">
            <Coins className="size-3" />
            HK${detail.cost.toFixed(2)}
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <p className="text-sm text-muted-foreground">
          {ht.items
            .replace("{count}", String(total))
            .replace("{plural}", total !== 1 ? "s" : "")}
        </p>
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">{ht.perPage}</span>
          <Select
            value={String(limit)}
            onValueChange={(val) => {
              setLimit(Number(val));
              setPage(1);
            }}
          >
            <SelectTrigger className="h-8 w-[70px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {PAGE_SIZE_OPTIONS.map((size) => (
                <SelectItem key={size} value={String(size)}>
                  {size}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

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
                  <Badge
                    className={cn(
                      "text-xs font-semibold tabular-nums",
                      item.pronScore >= 80
                        ? "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 hover:bg-emerald-500/25"
                        : item.pronScore >= 60
                          ? "bg-amber-500/15 text-amber-700 dark:text-amber-400 hover:bg-amber-500/25"
                          : "bg-red-500/15 text-red-700 dark:text-red-400 hover:bg-red-500/25",
                    )}
                  >
                    <Star className="size-3" />
                    {Math.round(item.pronScore)}
                  </Badge>
                  <span className="flex items-center gap-1">
                    <Clock className="size-3" />
                    {formatDate(item.createdAt)}
                  </span>
                  <span className="flex items-center gap-1">
                    <Coins className="size-3" />
                    HK${item.cost.toFixed(2)}
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

      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 pt-2">
          <Button
            variant="outline"
            size="sm"
            disabled={page <= 1}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
          >
            <ChevronLeft className="size-4" />
            {ht.previous}
          </Button>
          <span className="text-sm text-muted-foreground px-2">
            {ht.pageOf
              .replace("{page}", String(page))
              .replace("{total}", String(totalPages))}
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={page >= totalPages}
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
          >
            {ht.next}
            <ChevronRight className="size-4" />
          </Button>
        </div>
      )}
    </div>
  );
}
