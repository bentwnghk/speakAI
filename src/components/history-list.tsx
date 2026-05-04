"use client";

import { useState, useEffect } from "react";
import { useSession } from "next-auth/react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import {
  Play,
  Download,
  Trash2,
  Pencil,
  Volume2,
  Clock,
  ArrowLeft,
  Mic,
  Gauge,
  Coins,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { toast } from "sonner";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { AudioPlayer } from "@/components/audio-player";
import { KaraokeText } from "@/components/karaoke-text";
import { formatVoiceBadge } from "@/lib/constants";
import { useUserSettings } from "@/hooks/use-settings";
import type { Segment } from "@/types/karaoke";

interface Generation {
  id: string;
  title: string;
  transcript: string;
  voice: string;
  speed: number;
  audioUrl: string;
  segments?: Segment[];
  ttsCost: string | null;
  createdAt: string;
}

function formatHongKongDateTime(dateStr: string): string {
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

function formatDownloadTimestamp(dateStr: string): string {
  return new Date(dateStr)
    .toLocaleString("en-HK", {
      timeZone: "Asia/Hong_Kong",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    })
    .replace(/[/:, ]/g, "-");
}

function downloadAudio(gen: Generation) {
  const ts = formatDownloadTimestamp(gen.createdAt);
  const filename = `MrNg-SpeakAI-audio-${ts}.mp3`;

  fetch(gen.audioUrl)
    .then((res) => res.blob())
    .then((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    })
    .catch(() => {
      window.open(gen.audioUrl, "_blank");
    });
}

const PAGE_SIZE_OPTIONS = [10, 20, 30, 50] as const;

export function HistoryList() {
  const { data: session } = useSession();
  const [generations, setGenerations] = useState<Generation[]>([]);
  const [loading, setLoading] = useState(true);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [loadedGeneration, setLoadedGeneration] =
    useState<Generation | null>(null);
  const [audioCurrentTime, setAudioCurrentTime] = useState(0);
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [karaokeActive, setKaraokeActive] = useState(false);
  const [page, setPage] = useState(1);
  const [limit, setLimit] = useState(10);
  const [total, setTotal] = useState(0);
  const { t } = useUserSettings();

  const totalPages = Math.ceil(total / limit);

  useEffect(() => {
    if (session) {
      void fetchGenerations();
    }
  }, [session, page, limit]);

  const fetchGenerations = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        page: String(page),
        limit: String(limit),
      });
      const res = await fetch(`/api/tts?${params}`);
      if (res.ok) {
        const data = (await res.json()) as {
          items: Generation[];
          total: number;
        };
        setGenerations(data.items);
        setTotal(data.total);
      }
    } catch {
      toast.error(t.history.loadFailed);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm(t.history.confirmDelete)) return;

    try {
      const res = await fetch(`/api/generations/${id}`, { method: "DELETE" });
      if (res.ok) {
        const newTotal = total - 1;
        setTotal(newTotal);
        const newTotalPages = Math.ceil(newTotal / limit);
        if (page > newTotalPages && newTotalPages > 0) {
          setPage(newTotalPages);
        } else {
          void fetchGenerations();
        }
        if (loadedGeneration?.id === id) {
          setLoadedGeneration(null);
        }
        toast.success(t.history.audioDeleted);
      }
    } catch {
      toast.error(t.history.deleteFailed);
    }
  };

  const handleRename = async (id: string) => {
    if (!editTitle.trim()) return;

    try {
      const res = await fetch(`/api/generations/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: editTitle }),
      });
      if (res.ok) {
        setGenerations((prev) =>
          prev.map((g) => (g.id === id ? { ...g, title: editTitle } : g))
        );
        if (loadedGeneration?.id === id) {
          setLoadedGeneration((prev) =>
            prev ? { ...prev, title: editTitle } : null
          );
        }
        toast.success(t.history.renamed);
      }
    } catch {
      toast.error(t.history.renameFailed);
    } finally {
      setEditingId(null);
    }
  };

  if (loading) {
    return (
      <div className="space-y-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={i} className="h-20 w-full rounded-lg" />
        ))}
      </div>
    );
  }

  if (generations.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        <Volume2 className="size-8 mx-auto mb-2 opacity-30" />
        <p className="text-sm">{t.history.noAudio}</p>
      </div>
    );
  }

  if (loadedGeneration) {
    const showKaraoke =
      karaokeActive &&
      loadedGeneration.segments &&
      loadedGeneration.segments.length > 0;

    return (
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setLoadedGeneration(null);
              setAudioCurrentTime(0);
              setIsAudioPlaying(false);
              setKaraokeActive(false);
            }}
          >
            <ArrowLeft className="size-4" />
            {t.history.backToList}
          </Button>
          <h2 className="text-lg font-semibold truncate">
            {loadedGeneration.title}
          </h2>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t.history.sourceText}</CardTitle>
          </CardHeader>
          <CardContent>
            {showKaraoke ? (
              <KaraokeText
                text={loadedGeneration.transcript}
                segments={loadedGeneration.segments!}
                currentTime={audioCurrentTime}
                isPlaying={isAudioPlaying}
              />
            ) : (
              <Textarea
                value={loadedGeneration.transcript}
                readOnly
                rows={10}
                className="text-base md:text-base resize-none max-h-[50vh] overflow-y-auto"
              />
            )}
          </CardContent>
        </Card>

        <AudioPlayer
          src={loadedGeneration.audioUrl}
          title={loadedGeneration.title}
          createdAt={loadedGeneration.createdAt}
          onTimeUpdate={setAudioCurrentTime}
          onPlayStateChange={(playing) => {
            setIsAudioPlaying(playing);
            if (playing) setKaraokeActive(true);
          }}
          onStop={() => setKaraokeActive(false)}
          onEnded={() => setKaraokeActive(false)}
        />

        <div className="flex items-center gap-3 flex-wrap">
          <Button
            variant="outline"
            onClick={() => downloadAudio(loadedGeneration)}
          >
            <Download className="size-4" />
            {t.common.download}
          </Button>
          <div className="flex items-center gap-2 text-sm text-muted-foreground flex-wrap">
            <Badge variant="secondary"><Mic className="size-3" />{formatVoiceBadge(loadedGeneration.voice).split(" (")[0]}</Badge>
            <Badge variant="outline"><Gauge className="size-3" />{loadedGeneration.speed}%</Badge>
            <span className="flex items-center gap-1">
              <Clock className="size-3" />
              {formatHongKongDateTime(loadedGeneration.createdAt)}
            </span>
            {loadedGeneration.ttsCost && (
              <span className="flex items-center gap-1"><Coins className="size-3" />HK${loadedGeneration.ttsCost}</span>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <p className="text-sm text-muted-foreground">
          {t.history.items
            .replace("{count}", String(total))
            .replace("{plural}", total !== 1 ? "s" : "")}
        </p>
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">{t.history.perPage}</span>
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

      {generations.map((gen) => (
        <Card key={gen.id}>
           <CardContent className="flex items-start gap-3 py-3">
            <div className="flex-1 min-w-0 space-y-1">
              {editingId === gen.id ? (
                <div className="flex gap-2">
                  <Input
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                    onKeyDown={(e) =>
                      e.key === "Enter" && void handleRename(gen.id)
                    }
                    className="h-7 text-sm"
                  />
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => void handleRename(gen.id)}
                  >
                    {t.common.save}
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => setEditingId(null)}
                  >
                    {t.common.cancel}
                  </Button>
                </div>
              ) : (
                <p className="text-sm font-medium truncate">{gen.title}</p>
              )}
              <div className="flex items-center gap-2 text-xs text-muted-foreground flex-wrap">
                <Badge variant="secondary" className="text-xs gap-1">
                  <Mic className="size-3" />{formatVoiceBadge(gen.voice).split(" (")[0]}
                </Badge>
                <Badge variant="outline" className="text-xs gap-1">
                  <Gauge className="size-3" />{gen.speed}%
                </Badge>
                <span className="flex items-center gap-1">
                  <Clock className="size-3" />
                  {formatHongKongDateTime(gen.createdAt)}
                </span>
                {gen.ttsCost && <span className="flex items-center gap-1"><Coins className="size-3" />HK${gen.ttsCost}</span>}
              </div>
            </div>

            <div className="flex items-center gap-1 shrink-0">
              <Button
                variant="ghost"
                size="icon"
                className="size-8"
                onClick={() => {
                  setAudioCurrentTime(0);
                  setIsAudioPlaying(false);
                  setKaraokeActive(false);
                  setLoadedGeneration(gen);
                }}
                title={t.history.loadSession}
              >
                <Play className="size-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="size-8"
                onClick={() => downloadAudio(gen)}
                title={t.common.download}
              >
                <Download className="size-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="size-8"
                onClick={() => {
                  setEditingId(gen.id);
                  setEditTitle(gen.title);
                }}
              >
                <Pencil className="size-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="size-8 text-destructive"
                onClick={() => void handleDelete(gen.id)}
              >
                <Trash2 className="size-4" />
              </Button>
            </div>
          </CardContent>
        </Card>
      ))}

      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 pt-2">
          <Button
            variant="outline"
            size="sm"
            disabled={page <= 1}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
          >
            <ChevronLeft className="size-4" />
            {t.history.previous}
          </Button>
          <span className="text-sm text-muted-foreground px-2">
            {t.history.pageOf
              .replace("{page}", String(page))
              .replace("{total}", String(totalPages))}
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={page >= totalPages}
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
          >
            {t.history.next}
            <ChevronRight className="size-4" />
          </Button>
        </div>
      )}
    </div>
  );
}
