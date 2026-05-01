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
} from "lucide-react";
import { toast } from "sonner";
import { Input } from "@/components/ui/input";
import { AudioPlayer } from "@/components/audio-player";

interface Generation {
  id: string;
  title: string;
  transcript: string;
  voice: string;
  speed: number;
  audioUrl: string;
  ttsCost: string | null;
  createdAt: string;
}

function formatHongKongTimestamp(date: Date): string {
  return date
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

export function HistoryList() {
  const { data: session } = useSession();
  const [generations, setGenerations] = useState<Generation[]>([]);
  const [loading, setLoading] = useState(true);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [loadedGeneration, setLoadedGeneration] =
    useState<Generation | null>(null);

  useEffect(() => {
    if (session) {
      void fetchGenerations();
    }
  }, [session]);

  const fetchGenerations = async () => {
    try {
      const res = await fetch("/api/tts");
      if (res.ok) {
        const data = (await res.json()) as Generation[];
        setGenerations(data);
      }
    } catch {
      toast.error("Failed to load history");
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Are you sure you want to delete this audio?")) return;

    try {
      const res = await fetch(`/api/generations/${id}`, { method: "DELETE" });
      if (res.ok) {
        setGenerations((prev) => prev.filter((g) => g.id !== id));
        if (loadedGeneration?.id === id) {
          setLoadedGeneration(null);
        }
        toast.success("Audio deleted");
      }
    } catch {
      toast.error("Failed to delete");
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
        toast.success("Renamed");
      }
    } catch {
      toast.error("Failed to rename");
    } finally {
      setEditingId(null);
    }
  };

  const handleDownload = (gen: Generation) => {
    const ts = formatHongKongTimestamp(new Date(gen.createdAt));
    const filename = `MrNg-SpeakAI-audio-${ts}.mp3`;

    const a = document.createElement("a");
    a.href = gen.audioUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
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
        <p className="text-sm">No audio in history yet</p>
      </div>
    );
  }

  if (loadedGeneration) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setLoadedGeneration(null)}
          >
            <ArrowLeft className="size-4" />
            Back to list
          </Button>
          <h2 className="text-lg font-semibold truncate">
            {loadedGeneration.title}
          </h2>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Source Text</CardTitle>
          </CardHeader>
          <CardContent>
            <Textarea
              value={loadedGeneration.transcript}
              readOnly
              rows={10}
              className="resize-none"
            />
          </CardContent>
        </Card>

        <AudioPlayer
          src={loadedGeneration.audioUrl}
          title={loadedGeneration.title}
        />

        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            onClick={() => handleDownload(loadedGeneration)}
          >
            <Download className="size-4" />
            Download
          </Button>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Badge variant="secondary">{loadedGeneration.voice}</Badge>
            <Badge variant="outline">{loadedGeneration.speed}%</Badge>
            {loadedGeneration.ttsCost && (
              <span>HK${loadedGeneration.ttsCost}</span>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {generations.map((gen) => (
        <Card key={gen.id}>
          <CardContent className="flex items-start gap-3 pt-6">
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
                    Save
                  </Button>
                </div>
              ) : (
                <p className="text-sm font-medium truncate">{gen.title}</p>
              )}
              <div className="flex items-center gap-2 text-xs text-muted-foreground flex-wrap">
                <Badge variant="secondary" className="text-xs">
                  {gen.voice}
                </Badge>
                <Badge variant="outline" className="text-xs">
                  {gen.speed}%
                </Badge>
                <span className="flex items-center gap-1">
                  <Clock className="size-3" />
                  {new Date(gen.createdAt).toLocaleDateString()}
                </span>
                {gen.ttsCost && <span>HK${gen.ttsCost}</span>}
              </div>
            </div>

            <div className="flex items-center gap-1 shrink-0">
              <Button
                variant="ghost"
                size="icon"
                className="size-8"
                onClick={() => setLoadedGeneration(gen)}
                title="Load session"
              >
                <Play className="size-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="size-8"
                onClick={() => handleDownload(gen)}
                title="Download"
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
    </div>
  );
}
