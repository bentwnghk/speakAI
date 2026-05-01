"use client";

import { useState, useEffect } from "react";
import { useSession } from "next-auth/react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Play,
  Download,
  Trash2,
  Pencil,
  Volume2,
  Clock,
} from "lucide-react";
import { toast } from "sonner";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

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

interface HistoryListProps {
  onLoad?: (generation: Generation) => void;
}

export function HistoryList({ onLoad }: HistoryListProps) {
  const { data: session } = useSession();
  const [generations, setGenerations] = useState<Generation[]>([]);
  const [loading, setLoading] = useState(true);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");

  useEffect(() => {
    if (session) {
      fetchGenerations();
    }
  }, [session]);

  const fetchGenerations = async () => {
    try {
      const res = await fetch("/api/tts");
      if (res.ok) {
        const data = await res.json();
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
        toast.success("Renamed");
      }
    } catch {
      toast.error("Failed to rename");
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
        <p className="text-sm">No audio in history yet</p>
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
                      e.key === "Enter" && handleRename(gen.id)
                    }
                    className="h-7 text-sm"
                  />
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => handleRename(gen.id)}
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
                {gen.ttsCost && (
                  <span>HK${gen.ttsCost}</span>
                )}
              </div>
            </div>

            <div className="flex items-center gap-1 shrink-0">
              {onLoad && (
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-8"
                  onClick={() => onLoad(gen)}
                >
                  <Play className="size-4" />
                </Button>
              )}
              <a href={gen.audioUrl} download>
                <Button variant="ghost" size="icon" className="size-8">
                  <Download className="size-4" />
                </Button>
              </a>
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
                onClick={() => handleDelete(gen.id)}
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
