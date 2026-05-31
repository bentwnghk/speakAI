"use client";

import { useState, useEffect, useCallback } from "react";
import { Trash2, Clock } from "lucide-react";
import { toast } from "sonner";
import type { SavedAssessment } from "@/types/assessment";

interface AssessmentHistoryProps {
  t: Record<string, string>;
  onRefresh?: number;
}

interface HistoryResponse {
  items: SavedAssessment[];
  total: number;
}

export function AssessmentHistory({ t, onRefresh }: AssessmentHistoryProps) {
  const [items, setItems] = useState<SavedAssessment[]>([]);
  const [loading, setLoading] = useState(true);

  const loadHistory = useCallback(async () => {
    try {
      const res = await fetch("/api/assessment?limit=5");
      if (!res.ok) throw new Error("Failed");
      const data = (await res.json()) as HistoryResponse;
      setItems(data.items);
    } catch {
      toast.error(t.loadFailed);
    } finally {
      setLoading(false);
    }
  }, [t]);

  useEffect(() => {
    void loadHistory();
  }, [loadHistory, onRefresh]);

  async function handleDelete(id: string) {
    if (!confirm(t.deleteConfirm)) return;
    try {
      const res = await fetch(`/api/assessment/${id}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Failed");
      toast.success(t.deleted);
      setItems((prev) => prev.filter((a) => a.id !== id));
    } catch {
      toast.error(t.deleteFailed);
    }
  }

  function timeAgo(dateStr: string): string {
    const diff = Date.now() - new Date(dateStr).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return "Just now";
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
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
      <p className="py-8 text-center text-sm text-muted-foreground">
        {t.noHistory}
      </p>
    );
  }

  return (
    <div className="space-y-2">
      {items.map((item) => (
        <div
          key={item.id}
          className="flex items-center gap-3 rounded-lg border bg-card px-4 py-3 transition-colors hover:bg-accent/50"
        >
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium">
              {item.referenceText.slice(0, 60)}
              {item.referenceText.length > 60 ? "..." : ""}
            </p>
            <div className="flex items-center gap-3 text-xs text-muted-foreground">
              <span className="font-semibold text-foreground">
                {Math.round(item.pronScore)}/100
              </span>
              <span className="flex items-center gap-1">
                <Clock className="size-3" />
                {timeAgo(item.createdAt)}
              </span>
            </div>
          </div>

          <button
            type="button"
            onClick={() => void handleDelete(item.id)}
            className="shrink-0 rounded p-1.5 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
          >
            <Trash2 className="size-4" />
          </button>
        </div>
      ))}
    </div>
  );
}
