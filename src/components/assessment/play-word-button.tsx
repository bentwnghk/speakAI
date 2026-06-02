"use client";

import { Volume2 } from "lucide-react";

export async function fetchWordAudio(word: string): Promise<string | null> {
  try {
    const res = await fetch("/api/tts/word", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ word }),
    });
    if (!res.ok) return null;
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  } catch {
    return null;
  }
}

export function PlayWordButton({
  word,
  label,
}: {
  word: string;
  label?: string;
}) {
  function handlePlay(e: React.MouseEvent) {
    e.stopPropagation();
    void fetchWordAudio(word).then((url) => {
      if (!url) return;
      const audio = new Audio(url);
      audio.addEventListener("ended", () => URL.revokeObjectURL(url));
      audio.play().catch(() => URL.revokeObjectURL(url));
    });
  }

  return (
    <button
      type="button"
      onClick={handlePlay}
      onTouchStart={(e) => e.stopPropagation()}
      onPointerDown={(e) => e.stopPropagation()}
      className="inline-flex items-center justify-center rounded p-0.5 text-muted-foreground transition-colors hover:text-foreground hover:bg-muted"
      aria-label={label ?? "Play pronunciation"}
    >
      <Volume2 className="size-3.5" />
    </button>
  );
}
