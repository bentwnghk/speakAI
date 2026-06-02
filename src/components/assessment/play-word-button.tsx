"use client";

import { Volume2 } from "lucide-react";
import { toast } from "sonner";
import { useCredits } from "@/hooks/use-credits";

export interface WordAudioResult {
  audioUrl: string;
  creditsUsed: number;
  remainingCredits: number;
}

export async function fetchWordAudio(word: string): Promise<WordAudioResult | null> {
  try {
    const res = await fetch("/api/tts/word", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ word }),
    });
    if (!res.ok) return null;
    const creditsUsed = parseFloat(res.headers.get("X-Credits-Used") ?? "0");
    const remainingCredits = parseFloat(res.headers.get("X-Remaining-Credits") ?? "0");
    const blob = await res.blob();
    const audioUrl = URL.createObjectURL(blob);
    return { audioUrl, creditsUsed, remainingCredits };
  } catch {
    return null;
  }
}

export function PlayWordButton({
  word,
  label,
  costLabel,
}: {
  word: string;
  label?: string;
  costLabel?: string;
}) {
  const { refreshBalance } = useCredits();

  function handlePlay(e: React.MouseEvent) {
    e.stopPropagation();
    void fetchWordAudio(word).then((result) => {
      if (!result) return;
      const audio = new Audio(result.audioUrl);
      audio.addEventListener("ended", () => URL.revokeObjectURL(result.audioUrl));
      audio.play().catch(() => URL.revokeObjectURL(result.audioUrl));
      void refreshBalance();
      if (costLabel && result.creditsUsed > 0) {
        toast.info(costLabel.replace("${cost}", result.creditsUsed.toFixed(4)));
      }
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
