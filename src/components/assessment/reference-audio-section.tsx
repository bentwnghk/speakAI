"use client";

import { useState } from "react";
import { Volume2, Loader2, Download } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { useCredits } from "@/hooks/use-credits";
import { VoiceSelect } from "@/components/voice-select";
import { SpeedSlider } from "@/components/speed-slider";

interface ReferenceAudioSectionProps {
  referenceText: string;
  t: Record<string, string>;
}

export function ReferenceAudioSection({ referenceText, t }: ReferenceAudioSectionProps) {
  const [refAudioUrl, setRefAudioUrl] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [voice, setVoice] = useState("Female 1");
  const [speed, setSpeed] = useState(100);
  const { refreshBalance } = useCredits();

  async function handleGenerate() {
    if (!referenceText.trim()) return;
    setIsGenerating(true);
    try {
      const res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: referenceText.trim(), voice, speed }),
      });
      if (!res.ok) {
        const err = (await res.json()) as { error?: string };
        throw new Error(err.error || "Generation failed");
      }
      const data = (await res.json()) as { audioUrl: string; ttsCost: string };
      if (refAudioUrl) URL.revokeObjectURL(refAudioUrl);
      setRefAudioUrl(data.audioUrl);
      void refreshBalance();
      toast.info(
        (t.referenceAudioCost ?? "Reference audio generated — HK${cost} deducted")
          .replace("${cost}", data.ttsCost ?? "0.00")
      );
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Generation failed");
    } finally {
      setIsGenerating(false);
    }
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">{t.listenToReference}</h3>
        <p className="text-xs text-muted-foreground">{t.listenToReferenceDesc}</p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <VoiceSelect value={voice} onValueChange={setVoice} />
        <SpeedSlider value={speed} onValueChange={setSpeed} />
      </div>

      {refAudioUrl && (
        <Card>
          <CardContent className="flex items-center gap-3 py-3">
            <audio controls className="w-full" preload="metadata">
              <source src={refAudioUrl} />
            </audio>
            <a href={refAudioUrl} download="reference-audio.mp3">
              <Button variant="ghost" size="icon" className="shrink-0" asChild>
                <span>
                  <Download className="size-4" />
                </span>
              </Button>
            </a>
          </CardContent>
        </Card>
      )}

      <Button
        onClick={() => void handleGenerate()}
        disabled={isGenerating || !referenceText.trim()}
        className="w-full"
      >
        {isGenerating ? (
          <>
            <Loader2 className="size-4 animate-spin" />
            {t.generatingReferenceAudio}
          </>
        ) : (
          <>
            <Volume2 className="size-4" />
            {t.generateReferenceAudio}
          </>
        )}
      </Button>
    </div>
  );
}
