"use client";

import { useState, useEffect } from "react";
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
  assessmentId?: string;
  hasReferenceAudio?: boolean;
  onCostUpdate?: (cost: number) => void;
}

export function ReferenceAudioSection({ referenceText, t, assessmentId, hasReferenceAudio, onCostUpdate }: ReferenceAudioSectionProps) {
  const [refAudioUrl, setRefAudioUrl] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [voice, setVoice] = useState("Female 1");
  const [speed, setSpeed] = useState(100);
  const { refreshBalance } = useCredits();

  useEffect(() => {
    if (!assessmentId) return;
    if (hasReferenceAudio === false) return;
    const audioUrl = `/api/assessment/${assessmentId}/reference-audio`;
    if (hasReferenceAudio === true) {
      setRefAudioUrl(audioUrl);
      return;
    }
    let cancelled = false;
    fetch(audioUrl, { method: "HEAD" }).then((res) => {
      if (!cancelled && res.ok) {
        setRefAudioUrl(audioUrl);
      }
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [assessmentId, hasReferenceAudio]);

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
      const data = (await res.json()) as { audioUrl: string; audioPath: string; ttsCost: string };
      if (refAudioUrl && refAudioUrl.startsWith("blob:")) URL.revokeObjectURL(refAudioUrl);
      setRefAudioUrl(data.audioUrl);
      void refreshBalance();
      const cost = parseFloat(data.ttsCost ?? "0");
      onCostUpdate?.(cost);
      toast.info(
        (t.referenceAudioCost ?? "Reference audio generated — HK${cost} deducted")
          .replace("${cost}", data.ttsCost ?? "0.00")
      );

      if (assessmentId && data.audioPath) {
        await fetch(`/api/assessment/${assessmentId}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            referenceAudioPath: data.audioPath,
            additionalCost: cost,
          }),
        });
      }
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

      <div className="grid gap-4 sm:grid-cols-2">
        <VoiceSelect value={voice} onValueChange={setVoice} />
        <SpeedSlider value={speed} onValueChange={setSpeed} />
      </div>

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
