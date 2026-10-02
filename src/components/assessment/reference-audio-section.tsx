"use client";

import { useState, useEffect, useRef } from "react";
import { Volume2, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { useCredits } from "@/hooks/use-credits";
import { VoiceSelect } from "@/components/voice-select";
import { SpeedSlider } from "@/components/speed-slider";
import { AudioPlayer } from "@/components/audio-player";
import { KaraokeText } from "@/components/karaoke-text";
import type { Segment } from "@/types/karaoke";

interface ReferenceAudioSectionProps {
  referenceText: string;
  t: Record<string, string>;
  assessmentId?: string;
  hasReferenceAudio?: boolean;
  /** Karaoke segments persisted with the assessment's reference audio. */
  referenceSegments?: Segment[] | null;
  onCostUpdate?: (cost: number) => void;
}

export function ReferenceAudioSection({ referenceText, t, assessmentId, hasReferenceAudio, referenceSegments, onCostUpdate }: ReferenceAudioSectionProps) {
  const [refAudioUrl, setRefAudioUrl] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [voice, setVoice] = useState("Female 1");
  const [speed, setSpeed] = useState(100);
  const [refSegments, setRefSegments] = useState<Segment[]>([]);
  const [generatedText, setGeneratedText] = useState("");
  const [refCurrentTime, setRefCurrentTime] = useState(0);
  const [refIsPlaying, setRefIsPlaying] = useState(false);
  const [karaokeActive, setKaraokeActive] = useState(false);
  // Set when the user generates audio in this session. The assessment-detail
  // refetch after PATCH can flip hasReferenceAudio to true and re-fire the
  // effect below — without this guard the player src would be swapped
  // mid-playback, interrupting the audio and resetting karaoke timing.
  const generatedUrlRef = useRef<string | null>(null);
  const { refreshBalance } = useCredits();

  useEffect(() => {
    if (refIsPlaying) {
      setKaraokeActive(true);
    }
  }, [refIsPlaying]);

  // Switching to a different assessment (e.g. admin detail view) — drop
  // session-generated audio state so persisted data for the new record
  // takes over cleanly.
  useEffect(() => {
    generatedUrlRef.current = null;
    setRefSegments([]);
    setKaraokeActive(false);
  }, [assessmentId]);

  useEffect(() => {
    if (!assessmentId) return;
    if (hasReferenceAudio === false) return;
    const audioUrl = `/api/assessment/${assessmentId}/reference-audio`;
    if (hasReferenceAudio === true) {
      if (!generatedUrlRef.current) setRefAudioUrl(audioUrl);
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
      const data = (await res.json()) as { audioUrl: string; audioPath: string; ttsCost: string; segments?: Segment[] };
      if (refAudioUrl && refAudioUrl.startsWith("blob:")) URL.revokeObjectURL(refAudioUrl);
      setRefAudioUrl(data.audioUrl);
      generatedUrlRef.current = data.audioUrl;
      setRefSegments(data.segments ?? []);
      setGeneratedText(referenceText.trim());
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
            referenceSegments: data.segments ?? [],
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
        <div className="space-y-3">
          {(karaokeActive && (refSegments.length > 0 || (referenceSegments?.length ?? 0) > 0)) ? (
            <div className="rounded-lg border p-4">
              <KaraokeText
                text={refSegments.length > 0 ? generatedText : referenceText.trim()}
                segments={refSegments.length > 0 ? refSegments : referenceSegments!}
                currentTime={refCurrentTime}
                isPlaying={refIsPlaying}
              />
            </div>
          ) : null}
          <AudioPlayer
            src={refAudioUrl}
            onTimeUpdate={setRefCurrentTime}
            onPlayStateChange={setRefIsPlaying}
            onStop={() => setKaraokeActive(false)}
            onEnded={() => setKaraokeActive(false)}
          />
        </div>
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
