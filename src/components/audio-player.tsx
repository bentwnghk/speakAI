"use client";

import { useRef, useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Play, Pause, StopCircle, Download, Volume2 } from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { useUserSettings } from "@/hooks/use-settings";

interface AudioPlayerProps {
  src: string | null;
  title?: string;
  createdAt?: string;
  onTimeUpdate?: (currentTime: number) => void;
  onPlayStateChange?: (isPlaying: boolean) => void;
  onStop?: () => void;
  onEnded?: () => void;
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

export function AudioPlayer({ src, title, createdAt, onTimeUpdate, onPlayStateChange, onStop, onEnded }: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const { t } = useUserSettings();

  const onTimeUpdateRef = useRef(onTimeUpdate);
  const onPlayStateChangeRef = useRef(onPlayStateChange);
  const onEndedRef = useRef(onEnded);
  useEffect(() => { onTimeUpdateRef.current = onTimeUpdate; });
  useEffect(() => { onPlayStateChangeRef.current = onPlayStateChange; });
  useEffect(() => { onEndedRef.current = onEnded; });

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    let rafId: number | null = null;

    const tick = () => {
      const time = audio.currentTime;
      setCurrentTime(time);
      onTimeUpdateRef.current?.(time);
      rafId = requestAnimationFrame(tick);
    };

    const startRaf = () => {
      if (rafId === null) rafId = requestAnimationFrame(tick);
    };

    const stopRaf = () => {
      if (rafId !== null) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
    };

    const onPlay = () => startRaf();

    const onPause = () => {
      stopRaf();
      const time = audio.currentTime;
      setCurrentTime(time);
      onTimeUpdateRef.current?.(time);
    };

    const onLoadedMetadata = () => setDuration(audio.duration);

    const onEndedHandler = () => {
      stopRaf();
      setIsPlaying(false);
      onPlayStateChangeRef.current?.(false);
      onEndedRef.current?.();
    };

    audio.addEventListener("play", onPlay);
    audio.addEventListener("pause", onPause);
    audio.addEventListener("loadedmetadata", onLoadedMetadata);
    audio.addEventListener("ended", onEndedHandler);

    if (!audio.paused) startRaf();

    return () => {
      stopRaf();
      audio.removeEventListener("play", onPlay);
      audio.removeEventListener("pause", onPause);
      audio.removeEventListener("loadedmetadata", onLoadedMetadata);
      audio.removeEventListener("ended", onEndedHandler);
    };
  }, [src]);

  useEffect(() => {
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    onPlayStateChange?.(false);
    onTimeUpdate?.(0);
  }, [src]);

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.pause();
    } else {
      void audio.play();
    }
    const newState = !isPlaying;
    setIsPlaying(newState);
    onPlayStateChange?.(newState);
  };

  const handleStop = () => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.pause();
    audio.currentTime = 0;
    setIsPlaying(false);
    setCurrentTime(0);
    onPlayStateChange?.(false);
    onTimeUpdate?.(0);
    onStop?.();
  };

  const handleSeek = (value: number[]) => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = value[0];
    setCurrentTime(value[0]);
  };

  const formatTime = (seconds: number) => {
    if (!isFinite(seconds)) return "0:00";
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  const handleDownload = () => {
    const ts = createdAt
      ? formatDownloadTimestamp(createdAt)
      : new Date()
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
    const filename = `MrNg-SpeakAI-audio-${ts}.mp3`;

    fetch(src!)
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
        window.open(src!, "_blank");
      });
  };

  if (!src) {
    return (
      <Card>
        <CardContent className="flex flex-col items-center justify-center py-6 text-muted-foreground">
          <Volume2 className="size-12 mb-3 opacity-30" />
          <p className="text-sm">{t.tts.audioWillAppear}</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent className="space-y-3 pt-2">
        {title && (
          <p className="text-sm font-medium truncate">{title}</p>
        )}
        <audio ref={audioRef} src={src} preload="metadata" />

        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="icon"
            className="shrink-0 rounded-full"
            onClick={togglePlay}
          >
            {isPlaying ? (
              <Pause className="size-4" />
            ) : (
              <Play className="size-4" />
            )}
          </Button>

          <Button
            variant="outline"
            size="icon"
            className="shrink-0 rounded-full"
            onClick={handleStop}
            disabled={!isPlaying && currentTime === 0}
          >
            <StopCircle className="size-4" />
          </Button>

          <span className="text-xs text-muted-foreground w-10 text-right tabular-nums">
            {formatTime(currentTime)}
          </span>

          <Slider
            value={[currentTime]}
            max={duration || 100}
            step={0.1}
            onValueChange={handleSeek}
            className="flex-1"
          />

          <span className="text-xs text-muted-foreground w-10 tabular-nums">
            {formatTime(duration)}
          </span>

          <Button
            variant="ghost"
            size="icon"
            className="shrink-0"
            onClick={handleDownload}
          >
            <Download className="size-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
