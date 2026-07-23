"use client";

import { useEffect, useState, useRef } from "react";
import { Mic, Square, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { motion, AnimatePresence } from "motion/react";
import type { RecordingState, RecordingMode } from "@/types/assessment";

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

const NUM_BARS = 5;

interface RecordingControlsProps {
  state: RecordingState;
  mode: RecordingMode;
  onStart: () => void;
  onStop: () => void;
  t: Record<string, string>;
  disabled?: boolean;
  micLevelRef?: { current: number };
}

export function RecordingControls({
  state,
  mode,
  onStart,
  onStop,
  t,
  disabled,
  micLevelRef,
}: RecordingControlsProps) {
  const [elapsed, setElapsed] = useState(0);
  const [level, setLevel] = useState(0);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    if (state !== "recording") {
      setElapsed(0);
      return;
    }
    const interval = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(interval);
  }, [state]);

  useEffect(() => {
    if (state !== "recording" || !micLevelRef) {
      setLevel(0);
      return;
    }
    let lastUpdate = 0;
    const tick = (time: number) => {
      if (time - lastUpdate >= 50) {
        setLevel(micLevelRef.current);
        lastUpdate = time;
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [state, micLevelRef]);

  const isRecording = state === "recording";
  const isProcessing = state === "processing";
  const isDisabled = disabled || isProcessing;

  return (
    <div className="flex flex-col items-center gap-4 py-6">
      <AnimatePresence mode="wait">
        {isRecording && (
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.8, opacity: 0 }}
            className="text-3xl font-mono font-bold tabular-nums text-red-500 dark:text-red-400"
          >
            {formatTime(elapsed)}
          </motion.div>
        )}
        {isProcessing && (
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.8, opacity: 0 }}
            className="flex items-center gap-2 text-muted-foreground"
          >
            <Loader2 className="size-5 animate-spin" />
            <span>{t.processing}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {isRecording && (
        <motion.div
          className="flex h-8 items-end gap-[3px]"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.1 }}
        >
          {Array.from({ length: NUM_BARS }).map((_, i) => {
            const center = (NUM_BARS - 1) / 2;
            const distance = Math.abs(i - center);
            const weight = 1 - distance * 0.2;
            const height = 4 + level * 24 * weight;
            return (
              <div
                key={i}
                className="w-[3px] rounded-full bg-primary transition-[height] duration-75 ease-out"
                style={{ height: `${height}px` }}
              />
            );
          })}
        </motion.div>
      )}

      {isRecording && (
        <motion.div
          className="flex items-center gap-2"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.1 }}
        >
          <span className="text-sm text-muted-foreground">
            {mode === "auto"
              ? t.modeAutoDesc
              : t.modeManualDesc}
          </span>
        </motion.div>
      )}

      <div className="relative">
        {isRecording && (
          <div
            className="absolute inset-0 rounded-full bg-red-500/20 blur-xl transition-all duration-100"
            style={{
              transform: `scale(${1 + level * 0.4})`,
              opacity: 0.3 + level * 0.4,
            }}
          />
        )}
        <Button
          size="lg"
          variant={isRecording ? "destructive" : "default"}
          className="relative size-20 rounded-full shadow-lg"
          onClick={isRecording ? onStop : onStart}
          disabled={isDisabled}
        >
          {isRecording ? (
            <Square className="size-8 fill-current" />
          ) : (
            <Mic className="size-8" />
          )}
          {isRecording && (
            <motion.div
              className="absolute inset-0 rounded-full border-2 border-red-500"
              animate={{ scale: [1, 1.3, 1], opacity: [0.8, 0, 0.8] }}
              transition={{ repeat: Infinity, duration: 2 }}
            />
          )}
        </Button>
      </div>

      <p className="text-sm text-muted-foreground">
        {isRecording
          ? t.stopRecording
          : isProcessing
            ? t.processing
            : t.startRecording}
      </p>
    </div>
  );
}
