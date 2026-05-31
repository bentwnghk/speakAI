"use client";

import { useEffect, useState } from "react";
import { Mic, Square, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { motion, AnimatePresence } from "motion/react";
import type { RecordingState, RecordingMode } from "@/types/assessment";

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

interface RecordingControlsProps {
  state: RecordingState;
  mode: RecordingMode;
  onStart: () => void;
  onStop: () => void;
  t: Record<string, string>;
  disabled?: boolean;
}

export function RecordingControls({
  state,
  mode,
  onStart,
  onStop,
  t,
  disabled,
}: RecordingControlsProps) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (state !== "recording") {
      setElapsed(0);
      return;
    }
    const interval = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(interval);
  }, [state]);

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

      {isRecording && (
        <motion.div
          className="relative"
          animate={{ scale: [1, 1.15, 1] }}
          transition={{ repeat: Infinity, duration: 1.5 }}
        >
          <div className="absolute inset-0 rounded-full bg-red-500/20 blur-xl" />
        </motion.div>
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
