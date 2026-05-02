"use client";

import { useRef, useEffect, useMemo, Fragment } from "react";
import type { Segment } from "@/types/karaoke";

interface KaraokeTextProps {
  // `text` is accepted for API compatibility but ignored; rendering is driven
  // entirely by segment word timestamps from Whisper.
  text?: string;
  segments: Segment[];
  currentTime: number;
  isPlaying: boolean;
  className?: string;
}

export function KaraokeText({
  segments,
  currentTime,
  isPlaying,
  className,
}: KaraokeTextProps) {
  const activeWordRef = useRef<HTMLSpanElement>(null);

  // Flat list of every timed word across all segments.
  const allWords = useMemo(
    () => segments.flatMap((s) => s.words),
    [segments]
  );

  // Index of the word currently being spoken (-1 when none active).
  const activeWordIdx = useMemo(() => {
    for (let i = 0; i < allWords.length; i++) {
      if (currentTime >= allWords[i].start && currentTime <= allWords[i].end) {
        return i;
      }
    }
    return -1;
  }, [allWords, currentTime]);

  // Scroll the active word into view whenever it changes.
  useEffect(() => {
    if (isPlaying && activeWordIdx >= 0) {
      activeWordRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    }
  }, [activeWordIdx, isPlaying]);

  return (
    <div
      className={`text-sm leading-relaxed max-h-[50vh] overflow-y-auto ${className ?? ""}`}
    >
      {allWords.map((word, idx) => {
        const isActive = idx === activeWordIdx;
        const isPast = currentTime > word.end;
        return (
          <Fragment key={idx}>
            {idx > 0 && " "}
            <span
              ref={isActive ? activeWordRef : undefined}
              className={
                isActive
                  ? "bg-yellow-300/50 text-foreground rounded-sm font-bold -mx-0.5 px-0.5"
                  : isPast
                    ? "text-muted-foreground/50 transition-colors duration-300"
                    : "transition-colors duration-300"
              }
            >
              {word.word}
            </span>
          </Fragment>
        );
      })}
    </div>
  );
}
