"use client";

import { useRef, useEffect, useCallback } from "react";
import type { Segment } from "@/types/karaoke";

interface KaraokeTextProps {
  text: string;
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
  const activeSegmentRef = useRef<HTMLSpanElement>(null);

  const scrollToActive = useCallback(() => {
    activeSegmentRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  }, []);

  useEffect(() => {
    if (isPlaying) {
      scrollToActive();
    }
  }, [isPlaying, currentTime, scrollToActive]);

  if (!segments || segments.length === 0) {
    return (
      <div
        className={`whitespace-pre-wrap text-sm leading-relaxed ${className ?? ""}`}
      >
        {segments.length === 0 ? "No timing data available." : ""}
      </div>
    );
  }

  return (
    <div
      className={`whitespace-pre-wrap text-sm leading-relaxed ${className ?? ""}`}
    >
      {segments.map((segment, sIdx) => {
        const isActive =
          currentTime >= segment.startTime && currentTime <= segment.endTime;
        const isPast = currentTime > segment.endTime;
        const isUpcoming = currentTime < segment.startTime;

        return (
          <span
            key={sIdx}
            ref={isActive ? activeSegmentRef : undefined}
            className={`inline ${
              isActive
                ? "bg-primary/10 rounded px-0.5"
                : isPast
                  ? "text-muted-foreground/60"
                  : isUpcoming
                    ? "text-foreground"
                    : ""
            }`}
          >
            {segment.words.map((word, wIdx) => {
              const isWordActive =
                isActive &&
                currentTime >= word.start &&
                currentTime <= word.end;
              const isWordPast = currentTime > word.end;

              return (
                <span
                  key={wIdx}
                  className={`transition-colors duration-150 ${
                    isWordActive
                      ? "font-bold text-primary"
                      : isWordPast
                        ? "text-muted-foreground/60"
                        : ""
                  }`}
                >
                  {word.word}
                  {wIdx < segment.words.length - 1 ? " " : ""}
                </span>
              );
            })}
          </span>
        );
      })}
    </div>
  );
}
