"use client";

import { useRef, useEffect, useCallback, useMemo, Fragment } from "react";
import type { Segment } from "@/types/karaoke";

interface KaraokeTextProps {
  text: string;
  segments: Segment[];
  currentTime: number;
  isPlaying: boolean;
  className?: string;
}

/** Matches how tts.ts decides which lines produce segments. */
function isSpeakable(token: string): boolean {
  return /[a-zA-Z0-9]/.test(token);
}

type BlankLine = { type: "blank"; text: string };
type SegmentLine = { type: "segment"; segment: Segment };
type LineData = BlankLine | SegmentLine;

export function KaraokeText({
  text,
  segments,
  currentTime,
  isPlaying,
  className,
}: KaraokeTextProps) {
  const activeWordRef = useRef<HTMLSpanElement>(null);

  // Index of the segment currently being spoken (-1 when idle).
  const activeSegmentIdx = useMemo(() => {
    for (let i = 0; i < segments.length; i++) {
      if (
        currentTime >= segments[i].startTime &&
        currentTime <= segments[i].endTime
      ) {
        return i;
      }
    }
    return -1;
  }, [segments, currentTime]);

  // Map each original text line to either a blank spacer or its corresponding
  // segment.  Non-speakable lines (empty lines, bare bullet symbols like "•")
  // become blank spacers that preserve the paragraph structure of the original
  // text.  The segment counter increments only for speakable lines, matching
  // exactly the segments produced by alignAudio in tts.ts.
  const lineData = useMemo<LineData[]>(() => {
    let segIdx = 0;
    return text.split("\n").map((line): LineData => {
      const trimmed = line.trim();
      if (!trimmed || !isSpeakable(trimmed)) {
        return { type: "blank", text: trimmed };
      }
      const segment = segments[segIdx++];
      if (!segment) return { type: "blank", text: "" };
      return { type: "segment", segment };
    });
  }, [text, segments]);

  const scrollToActive = useCallback(() => {
    activeWordRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  }, []);

  // Scroll when the active segment changes, not on every word tick.
  useEffect(() => {
    if (isPlaying && activeSegmentIdx >= 0) {
      scrollToActive();
    }
  }, [activeSegmentIdx, isPlaying, scrollToActive]);

  return (
    <div
      className={`text-sm leading-relaxed max-h-[50vh] overflow-y-auto ${className ?? ""}`}
    >
      {lineData.map((lineItem, li) => {
        if (lineItem.type === "blank") {
          return (
            <div key={li} className="min-h-[1.25em]">
              {lineItem.text}
            </div>
          );
        }

        const { segment } = lineItem;
        return (
          <div key={li} className="min-h-[1.25em]">
            {segment.words.map((word, wIdx) => {
              const isWordActive =
                currentTime >= word.start && currentTime <= word.end;
              const isPast = currentTime > word.end;
              return (
                <Fragment key={wIdx}>
                  {wIdx > 0 && " "}
                  <span
                    ref={isWordActive ? activeWordRef : undefined}
                    className={
                      isWordActive
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
      })}
    </div>
  );
}
