"use client";

import { useRef, useEffect, useCallback, useMemo } from "react";
import type { Segment } from "@/types/karaoke";

interface KaraokeTextProps {
  text: string;
  segments: Segment[];
  currentTime: number;
  isPlaying: boolean;
  className?: string;
}

/** Matches how tts.ts decides which tokens are speakable. */
function isSpeakable(token: string): boolean {
  return /[a-zA-Z0-9]/.test(token);
}

type SpaceToken = { type: "space"; text: string };
type NonWordToken = { type: "nonword"; text: string };
type WordToken = { type: "word"; text: string; wordIdx: number };
type LineToken = SpaceToken | NonWordToken | WordToken;

export function KaraokeText({
  text,
  segments,
  currentTime,
  isPlaying,
  className,
}: KaraokeTextProps) {
  const activeWordRef = useRef<HTMLSpanElement>(null);

  // Flat list of all timed words from all segments, in source-text order.
  // These come from distributeTimingToWords() in tts.ts, so the word strings
  // are the original source words — "per", "cent", "gap-year", etc.
  const allTimedWords = useMemo(
    () => segments.flatMap((s) => s.words),
    [segments]
  );

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

  // Build a per-line token array from the original text.
  // Each speakable whitespace-split token gets an incrementing wordIdx that
  // maps directly into allTimedWords[]. Non-speakable tokens (bullets, dashes,
  // colons) and whitespace are rendered as-is with no timing.
  const lines = useMemo<LineToken[][]>(() => {
    let wordIdx = 0;
    return text.split("\n").map((line) => {
      const parts = line.split(/(\s+)/);
      return parts
        .filter((p) => p.length > 0)
        .map((part): LineToken => {
          if (/^\s+$/.test(part)) return { type: "space", text: part };
          if (isSpeakable(part)) return { type: "word", text: part, wordIdx: wordIdx++ };
          return { type: "nonword", text: part };
        });
    });
  }, [text]);

  const scrollToActive = useCallback(() => {
    activeWordRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  }, []);

  // Scroll when the active sentence changes, not on every word tick.
  useEffect(() => {
    if (isPlaying && activeSegmentIdx >= 0) {
      scrollToActive();
    }
  }, [activeSegmentIdx, isPlaying, scrollToActive]);

  return (
    <div
      className={`text-sm leading-relaxed max-h-[50vh] overflow-y-auto ${className ?? ""}`}
    >
      {lines.map((lineTokens, li) => (
        <div key={li} className="min-h-[1.25em]">
          {lineTokens.map((token, ti) => {
            // Spaces and non-word characters (bullets, punctuation) render as-is.
            if (token.type !== "word") {
              return <span key={ti}>{token.text}</span>;
            }

            // Word token — look up timing from the aligned segments.
            const timing = allTimedWords[token.wordIdx];
            if (!timing) {
              return <span key={ti}>{token.text}</span>;
            }

            const isWordActive =
              currentTime >= timing.start && currentTime <= timing.end;
            const isPast = currentTime > timing.end;

            return (
              <span
                key={ti}
                ref={isWordActive ? activeWordRef : undefined}
                className={`transition-colors duration-75 ${
                  isWordActive
                    ? "font-bold text-primary"
                    : isPast
                      ? "text-muted-foreground/50"
                      : ""
                }`}
              >
                {token.text}
              </span>
            );
          })}
        </div>
      ))}
    </div>
  );
}
