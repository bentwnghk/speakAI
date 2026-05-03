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
  // The gap-filling pass in tts.ts ensures consecutive words are contiguous
  // (word[i].end === word[i+1].start), so there are no silent gaps where the
  // highlight would disappear between words.
  const allTimedWords = useMemo(
    () => segments.flatMap((s) => s.words),
    [segments]
  );

  // ── Active word — binary search O(log n) ────────────────────────────────
  // Find the last word whose `start` ≤ currentTime.  Because the gap-filling
  // pass has made the word list contiguous, this word is guaranteed to be the
  // one currently being spoken as long as currentTime ≤ its `end`.
  // If currentTime has moved past the last word's `end`, return -1 (no active
  // word) so all words render as "past".
  const activeWordIdx = useMemo(() => {
    if (allTimedWords.length === 0) return -1;

    let lo = 0;
    let hi = allTimedWords.length - 1;
    let result = -1;

    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (allTimedWords[mid].start <= currentTime) {
        result = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }

    // Nothing found yet (before first word), or we've moved past the found
    // word's end — neither state has an active word.
    if (result < 0) return -1;
    if (currentTime > allTimedWords[result].end) return -1;

    return result;
  }, [allTimedWords, currentTime]);

  // ── Active segment — derived from active word ────────────────────────────
  // Scroll is triggered at sentence granularity; derive the segment index from
  // the active word index so we don't need a separate linear scan.
  const activeSegmentIdx = useMemo(() => {
    if (activeWordIdx < 0) return -1;
    let wordCount = 0;
    for (let i = 0; i < segments.length; i++) {
      wordCount += segments[i].words.length;
      if (activeWordIdx < wordCount) return i;
    }
    return -1;
  }, [segments, activeWordIdx]);

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

            const isWordActive = token.wordIdx === activeWordIdx;

            // A word is "past" when the active cursor has moved beyond it.
            // When activeWordIdx === -1 (after last word), every word whose
            // end time is before currentTime is considered past.
            const isPast =
              activeWordIdx >= 0
                ? token.wordIdx < activeWordIdx
                : currentTime > timing.end;

            return (
              <span
                key={ti}
                ref={isWordActive ? activeWordRef : undefined}
                className={`${
                  isWordActive
                    ? "bg-yellow-300/50 text-foreground rounded-sm font-bold -mx-0.5 px-0.5"
                    : isPast
                      ? "text-muted-foreground/50 transition-colors duration-300"
                      : "transition-colors duration-300"
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
