import { mkdir, writeFile } from "fs/promises";
import { join } from "path";
import { nanoid } from "nanoid";

import { VOICE_MAP, VOICE_OPTIONS } from "./constants";
import type { Segment, WordTimestamp } from "@/types/karaoke";

export { VOICE_MAP, VOICE_OPTIONS };

/**
 * Append a period to any line that looks like a section heading — that is,
 * a line that:
 *   1. Is non-empty after trimming.
 *   2. Does NOT already end with terminal punctuation (.?!:;).
 *   3. Is short enough to be a heading (≤ 120 characters).
 *   4. Is structurally isolated: preceded by a blank line, followed by a
 *      blank line, or both.  Body sentences inside a paragraph are never
 *      blank-line-delimited; headings almost always are.
 *
 * Why this matters for TTS + Whisper:
 *   - A heading without punctuation causes the TTS engine to run straight
 *     into the next sentence with no prosodic pause, making the heading
 *     indistinguishable from body text.
 *   - Whisper then absorbs the heading into the following segment, so the
 *     karaoke cursor skips the heading entirely and jumps mid-sentence.
 *   - Adding a period gives the TTS engine a natural pause cue and gives
 *     Whisper a reliable segment boundary.
 *
 * Safety for karaoke display:
 *   This function only appends "." to the last character of an existing
 *   token — it never inserts or removes words.  The speakable-word count
 *   per line is therefore identical between the original and processed text,
 *   so the word-index mapping in KaraokeText stays correct when the original
 *   text is shown and the processed text is used for alignment.
 */
export function normalizeHeadingPunctuation(text: string): string {
  const lines = text.split("\n");
  const out: string[] = [];

  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];
    const trimmed = raw.trimEnd();
    const content = trimmed.trim();

    // Blank line — pass through unchanged.
    if (!content) {
      out.push(raw);
      continue;
    }

    // Already ends with terminal punctuation — leave it alone.
    if (/[.?!:;]$/.test(content)) {
      out.push(raw);
      continue;
    }

    // Too long to be a heading.  Body sentences occasionally lack a period
    // (truncated extraction, informal writing) but are never this short.
    if (content.length > 120) {
      out.push(raw);
      continue;
    }

    // Structural isolation test: at least one neighbouring line is blank.
    // i === 0 counts as "preceded by blank" (start of document).
    // i === last counts as "followed by blank" (end of document).
    const prevBlank = i === 0 || !lines[i - 1].trim();
    const nextBlank = i === lines.length - 1 || !lines[i + 1].trim();

    if (prevBlank || nextBlank) {
      out.push(trimmed + ".");
    } else {
      out.push(raw);
    }
  }

  return out.join("\n");
}

export function splitText(text: string, maxChunkSize = 4000): string[] {
  const chunks: string[] = [];
  const paragraphs = text.split("\n\n");

  // Accumulator for paragraphs that are being batched into one chunk.
  // Paragraphs are joined with "\n\n" so the TTS engine gets natural
  // paragraph-break pauses and the resulting audio is one continuous stream
  // with no ID3-header splice points that would confuse Whisper's timestamper.
  let currentChunk = "";

  const flushChunk = () => {
    if (currentChunk) {
      chunks.push(currentChunk);
      currentChunk = "";
    }
  };

  for (const paragraph of paragraphs) {
    const trimmed = paragraph.trim();
    if (!trimmed) continue;

    if (trimmed.length > maxChunkSize) {
      // This single paragraph is too long for one API call.
      // Flush whatever was accumulated, then split the paragraph internally.
      flushChunk();
      const sentences = trimmed.split(/(?<=[.!?])\s+/);
      let sentenceChunk = "";
      for (const sentence of sentences) {
        if (sentence.length > maxChunkSize) {
          if (sentenceChunk) { chunks.push(sentenceChunk); sentenceChunk = ""; }
          const words = sentence.split(" ");
          let wordChunk = "";
          for (const word of words) {
            if (wordChunk.length + word.length + 1 > maxChunkSize) {
              chunks.push(wordChunk);
              wordChunk = word;
            } else {
              wordChunk += (wordChunk ? " " : "") + word;
            }
          }
          if (wordChunk) chunks.push(wordChunk);
        } else if (sentenceChunk.length + sentence.length + 1 > maxChunkSize) {
          chunks.push(sentenceChunk);
          sentenceChunk = sentence;
        } else {
          sentenceChunk += (sentenceChunk ? " " : "") + sentence;
        }
      }
      if (sentenceChunk) chunks.push(sentenceChunk);
    } else {
      // Normal paragraph: accumulate with previous ones as long as the
      // combined length stays within the limit.  This keeps short texts in a
      // single TTS call, producing one unbroken MP3 that Whisper can timestamp
      // reliably from the very first word.
      const joined = currentChunk ? `${currentChunk}\n\n${trimmed}` : trimmed;
      if (joined.length > maxChunkSize) {
        flushChunk();
        currentChunk = trimmed;
      } else {
        currentChunk = joined;
      }
    }
  }

  flushChunk();
  return chunks;
}

async function fetchAudioChunk(
  text: string,
  voice: string,
  speed: number,
  apiKey: string,
  baseUrl: string
): Promise<ArrayBuffer> {
  const url = `${baseUrl}/audio/speech`;

  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "tts-1",
          voice,
          input: text,
          response_format: "mp3",
          speed,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          `TTS API error: ${response.status} - ${errorText}`
        );
      }

      return await response.arrayBuffer();
    } catch (error) {
      if (attempt === 2) throw error;
      const delay = Math.min(1000 * Math.pow(2, attempt), 10000);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  throw new Error("Failed after 3 attempts");
}

export interface AudioChunk {
  text: string;
  buffer: ArrayBuffer;
}

export async function generateTtsAudio(
  text: string,
  voice: string,
  speedPercent: number
): Promise<{ audioPath: string; cost: string; audioDurationSeconds: number; chunks: AudioChunk[] }> {
  const apiKey = process.env.TTS_API_KEY;
  const baseUrl = (process.env.TTS_BASE_URL || "https://api.openai.com/v1").replace(/\/$/, "");

  if (!apiKey) {
    throw new Error("TTS_API_KEY is not configured");
  }

  const actualVoice = VOICE_MAP[voice] || "nova";
  const speed = speedPercent / 100;

  const chunkTexts = splitText(text);
  const audioBuffers: ArrayBuffer[] = [];

  const concurrencyLimit = 10;
  const batches: string[][] = [];
  for (let i = 0; i < chunkTexts.length; i += concurrencyLimit) {
    batches.push(chunkTexts.slice(i, i + concurrencyLimit));
  }

  for (const batch of batches) {
    const results = await Promise.all(
      batch.map((chunk) =>
        fetchAudioChunk(chunk, actualVoice, speed, apiKey, baseUrl)
      )
    );
    audioBuffers.push(...results);
  }

  const totalBytes = audioBuffers.reduce((sum, buf) => sum + buf.byteLength, 0);
  const combinedBuffer = new Uint8Array(totalBytes);
  let offset = 0;
  for (const buf of audioBuffers) {
    combinedBuffer.set(new Uint8Array(buf), offset);
    offset += buf.byteLength;
  }

  const audioDir = join(process.cwd(), "data", "audio");
  await mkdir(audioDir, { recursive: true });

  const audioId = nanoid();
  const audioPath = join(audioDir, `${audioId}.mp3`);
  await writeFile(audioPath, combinedBuffer);

  const characters = text.length;
  const ttsCost = (characters / 1_000_000) * 15 * 7.8;

  return {
    audioPath: `data/audio/${audioId}.mp3`,
    cost: ttsCost.toFixed(2),
    audioDurationSeconds: 0,
    chunks: chunkTexts.map((t, i) => ({ text: t, buffer: audioBuffers[i] })),
  };
}

interface WhisperWord {
  word: string;
  start: number;
  end: number;
}

interface WhisperSegment {
  text: string;
  start: number;
  end: number;
}

/** A token is "speakable" if it contains at least one alphanumeric character.
 *  Filters out bullets (•), em dashes (—), bare colons, etc. */
function isSpeakableWord(token: string): boolean {
  return /[a-zA-Z0-9]/.test(token);
}

/**
 * Regex that matches a sentence boundary *within* a paragraph line.
 *
 * A boundary is the position between:
 *   • terminal punctuation (.?!) optionally followed by a closing quote, AND
 *   • one or more whitespace characters, AND
 *   • an uppercase letter or opening quotation mark.
 *
 * The positive lookbehind requires the period to be preceded by ≥ 2
 * consecutive lowercase letters.  This excludes abbreviation dots whose
 * word ends in a single uppercase or single lowercase letter:
 *   "Dr."  → "r"  — only 1 lowercase before "."  → NOT split  ✓
 *   "U.S." → "S"  — uppercase before "."          → NOT split  ✓
 *   "vs."  → "s"  — only 1 lowercase before "."  → NOT split  ✓
 *   "lives." → "es" — 2+ lowercase before "."    → split       ✓
 *   "time."  → "me" — 2+ lowercase before "."    → split       ✓
 *
 * Bare ? and ! are always sentence-enders regardless of context
 * (they are never used in abbreviations).
 *
 * Unicode closing quotes U+201D ("), U+2019 (') and plain ASCII equivalents
 * are included so `really." Next` and `done?' She` are handled correctly.
 */
const SENTENCE_BOUNDARY_RE =
  /(?<=[a-z]{2,}[.?!][\u201D\u2019"']?|[?!][\u201D\u2019"']?)\s+(?=[A-Z\u201C\u2018"'])/;

/**
 * Split source text into individual sentences for Whisper alignment.
 *
 * Previously this function split only on newlines, treating every paragraph
 * as a single "sentence".  For a multi-paragraph text that produces ~10
 * source sentences while Whisper returns ~35–40 segments (one per actual
 * sentence), guaranteeing the count-mismatch fallback for the whole text.
 *
 * The fallback path distributes paragraph-level timing proportionally, which
 * accumulates errors across paragraphs: a small boundary error in paragraph 1
 * shifts everything in paragraph 2, and so on, producing several seconds of
 * drift by the second half of a long text.
 *
 * This version additionally splits each paragraph line at sentence boundaries
 * using SENTENCE_BOUNDARY_RE, bringing the source sentence count in line with
 * Whisper's natural segmentation.  For the typical long-form text this enables
 * the direct per-segment timestamp path and eliminates cascading drift.
 */
function splitIntoSentences(text: string): string[] {
  const result: string[] = [];
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || !isSpeakableWord(trimmed)) continue;

    // Split the paragraph line at any intra-line sentence boundaries.
    for (const part of trimmed.split(SENTENCE_BOUNDARY_RE)) {
      const s = part.trim();
      if (s && isSpeakableWord(s)) result.push(s);
    }
  }
  return result.length > 0 ? result : [text.trim()].filter(Boolean);
}

/** Return the ordered list of speakable whitespace-split tokens from a sentence. */
function getSpeakableWords(sentence: string): string[] {
  return sentence.split(/\s+/).filter(isSpeakableWord);
}

/**
 * Map N source speakable words onto M Whisper word timestamps using
 * character-position proportional alignment.
 *
 * This handles vocabulary mismatches that survive even with a Whisper prompt:
 *   "per cent" (2 src words, 7 chars) → "percent" (1 whisper word, 7 chars)
 *   "gap-year"  (1 src word,  8 chars) → "gap" "year" (2 whisper words, 7 chars)
 *
 * Each source word is assigned a [start, end] time by interpolating within
 * the Whisper word(s) that occupy the same proportional character position.
 */
function mapWordsToTimings(
  sourceWords: string[],
  whisperWords: WhisperWord[]
): WordTimestamp[] {
  const N = sourceWords.length;
  const M = whisperWords.length;
  if (N === 0 || M === 0) return [];

  // Cumulative char counts for source words
  const srcCum = [0];
  for (const w of sourceWords) srcCum.push(srcCum[srcCum.length - 1] + w.length);
  const srcTotal = srcCum[N];

  // Cumulative char counts for Whisper words (minimum 1 to avoid zero-div)
  const wCum = [0];
  for (const w of whisperWords) {
    const len = Math.max(w.word.trim().length, 1);
    wCum.push(wCum[wCum.length - 1] + len);
  }
  const wTotal = wCum[M];

  const result: WordTimestamp[] = [];

  for (let i = 0; i < N; i++) {
    const fStart = srcCum[i] / srcTotal;
    const fEnd   = srcCum[i + 1] / srcTotal;

    // Map source char fractions → Whisper char space
    const wFStart = fStart * wTotal;
    const wFEnd   = fEnd   * wTotal;

    // Whisper word index whose range contains wFStart
    let wsi = 0;
    while (wsi < M - 1 && wCum[wsi + 1] <= wFStart) wsi++;

    // Whisper word index whose range contains wFEnd
    let wei = M - 1;
    while (wei > 0 && wCum[wei] >= wFEnd) wei--;

    // Interpolate start time within Whisper word wsi
    const wsRange = wCum[wsi + 1] - wCum[wsi];
    const wsRel   = wsRange > 0 ? Math.max(0, (wFStart - wCum[wsi]) / wsRange) : 0;
    const actualStart =
      whisperWords[wsi].start + wsRel * (whisperWords[wsi].end - whisperWords[wsi].start);

    // Interpolate end time within Whisper word wei
    const weRange = wCum[wei + 1] - wCum[wei];
    const weRel   = weRange > 0 ? Math.min(1, (wFEnd - wCum[wei]) / weRange) : 1;
    const actualEnd =
      whisperWords[wei].start + weRel * (whisperWords[wei].end - whisperWords[wei].start);

    result.push({
      word:  sourceWords[i],
      start: actualStart,
      end:   Math.max(actualEnd, actualStart + 0.05), // guarantee non-zero duration
    });
  }

  return result;
}

/**
 * Re-scale a sentence's Whisper word timestamps so the first word starts at
 * exactly segStart and the last word ends at exactly segEnd.
 *
 * Why this matters for fast voices:
 *   Whisper's word-level detector becomes less precise as speech speeds up —
 *   word boundaries can drift by 100–300 ms relative to the true audio
 *   position.  Sentence (segment) boundaries are estimated at a coarser
 *   acoustic resolution and are considerably more reliable.  Anchoring the
 *   word timestamps to those known-good boundaries corrects the drift while
 *   preserving the relative ordering and proportions of words within the
 *   sentence.
 */
function normalizeWordTimingsToSegment(
  words: WhisperWord[],
  segStart: number,
  segEnd: number
): WhisperWord[] {
  if (words.length === 0) return [];
  if (words.length === 1) return [{ ...words[0], start: segStart, end: segEnd }];

  const rawStart    = words[0].start;
  const rawEnd      = words[words.length - 1].end;
  const rawDuration = rawEnd - rawStart;
  if (rawDuration <= 0) return words;

  const segDuration = segEnd - segStart;
  return words.map((w) => ({
    ...w,
    start: segStart + ((w.start - rawStart) / rawDuration) * segDuration,
    end:   segStart + ((w.end   - rawStart) / rawDuration) * segDuration,
  }));
}

/**
 * Fallback: distribute [startTime, endTime] proportionally by character length.
 * Used only when Whisper returns no words for a sentence's time range.
 */
function distributeTimingToWords(
  sentenceText: string,
  startTime: number,
  endTime: number
): WordTimestamp[] {
  const words = getSpeakableWords(sentenceText);
  if (words.length === 0) return [];

  const duration   = endTime - startTime;
  const totalChars = words.reduce((sum, w) => sum + w.length, 0);
  const result: WordTimestamp[] = [];
  let currentTime = startTime;

  for (const word of words) {
    const wordDuration =
      totalChars > 0 ? (word.length / totalChars) * duration : duration / words.length;
    result.push({ word, start: currentTime, end: currentTime + wordDuration });
    currentTime += wordDuration;
  }
  return result;
}

/**
 * Transcribe a single audio chunk with Whisper and return word/segment
 * timestamps shifted by `timeOffset` (the cumulative duration of all
 * preceding chunks).  Processing each chunk separately avoids the
 * ID3-header-splice problem that occurs when multiple MP3 files are
 * byte-concatenated: ffmpeg/Whisper may treat the second ID3 header as a
 * new stream origin, causing timestamps to restart mid-file and leaving the
 * first N paragraphs without reliable word-level timing data.
 */
async function transcribeChunk(
  chunkText: string,
  buffer: ArrayBuffer,
  timeOffset: number,
  apiKey: string,
  baseUrl: string,
): Promise<{ words: WhisperWord[]; segments: WhisperSegment[]; duration: number }> {
  const audioBlob = new Blob([buffer], { type: "audio/mpeg" });
  const formData = new FormData();
  formData.append("file", audioBlob, "audio.mp3");
  formData.append("model", "whisper-1");
  formData.append("response_format", "verbose_json");
  formData.append("timestamp_granularities[]", "word");
  formData.append("timestamp_granularities[]", "segment");
  // Use this chunk's own text as the prompt so Whisper's vocabulary bias is
  // tight — a 900-char window of the full text may not even reach this chunk.
  const prompt = chunkText.replace(/•/g, "").replace(/\s+/g, " ").trim().slice(0, 900);
  formData.append("prompt", prompt);

  const response = await fetch(`${baseUrl}/audio/transcriptions`, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}` },
    body: formData,
  });

  if (!response.ok) {
    console.error("Whisper chunk transcription failed:", response.status);
    return { words: [], segments: [], duration: 0 };
  }

  const data = (await response.json()) as {
    words?:    WhisperWord[];
    segments?: WhisperSegment[];
    duration?: number;
  };

  const rawWords = data.words    ?? [];
  const rawSegs  = data.segments ?? [];

  // Shift every timestamp by the running time offset so timestamps are
  // absolute within the combined audio file.
  const words    = rawWords.map((w) => ({ ...w, start: w.start + timeOffset, end: w.end + timeOffset }));
  const segments = rawSegs.map( (s) => ({ ...s, start: s.start + timeOffset, end: s.end + timeOffset }));

  // Whisper's verbose_json includes a top-level `duration` field for the
  // chunk; fall back to the last word/segment end if it's missing.
  const duration =
    data.duration ??
    (rawWords.length > 0
      ? rawWords[rawWords.length - 1].end
      : rawSegs.length > 0
        ? rawSegs[rawSegs.length - 1].end
        : 0);

  return { words, segments, duration };
}

export async function alignAudio(
  audioChunks: AudioChunk[],
  text: string
): Promise<{ segments: Segment[]; audioDurationSeconds: number }> {
  const apiKey = process.env.TTS_API_KEY;
  const baseUrl = (
    process.env.TTS_BASE_URL || "https://api.openai.com/v1"
  ).replace(/\/$/, "");

  if (!apiKey) return { segments: [], audioDurationSeconds: 0 };

  try {
    // ── Per-chunk Whisper transcription ──────────────────────────────────────
    // All chunks are transcribed in parallel.  Each chunk is a self-contained
    // MP3 whose timestamps start at t=0, so Whisper's word detector works on
    // clean, uninterrupted audio — no ID3-splice issues.  Each result carries
    // a `duration` field; we accumulate those to compute absolute offsets
    // before merging into the combined word/segment arrays below.
    const chunkResults = await Promise.all(
      audioChunks.map((chunk) =>
        transcribeChunk(chunk.text, chunk.buffer, 0, apiKey, baseUrl)
      )
    );

    // Compute cumulative offsets from reported chunk durations and re-apply.
    let cumOffset = 0;
    const whisperWords:    WhisperWord[]    = [];
    const whisperSegments: WhisperSegment[] = [];
    for (const result of chunkResults) {
      const off = cumOffset;
      for (const w of result.words)    whisperWords.push(   { ...w, start: w.start + off, end: w.end + off });
      for (const s of result.segments) whisperSegments.push({ ...s, start: s.start + off, end: s.end + off });
      cumOffset += result.duration;
    }
    const totalDuration = cumOffset;

    if (whisperWords.length === 0) return { segments: [], audioDurationSeconds: totalDuration };

    const sourceSentences = splitIntoSentences(text);
    if (sourceSentences.length === 0) return { segments: [], audioDurationSeconds: 0 };

    // Best estimate of the true audio end: last word end or last segment end,
    // whichever is later.  Used as the upper bound for Strategy B and as the
    // return value when Strategy A is active.
    const audioEnd = whisperSegments.length > 0
      ? Math.max(
          whisperWords[whisperWords.length - 1].end,
          whisperSegments[whisperSegments.length - 1].end,
        )
      : whisperWords[whisperWords.length - 1].end;

    // ── Sentence timing ──────────────────────────────────────────────────────
    const sentenceTimings: { start: number; end: number }[] = [];

    // A 1-to-1 mapping is only safe when Whisper segmented the audio the same
    // way as our source-sentence split.  We validate this by checking that
    // each Whisper segment's text length is within 2× of its paired source
    // sentence.  A larger ratio means Whisper merged or split differently —
    // common when a heading has no terminal punctuation (it gets absorbed into
    // the next segment) and another long sentence gets split to keep the total
    // count equal, making the 1-to-1 assignment silently wrong.
    const segmentsAlignWithSentences =
      whisperSegments.length === sourceSentences.length &&
      sourceSentences.every((src, i) => {
        const srcLen = src.trim().length;
        const wsLen  = whisperSegments[i].text.trim().length;
        return wsLen <= srcLen * 2 && srcLen <= wsLen * 2;
      });

    if (segmentsAlignWithSentences) {
      // Confirmed 1-to-1 match — use Whisper segment timing directly.
      // TTS has no leading silence so speech begins at t≈0; force the first
      // sentence to start there regardless of what Whisper reports (Whisper
      // routinely places its first segment start at 0.1–0.3 s due to its
      // internal silence-detection heuristic).
      for (let i = 0; i < whisperSegments.length; i++) {
        const ws = whisperSegments[i];
        sentenceTimings.push({
          start: i === 0 ? 0 : ws.start,
          end: ws.end,
        });
      }
    } else {
      // Count mismatch — distribute source-sentence boundaries across the
      // audio timeline using *actual Whisper word timestamps* as anchors
      // rather than a purely proportional interpolation.
      //
      // For each sentence boundary at cumulative source-word fraction F, we
      // locate the Whisper word at the same proportional position in the
      // Whisper word list and use its `end` time.  This is far more accurate
      // than linear interpolation because the Whisper words already carry the
      // true per-word speech timing for this audio.
      //
      // Always anchor at t=0 — TTS has no leading silence; speech begins at
      // t≈0, so 0 is the correct origin regardless of what Whisper reports.
      const timelineEnd = Math.max(totalDuration, audioEnd);

      const sentenceWordCounts = sourceSentences.map((s) => getSpeakableWords(s).length);
      const totalSrcWords = sentenceWordCounts.reduce((a, b) => a + b, 0);

      let cumSrcWords = 0;
      for (let i = 0; i < sourceSentences.length; i++) {
        cumSrcWords += sentenceWordCounts[i];

        const tStart = i === 0 ? 0 : sentenceTimings[i - 1].end;

        let tEnd: number;
        if (i === sourceSentences.length - 1) {
          // Last sentence always runs to the true audio end.
          tEnd = timelineEnd;
        } else {
          const fraction = cumSrcWords / totalSrcWords;
          if (whisperWords.length > 0) {
            // Find the Whisper word at this proportional position and use its
            // end time as the sentence boundary.  clamp to valid range.
            const targetIdx = Math.min(
              Math.max(0, Math.round(fraction * whisperWords.length) - 1),
              whisperWords.length - 1,
            );
            tEnd = whisperWords[targetIdx].end;
          } else {
            tEnd = fraction * timelineEnd;
          }
        }

        sentenceTimings.push({ start: tStart, end: Math.max(tEnd, tStart + 0.01) });
      }
      // Clamp the last sentence to the true audio end (safety net).
      if (sentenceTimings.length > 0) {
        sentenceTimings[sentenceTimings.length - 1].end = timelineEnd;
      }
    }

    // ── Per-word timing ──────────────────────────────────────────────────────
    // For each source sentence, collect the Whisper words whose centre time
    // falls within the sentence's range, then map source speakable words onto
    // those Whisper words via character-position alignment.
    // Words displayed are always from the SOURCE TEXT — Whisper words are used
    // only as timing guides, never as display text.
    const segments: Segment[] = [];

    for (let i = 0; i < sourceSentences.length; i++) {
      const { start, end } = sentenceTimings[i];
      const speakable = getSpeakableWords(sourceSentences[i]);
      if (speakable.length === 0) continue;

      const sentenceWhisperWords = whisperWords.filter((w) => {
        const centre = (w.start + w.end) / 2;
        return centre >= start && centre <= end;
      });

      // Re-anchor the word timestamps to the sentence boundaries before
      // mapping.  This corrects word-level drift that Whisper produces for
      // fast speech, while preserving the relative proportions between words.
      const anchoredWords = normalizeWordTimingsToSegment(sentenceWhisperWords, start, end);

      const timedWords =
        anchoredWords.length > 0
          ? mapWordsToTimings(speakable, anchoredWords)
          : distributeTimingToWords(sourceSentences[i], start, end);

      if (timedWords.length > 0) {
        segments.push({
          text:      sourceSentences[i],
          startTime: start,
          endTime:   end,
          words:     timedWords,
        });
      }
    }

    // ── Global gap-filling pass ──────────────────────────────────────────────
    // After all per-segment word timings are computed, walk the flat word list
    // and extend each word's `end` to meet the next word's `start`.  This
    // eliminates the sub-frame silences between words where no word would
    // otherwise be highlighted (the "flash-off" flicker visible on screen).
    // We also snap the very first word to t=0 so the highlight is live from
    // the moment audio starts — Whisper commonly places the first word at
    // ~0.05–0.2 s even though TTS speech begins at t≈0.
    //
    // Each word object is copied (not mutated) before modification so the
    // original timing data stored in each Segment is replaced cleanly.
    {
      // Build a flat, independent copy of all word objects.
      const flatWords: WordTimestamp[] = [];
      for (const seg of segments) {
        for (const w of seg.words) flatWords.push({ ...w });
      }

      // Snap first word to t=0.
      if (flatWords.length > 0 && flatWords[0].start > 0) {
        flatWords[0].start = 0;
      }

      // Fill every gap: extend word[i].end to word[i+1].start.
      for (let i = 0; i < flatWords.length - 1; i++) {
        if (flatWords[i].end < flatWords[i + 1].start) {
          flatWords[i].end = flatWords[i + 1].start;
        }
      }

      // Write the gap-filled words back into each segment.
      let flatIdx = 0;
      for (const seg of segments) {
        const count = seg.words.length;
        seg.words = flatWords.slice(flatIdx, flatIdx + count);
        flatIdx += count;
      }
    }

    // Use totalDuration (sum of per-chunk Whisper durations) as the canonical
    // audio length; it matches what the browser will actually play.
    return { segments, audioDurationSeconds: totalDuration };
  } catch (error) {
    console.error("Audio alignment error:", error);
    return { segments: [], audioDurationSeconds: 0 };
  }
}
