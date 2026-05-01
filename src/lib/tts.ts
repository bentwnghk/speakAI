import { mkdir, writeFile, readFile } from "fs/promises";
import { join } from "path";
import { nanoid } from "nanoid";

import { VOICE_MAP, VOICE_OPTIONS } from "./constants";
import type { Segment, WordTimestamp } from "@/types/karaoke";

export { VOICE_MAP, VOICE_OPTIONS };

export function splitText(text: string, maxChunkSize = 4000): string[] {
  const chunks: string[] = [];
  const paragraphs = text.split("\n\n");

  for (const paragraph of paragraphs) {
    if (!paragraph.trim()) continue;

    if (paragraph.length <= maxChunkSize) {
      chunks.push(paragraph);
    } else {
      const sentences = paragraph.split(/(?<=[.!?])\s+/);
      let currentChunk = "";
      for (const sentence of sentences) {
        if (sentence.length > maxChunkSize) {
          const words = sentence.split(" ");
          for (const word of words) {
            if (currentChunk.length + word.length + 1 > maxChunkSize) {
              chunks.push(currentChunk);
              currentChunk = word;
            } else {
              currentChunk += (currentChunk ? " " : "") + word;
            }
          }
          if (currentChunk) {
            chunks.push(currentChunk);
            currentChunk = "";
          }
        } else if (currentChunk.length + sentence.length + 1 > maxChunkSize) {
          chunks.push(currentChunk);
          currentChunk = sentence;
        } else {
          currentChunk += (currentChunk ? " " : "") + sentence;
        }
      }
      if (currentChunk) {
        chunks.push(currentChunk);
      }
    }
  }

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

export async function generateTtsAudio(
  text: string,
  voice: string,
  speedPercent: number
): Promise<{ audioPath: string; cost: string }> {
  const apiKey = process.env.TTS_API_KEY;
  const baseUrl = (process.env.TTS_BASE_URL || "https://api.openai.com/v1").replace(/\/$/, "");

  if (!apiKey) {
    throw new Error("TTS_API_KEY is not configured");
  }

  const actualVoice = VOICE_MAP[voice] || "nova";
  const speed = speedPercent / 100;

  const chunks = splitText(text);
  const audioBuffers: ArrayBuffer[] = [];

  const concurrencyLimit = 10;
  const batches: string[][] = [];
  for (let i = 0; i < chunks.length; i += concurrencyLimit) {
    batches.push(chunks.slice(i, i + concurrencyLimit));
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
  const ttsCost = ((characters / 1_000_000) * 15 * 7.8).toFixed(2);

  return {
    audioPath: `data/audio/${audioId}.mp3`,
    cost: ttsCost,
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
 * Split source text into sentences by line.
 * Each non-empty line becomes its own sentence — matches how Whisper
 * naturally pauses between lines of text.
 */
function splitIntoSentences(text: string): string[] {
  const result: string[] = [];
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (trimmed && isSpeakableWord(trimmed)) {
      result.push(trimmed);
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

export async function alignAudio(
  audioPath: string,
  text: string
): Promise<Segment[]> {
  const apiKey = process.env.TTS_API_KEY;
  const baseUrl = (
    process.env.TTS_BASE_URL || "https://api.openai.com/v1"
  ).replace(/\/$/, "");

  if (!apiKey) return [];

  try {
    const absolutePath = join(process.cwd(), audioPath);
    const audioBuffer = await readFile(absolutePath);
    const audioBlob = new Blob([audioBuffer], { type: "audio/mpeg" });

    const formData = new FormData();
    formData.append("file", audioBlob, "audio.mp3");
    formData.append("model", "whisper-1");
    formData.append("response_format", "verbose_json");
    // Both granularities:
    //   segment → accurate sentence-level start/end boundaries
    //   word    → accurate per-word timestamps within each sentence
    formData.append("timestamp_granularities[]", "word");
    formData.append("timestamp_granularities[]", "segment");
    // Providing the source text as prompt biases Whisper to follow the
    // original vocabulary so word sequences stay close to the source,
    // minimising alignment drift ("per cent" rather than "percent", etc.)
    const prompt = text.replace(/•/g, "").replace(/\s+/g, " ").trim().slice(0, 900);
    formData.append("prompt", prompt);

    const response = await fetch(`${baseUrl}/audio/transcriptions`, {
      method: "POST",
      headers: { Authorization: `Bearer ${apiKey}` },
      body: formData,
    });

    if (!response.ok) {
      console.error("Whisper alignment failed:", response.status);
      return [];
    }

    const data = (await response.json()) as {
      words?:    WhisperWord[];
      segments?: WhisperSegment[];
    };

    const whisperWords    = data.words    ?? [];
    const whisperSegments = data.segments ?? [];

    if (whisperWords.length === 0) return [];

    const sourceSentences = splitIntoSentences(text);
    if (sourceSentences.length === 0) return [];

    const audioStart    = whisperWords[0].start;
    const audioEnd      = whisperWords[whisperWords.length - 1].end;
    const audioDuration = audioEnd - audioStart;

    // ── Sentence timing ──────────────────────────────────────────────────────
    // When Whisper segment count matches source sentence count use 1-to-1
    // segment timing (most accurate).  Otherwise distribute total duration
    // proportionally by source sentence character count (graceful fallback).
    const sentenceTimings: { start: number; end: number }[] = [];

    if (whisperSegments.length === sourceSentences.length) {
      for (const ws of whisperSegments) {
        sentenceTimings.push({ start: ws.start, end: ws.end });
      }
    } else {
      const charCounts = sourceSentences.map((s) =>
        getSpeakableWords(s).reduce((sum, w) => sum + w.length, 0)
      );
      const totalChars = charCounts.reduce((a, b) => a + b, 0);
      let currentTime = audioStart;
      for (const charCount of charCounts) {
        const duration =
          totalChars > 0
            ? (charCount / totalChars) * audioDuration
            : audioDuration / sourceSentences.length;
        sentenceTimings.push({ start: currentTime, end: currentTime + duration });
        currentTime += duration;
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

      const timedWords =
        sentenceWhisperWords.length > 0
          ? mapWordsToTimings(speakable, sentenceWhisperWords)
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

    return segments;
  } catch (error) {
    console.error("Audio alignment error:", error);
    return [];
  }
}
