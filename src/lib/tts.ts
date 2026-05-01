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

/** A word token is "speakable" if it contains at least one alphanumeric character.
 *  This filters out bullets (•), em dashes (—), colons alone, etc. */
function isSpeakableWord(token: string): boolean {
  return /[a-zA-Z0-9]/.test(token);
}

/**
 * Split source text into sentences by line.
 * Each non-empty line becomes its own sentence — this gives fine-grained
 * segments that match how Whisper naturally pauses between lines.
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

/**
 * Distribute a time range [startTime, endTime] proportionally across the
 * speakable words in a sentence, weighted by character length.
 * Non-speakable tokens (bullets, dashes, etc.) are skipped.
 */
function distributeTimingToWords(
  sentenceText: string,
  startTime: number,
  endTime: number
): WordTimestamp[] {
  const tokens = sentenceText.split(/\s+/).filter((t) => t.length > 0);
  const speakable = tokens.filter(isSpeakableWord);
  if (speakable.length === 0) return [];

  const duration = endTime - startTime;
  const totalChars = speakable.reduce((sum, w) => sum + w.length, 0);
  const result: WordTimestamp[] = [];
  let currentTime = startTime;

  for (const word of speakable) {
    const wordDuration =
      totalChars > 0
        ? (word.length / totalChars) * duration
        : duration / speakable.length;
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
    // Request sentence-level segments only — word text from Whisper is not
    // used for display, so word-level timestamps are not needed here.
    formData.append("timestamp_granularities[]", "segment");

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
      segments?: { text: string; start: number; end: number }[];
    };

    const whisperSegments = data.segments ?? [];
    if (whisperSegments.length === 0) return [];

    const sourceSentences = splitIntoSentences(text);
    if (sourceSentences.length === 0) return [];

    const audioStart = whisperSegments[0].start;
    const audioEnd = whisperSegments[whisperSegments.length - 1].end;
    const audioDuration = audioEnd - audioStart;

    // Assign a [startTime, endTime] range to each source sentence.
    // When Whisper segment count matches source sentence count, use 1-to-1.
    // Otherwise fall back to distributing total audio time proportionally
    // by the number of speakable characters in each source sentence.
    const sentenceTimings: { start: number; end: number }[] = [];

    if (whisperSegments.length === sourceSentences.length) {
      for (const ws of whisperSegments) {
        sentenceTimings.push({ start: ws.start, end: ws.end });
      }
    } else {
      const charCounts = sourceSentences.map((s) =>
        s
          .split(/\s+/)
          .filter(isSpeakableWord)
          .reduce((sum, w) => sum + w.length, 0)
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

    // Build Segment[] — words come from the SOURCE TEXT, not Whisper.
    // This guarantees that the original wording ("per cent", "gap-year", etc.)
    // is always preserved regardless of how Whisper transcribed the audio.
    const segments: Segment[] = [];
    for (let i = 0; i < sourceSentences.length; i++) {
      const { start, end } = sentenceTimings[i];
      const timedWords = distributeTimingToWords(sourceSentences[i], start, end);
      if (timedWords.length > 0) {
        segments.push({
          text: sourceSentences[i],
          startTime: start,
          endTime: end,
          words: timedWords,
        });
      }
    }

    return segments;
  } catch (error) {
    console.error("Audio alignment error:", error);
    return [];
  }
}
