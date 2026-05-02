import { mkdir, writeFile } from "fs/promises";
import { join } from "path";
import { nanoid } from "nanoid";

import { VOICE_MAP, VOICE_OPTIONS } from "./constants";
import type { Segment } from "@/types/karaoke";

export { VOICE_MAP, VOICE_OPTIONS };

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
): Promise<{ words: WhisperWord[]; duration: number }> {
  const audioBlob = new Blob([buffer], { type: "audio/mpeg" });
  const formData = new FormData();
  formData.append("file", audioBlob, "audio.mp3");
  formData.append("model", "whisper-1");
  formData.append("response_format", "verbose_json");
  formData.append("timestamp_granularities[]", "word");
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
    return { words: [], duration: 0 };
  }

  const data = (await response.json()) as {
    words?:    WhisperWord[];
    duration?: number;
  };

  const rawWords = data.words ?? [];

  // Shift every timestamp by the running time offset so timestamps are
  // absolute within the combined audio file.
  const words = rawWords.map((w) => ({ ...w, start: w.start + timeOffset, end: w.end + timeOffset }));

  // verbose_json always includes a top-level `duration`; fall back to the
  // last word's end time if it is somehow absent.
  const duration =
    data.duration ?? (rawWords.length > 0 ? rawWords[rawWords.length - 1].end : 0);

  return { words, duration };
}

export async function alignAudio(
  audioChunks: AudioChunk[]
): Promise<{ segments: Segment[]; audioDurationSeconds: number }> {
  const apiKey = process.env.TTS_API_KEY;
  const baseUrl = (
    process.env.TTS_BASE_URL || "https://api.openai.com/v1"
  ).replace(/\/$/, "");

  if (!apiKey) return { segments: [], audioDurationSeconds: 0 };

  try {
    // ── Per-chunk Whisper transcription ──────────────────────────────────────
    // Each chunk is transcribed independently against its own clean MP3,
    // avoiding the ID3-splice timestamp problem that occurs when concatenated
    // buffers are sent as one file.
    const chunkResults = await Promise.all(
      audioChunks.map((chunk) =>
        transcribeChunk(chunk.text, chunk.buffer, 0, apiKey, baseUrl)
      )
    );

    // Merge word arrays, shifting each chunk's timestamps by the cumulative
    // duration of all preceding chunks so every timestamp is absolute.
    let cumOffset = 0;
    const whisperWords: WhisperWord[] = [];
    for (const result of chunkResults) {
      const off = cumOffset;
      for (const w of result.words)
        whisperWords.push({ ...w, start: w.start + off, end: w.end + off });
      cumOffset += result.duration;
    }
    const totalDuration = cumOffset;

    if (whisperWords.length === 0)
      return { segments: [], audioDurationSeconds: totalDuration };

    // Return one flat segment containing every Whisper word.
    // Timing is Whisper's own — perfectly accurate, zero remapping.
    const segment: Segment = {
      text: whisperWords.map((w) => w.word).join(" "),
      startTime: whisperWords[0].start,
      endTime: whisperWords[whisperWords.length - 1].end,
      words: whisperWords.map((w) => ({ word: w.word, start: w.start, end: w.end })),
    };

    return { segments: [segment], audioDurationSeconds: totalDuration };
  } catch (error) {
    console.error("Audio alignment error:", error);
    return { segments: [], audioDurationSeconds: 0 };
  }
}
