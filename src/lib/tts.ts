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

interface WhisperSegment {
  text: string;
  start: number;
  end: number;
  words?: WhisperWord[];
}

interface WhisperWord {
  word: string;
  start: number;
  end: number;
}

function normalize(w: string): string {
  return w.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function splitSentences(text: string): string[] {
  const parts = text.match(/[^.!?\n]+[.!?]?\n?/g);
  if (!parts) return text.trim() ? [text.trim()] : [];
  return parts.map((p) => p.trim()).filter(Boolean);
}

function mapOriginalToWhisper(
  originalWords: string[],
  whisperWords: WhisperWord[]
): WordTimestamp[] {
  const result: WordTimestamp[] = [];
  let oi = 0;
  let wi = 0;

  while (oi < originalWords.length && wi < whisperWords.length) {
    const origNorm = normalize(originalWords[oi]);
    const whispNorm = normalize(whisperWords[wi].word);

    if (origNorm === whispNorm) {
      result.push({
        word: originalWords[oi],
        start: whisperWords[wi].start,
        end: whisperWords[wi].end,
      });
      oi++;
      wi++;
      continue;
    }

    let matched = false;

    for (let take = 2; take <= 4 && oi + take <= originalWords.length; take++) {
      const joined = originalWords
        .slice(oi, oi + take)
        .map(normalize)
        .join("");
      if (joined === whispNorm) {
        const startTime = whisperWords[wi].start;
        const endTime = whisperWords[wi].end;
        const totalLen = originalWords
          .slice(oi, oi + take)
          .reduce((s, w) => s + w.length, 0);
        let curStart = startTime;
        for (let j = 0; j < take; j++) {
          const proportion = originalWords[oi + j].length / totalLen;
          const wordEnd = curStart + (endTime - startTime) * proportion;
          result.push({
            word: originalWords[oi + j],
            start: curStart,
            end: wordEnd,
          });
          curStart = wordEnd;
        }
        oi += take;
        wi++;
        matched = true;
        break;
      }
    }

    if (matched) continue;

    for (let take = 2; take <= 4 && wi + take <= whisperWords.length; take++) {
      const joined = whisperWords
        .slice(wi, wi + take)
        .map((w) => normalize(w.word))
        .join("");
      if (joined === origNorm) {
        result.push({
          word: originalWords[oi],
          start: whisperWords[wi].start,
          end: whisperWords[wi + take - 1].end,
        });
        oi++;
        wi += take;
        matched = true;
        break;
      }
    }

    if (matched) continue;

    result.push({
      word: originalWords[oi],
      start: whisperWords[wi].start,
      end: whisperWords[wi].end,
    });
    oi++;
    wi++;
  }

  const lastEnd =
    result.length > 0 ? result[result.length - 1].end : 0;
  while (oi < originalWords.length) {
    result.push({
      word: originalWords[oi],
      start: lastEnd,
      end: lastEnd,
    });
    oi++;
  }

  return result;
}

export async function alignAudio(
  audioPath: string,
  originalText: string
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
    formData.append("timestamp_granularities[]", "word");
    formData.append("timestamp_granularities[]", "segment");

    const response = await fetch(`${baseUrl}/audio/transcriptions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
      body: formData,
    });

    if (!response.ok) {
      console.error("Whisper alignment failed:", response.status);
      return [];
    }

    const data = (await response.json()) as {
      segments?: WhisperSegment[];
    };

    if (!data.segments || data.segments.length === 0) return [];

    const allWhisperWords = data.segments.flatMap(
      (seg) => seg.words ?? []
    );

    if (allWhisperWords.length === 0) return [];

    const sentences = splitSentences(originalText);
    const allOriginalWords = sentences.flatMap((s) =>
      s.split(/\s+/).filter(Boolean)
    );

    const mappedWords = mapOriginalToWhisper(
      allOriginalWords,
      allWhisperWords
    );

    let wordIdx = 0;
    return sentences
      .map((sentenceText) => {
        const count = sentenceText.split(/\s+/).filter(Boolean).length;
        const words = mappedWords.slice(wordIdx, wordIdx + count);
        wordIdx += count;
        if (words.length === 0) return null;
        return {
          text: sentenceText,
          startTime: words[0].start,
          endTime: words[words.length - 1].end,
          words,
        };
      })
      .filter((s): s is Segment => s !== null);
  } catch (error) {
    console.error("Audio alignment error:", error);
    return [];
  }
}
