import { mkdir, writeFile } from "fs/promises";
import { join } from "path";
import { nanoid } from "nanoid";
import * as sdk from "microsoft-cognitiveservices-speech-sdk";

import { VOICE_MAP, VOICE_OPTIONS } from "./constants";
import type { Segment, WordTimestamp } from "@/types/karaoke";

export { VOICE_MAP, VOICE_OPTIONS };

const AZURE_VOICE_MAP: Record<string, string> = {
  "Female 1": "en-US-NovaTurboMultilingualNeural",
  "Male 1": "en-US-AlloyTurboMultilingualNeural",
  "Female 2": "en-US-PhoebeMultilingualNeural",
  "Male 2": "en-US-AdamMultilingualNeural",
  "Female 3": "en-US-AvaMultilingualNeural",
  "Male 3": "en-GB-OllieMultilingualNeural",
};

// ── Azure endpoint pool ──────────────────────────────────────────────────────
// Supports an arbitrary number of Azure Speech endpoints for round-robin load
// distribution.  Each endpoint is a (key, region) pair read from env vars:
//
//   AZURE_SPEECH_KEY_1 / AZURE_SPEECH_REGION_1
//   AZURE_SPEECH_KEY_2 / AZURE_SPEECH_REGION_2
//   ...
//   AZURE_SPEECH_KEY_N / AZURE_SPEECH_REGION_N
//
// The unnumbered AZURE_SPEECH_KEY / AZURE_SPEECH_REGION are supported as a
// single-endpoint fallback for backward compatibility.
//
// Endpoints are collected at module initialisation time (once per process).

interface AzureEndpoint { key: string; region: string }

function loadAzureEndpoints(): AzureEndpoint[] {
  const endpoints: AzureEndpoint[] = [];

  // Scan AZURE_SPEECH_KEY_1 … AZURE_SPEECH_KEY_N (up to 20).
  for (let i = 1; i <= 20; i++) {
    const key    = process.env[`AZURE_SPEECH_KEY_${i}`];
    const region = process.env[`AZURE_SPEECH_REGION_${i}`];
    if (key && region) endpoints.push({ key, region });
  }

  // Fall back to unnumbered names when no numbered ones are present.
  if (endpoints.length === 0) {
    const key    = process.env.AZURE_SPEECH_KEY;
    const region = process.env.AZURE_SPEECH_REGION;
    if (key && region) endpoints.push({ key, region });
  }

  return endpoints;
}

const AZURE_ENDPOINTS: AzureEndpoint[] = loadAzureEndpoints();

// Module-level round-robin cursor.  In a long-running Node.js process this
// persists across requests so consecutive chunks and consecutive requests are
// each dispatched to the next endpoint in the pool.
let rrCursor = 0;

function pickNextEndpoint(): AzureEndpoint {
  if (AZURE_ENDPOINTS.length === 0) {
    throw new Error(
      "No Azure Speech endpoints configured. " +
      "Set AZURE_SPEECH_KEY_1 + AZURE_SPEECH_REGION_1 " +
      "(and optionally _2 … _N for a multi-endpoint pool).",
    );
  }
  const ep = AZURE_ENDPOINTS[rrCursor % AZURE_ENDPOINTS.length];
  rrCursor++;
  return ep;
}
// ────────────────────────────────────────────────────────────────────────────

const configuredAzureTtsPrice = Number(
  process.env.AZURE_SPEECH_PRICE_USD_PER_1M_CHARS ?? "16",
);
const AZURE_TTS_USD_PER_1M_CHARS =
  Number.isFinite(configuredAzureTtsPrice) && configuredAzureTtsPrice > 0
    ? configuredAzureTtsPrice
    : 16;

export function estimateTtsCost(text: string): number {
  return Math.max((text.length / 1_000_000) * AZURE_TTS_USD_PER_1M_CHARS * 7.8, 0.01);
}

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
 * Why this matters for TTS + word-boundary timing:
 *   - A heading without punctuation causes the TTS engine to run straight
 *     into the next sentence with no prosodic pause, making the heading
 *     indistinguishable from body text.
 *   - Adding a period gives the TTS engine a natural pause cue and gives the
 *     word-boundary stream a cleaner sentence-level reading cadence.
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
  // paragraph-break pauses and Azure emits a single ordered word-boundary
  // stream for each chunk.
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
      // single TTS call, producing one unbroken MP3 and one ordered Azure
      // word-boundary stream from the very first word.
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

export interface AudioChunk {
  text: string;
  buffer: ArrayBuffer;
}

interface AzureWordBoundary {
  word: string;
  start: number;
  end: number;
  textOffset: number;
}

interface AzureAudioChunk extends AudioChunk {
  boundaries: AzureWordBoundary[];
  duration: number;
}

function getAzureVoiceName(voice: string): string {
  const envKey = `AZURE_SPEECH_VOICE_${voice.toUpperCase().replace(/[^A-Z0-9]+/g, "_")}`;
  return process.env[envKey] || AZURE_VOICE_MAP[voice] || AZURE_VOICE_MAP["Female 1"];
}

function escapeXml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function buildAzureSsml(text: string, voiceName: string, speedPercent: number): string {
  const clampedSpeed = Math.min(Math.max(speedPercent, 50), 200);
  const rate = `${clampedSpeed >= 100 ? "+" : ""}${clampedSpeed - 100}%`;

  return `<speak version="1.0" xml:lang="en-US"><voice name="${voiceName}"><prosody rate="${rate}">${escapeXml(text)}</prosody></voice></speak>`;
}

async function synthesizeAzureChunk(
  text: string,
  voiceName: string,
  speedPercent: number,
  subscriptionKey: string,
  region: string,
): Promise<AzureAudioChunk> {
  const speechConfig = sdk.SpeechConfig.fromSubscription(subscriptionKey, region);
  speechConfig.speechSynthesisVoiceName = voiceName;
  speechConfig.speechSynthesisOutputFormat =
    sdk.SpeechSynthesisOutputFormat.Audio24Khz96KBitRateMonoMp3;

  const synthesizer = new sdk.SpeechSynthesizer(speechConfig, null);
  const boundaries: AzureWordBoundary[] = [];

  synthesizer.wordBoundary = (_sender, event) => {
    if (event.boundaryType !== sdk.SpeechSynthesisBoundaryType.Word) return;
    if (!isSpeakableWord(event.text)) return;

    const start = event.audioOffset / 10_000_000;
    const duration = event.duration / 10_000_000;
    boundaries.push({
      word: event.text,
      start,
      end: duration > 0 ? start + duration : start + 0.05,
      textOffset: event.textOffset,
    });
  };

  const ssml = buildAzureSsml(text, voiceName, speedPercent);

  try {
    return await new Promise<AzureAudioChunk>((resolve, reject) => {
      synthesizer.speakSsmlAsync(
        ssml,
        (result) => {
          synthesizer.close();

          if (result.reason !== sdk.ResultReason.SynthesizingAudioCompleted) {
            reject(new Error(result.errorDetails || "Azure Speech synthesis failed"));
            return;
          }

          const duration = result.audioDuration > 0
            ? result.audioDuration / 10_000_000
            : boundaries.length > 0
              ? boundaries[boundaries.length - 1].end
              : 0;

          resolve({
            text,
            buffer: result.audioData,
            boundaries,
            duration,
          });
        },
        (error) => {
          synthesizer.close();
          reject(new Error(error));
        },
      );
    });
  } catch (error) {
    synthesizer.close();
    throw error;
  }
}

function buildSegmentsFromAzureBoundaries(
  text: string,
  boundaries: AzureWordBoundary[],
  audioDurationSeconds: number,
): Segment[] {
  const sourceSentences = splitIntoSentences(text);
  const sentenceWords = sourceSentences.map((sentence) => getSpeakableWords(sentence));
  const sourceWords = sentenceWords.flat();
  if (sourceWords.length === 0) return [];

  const orderedBoundaries = [...boundaries].sort((a, b) => a.start - b.start);
  let timedWords: WordTimestamp[];

  if (orderedBoundaries.length === 0) {
    timedWords = distributeTimingToWords(text, 0, audioDurationSeconds);
  } else if (orderedBoundaries.length === sourceWords.length) {
    timedWords = sourceWords.map((word, index) => ({
      word,
      start: orderedBoundaries[index].start,
      end: orderedBoundaries[index].end,
    }));
  } else {
    // Azure occasionally expands abbreviations or normalizes tokens.  Map by
    // proportional character position against Azure's synthesis-time word
    // boundaries instead of falling back to post-hoc transcription.
    timedWords = mapWordsToTimings(sourceWords, orderedBoundaries);
  }

  if (timedWords.length === 0) return [];

  if (timedWords[0].start > 0) timedWords[0].start = 0;

  for (let i = 0; i < timedWords.length - 1; i++) {
    if (timedWords[i].end < timedWords[i + 1].start) {
      timedWords[i].end = timedWords[i + 1].start;
    }
  }

  const last = timedWords[timedWords.length - 1];
  if (audioDurationSeconds > last.start) {
    last.end = Math.max(last.end, audioDurationSeconds);
  }

  const segments: Segment[] = [];
  let wordOffset = 0;
  for (let i = 0; i < sourceSentences.length; i++) {
    const count = sentenceWords[i].length;
    if (count === 0) continue;

    const words = timedWords.slice(wordOffset, wordOffset + count);
    wordOffset += count;
    if (words.length === 0) continue;

    segments.push({
      text: sourceSentences[i],
      startTime: words[0].start,
      endTime: words[words.length - 1].end,
      words,
    });
  }

  return segments;
}

export async function generateTtsAudio(
  text: string,
  voice: string,
  speedPercent: number
): Promise<{
  audioPath: string;
  cost: string;
  audioDurationSeconds: number;
  chunks: AudioChunk[];
  segments: Segment[];
}> {
  // pickNextEndpoint() throws if the pool is empty, giving a clear startup error.
  const actualVoice = getAzureVoiceName(voice);

  const chunkTexts = splitText(text);
  const audioChunks: AzureAudioChunk[] = [];

  // Process chunks sequentially so cumulative audio offsets are deterministic.
  // Each chunk is dispatched to the next endpoint in the round-robin pool so
  // API usage (and billing) is spread evenly across all configured endpoints.
  for (const chunk of chunkTexts) {
    const { key, region } = pickNextEndpoint();
    audioChunks.push(
      await synthesizeAzureChunk(chunk, actualVoice, speedPercent, key, region),
    );
  }

  const audioBuffers = audioChunks.map((chunk) => chunk.buffer);

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

  let timeOffset = 0;
  const allBoundaries: AzureWordBoundary[] = [];
  for (const chunk of audioChunks) {
    for (const boundary of chunk.boundaries) {
      allBoundaries.push({
        ...boundary,
        start: boundary.start + timeOffset,
        end: boundary.end + timeOffset,
      });
    }
    timeOffset += chunk.duration;
  }

  const segments = buildSegmentsFromAzureBoundaries(text, allBoundaries, timeOffset);
  const ttsCost = estimateTtsCost(text);

  return {
    audioPath: `data/audio/${audioId}.mp3`,
    cost: ttsCost.toFixed(2),
    audioDurationSeconds: timeOffset,
    chunks: audioChunks.map((chunk) => ({ text: chunk.text, buffer: chunk.buffer })),
    segments,
  };
}

interface TimedWordBoundary {
  word: string;
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
 * Each non-empty line becomes its own sentence. This mirrors how extracted
 * documents usually represent headings and paragraphs, and lets Azure's word
 * boundaries be grouped back into source-text display segments.
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
 * Map N source speakable words onto M TTS word-boundary timestamps using
 * character-position proportional alignment.
 *
 * This handles vocabulary/token mismatches in the TTS event stream:
 *   "per cent" (2 src words, 7 chars) → "percent" (1 event word, 7 chars)
 *   "gap-year"  (1 src word,  8 chars) → "gap" "year" (2 event words, 7 chars)
 *
 * Each source word is assigned a [start, end] time by interpolating within
 * the event word(s) that occupy the same proportional character position.
 */
function mapWordsToTimings(
  sourceWords: string[],
  boundaryWords: TimedWordBoundary[]
): WordTimestamp[] {
  const N = sourceWords.length;
  const M = boundaryWords.length;
  if (N === 0 || M === 0) return [];

  // Cumulative char counts for source words
  const srcCum = [0];
  for (const w of sourceWords) srcCum.push(srcCum[srcCum.length - 1] + w.length);
  const srcTotal = srcCum[N];

  // Cumulative char counts for boundary words (minimum 1 to avoid zero-div)
  const wCum = [0];
  for (const w of boundaryWords) {
    const len = Math.max(w.word.trim().length, 1);
    wCum.push(wCum[wCum.length - 1] + len);
  }
  const wTotal = wCum[M];

  const result: WordTimestamp[] = [];

  for (let i = 0; i < N; i++) {
    const fStart = srcCum[i] / srcTotal;
    const fEnd   = srcCum[i + 1] / srcTotal;

    // Map source char fractions → boundary-word char space
    const wFStart = fStart * wTotal;
    const wFEnd   = fEnd   * wTotal;

    // Boundary word index whose range contains wFStart
    let wsi = 0;
    while (wsi < M - 1 && wCum[wsi + 1] <= wFStart) wsi++;

    // Boundary word index whose range contains wFEnd
    let wei = M - 1;
    while (wei > 0 && wCum[wei] >= wFEnd) wei--;

    // Interpolate start time within boundary word wsi
    const wsRange = wCum[wsi + 1] - wCum[wsi];
    const wsRel   = wsRange > 0 ? Math.max(0, (wFStart - wCum[wsi]) / wsRange) : 0;
    const actualStart =
      boundaryWords[wsi].start + wsRel * (boundaryWords[wsi].end - boundaryWords[wsi].start);

    // Interpolate end time within boundary word wei
    const weRange = wCum[wei + 1] - wCum[wei];
    const weRel   = weRange > 0 ? Math.min(1, (wFEnd - wCum[wei]) / weRange) : 1;
    const actualEnd =
      boundaryWords[wei].start + weRel * (boundaryWords[wei].end - boundaryWords[wei].start);

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
 * Used only if Azure returns no word-boundary events.
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
