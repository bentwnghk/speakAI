import { readFile } from "fs/promises";
import { extname } from "path";

const VISION_INPUT_PRICE_PER_1M = Number(
  process.env.VISION_PRICE_INPUT_PER_1M_TOKENS ?? "0.40",
);
const VISION_OUTPUT_PRICE_PER_1M = Number(
  process.env.VISION_PRICE_OUTPUT_PER_1M_TOKENS ?? "1.60",
);
const USD_TO_HKD = 7.8;

export function estimateVisionCostHkd(
  promptTokens: number,
  completionTokens: number,
): number {
  const inputUsd = (promptTokens / 1_000_000) * VISION_INPUT_PRICE_PER_1M;
  const outputUsd = (completionTokens / 1_000_000) * VISION_OUTPUT_PRICE_PER_1M;
  return Math.max((inputUsd + outputUsd) * USD_TO_HKD, 0.01);
}

export interface ExtractionResult {
  text: string;
  visionCostHkd: number;
}

async function extractFromDocx(filePath: string): Promise<ExtractionResult> {
  const mammoth = await import("mammoth");
  const buffer = await readFile(filePath);
  const result = await mammoth.extractRawText({ buffer });
  return { text: result.value, visionCostHkd: 0 };
}

async function extractFromTxt(filePath: string): Promise<ExtractionResult> {
  const text = await readFile(filePath, "utf-8");
  return { text, visionCostHkd: 0 };
}

async function extractFromImage(
  filePath: string,
  buffer: Buffer
): Promise<ExtractionResult> {
  const { generateText } = await import("ai");
  const { createOpenAI } = await import("@ai-sdk/openai");

  const visionApiKey = process.env.VISION_API_KEY || process.env.TTS_API_KEY;
  const visionBaseUrl =
    process.env.VISION_BASE_URL || process.env.TTS_BASE_URL;

  if (!visionApiKey) {
    throw new Error("VISION_API_KEY is not configured");
  }

  const openai = createOpenAI({
    apiKey: visionApiKey,
    baseURL: visionBaseUrl?.replace(/\/$/, ""),
  });

  const ext = extname(filePath).toLowerCase();
  const mimeType =
    ext === ".png"
      ? "image/png"
      : ext === ".jpg" || ext === ".jpeg"
        ? "image/jpeg"
        : "image/png";

  const { text, usage } = await generateText({
    model: openai(process.env.VISION_MODEL || "gpt-4.1-mini"),
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image",
            image: `data:${mimeType};base64,${buffer.toString("base64")}`,
          },
          {
            type: "text",
            text: `Extract all computer-readable text from the provided image.

Instructions:
- Preserve original paragraph breaks by inserting two newline characters between paragraphs.
- Remove any numbers in square brackets or parentheses that appear at the beginning of any paragraph.
- Do not insert any additional line breaks within paragraphs.
- Use visual spacing and indentation to detect paragraph breaks.
- Return only the extracted text, without commentary, metadata, or formatting.
- Output the result as a plain text string.`,
          },
        ],
      },
    ],
    maxTokens: 32768,
    temperature: 0,
  });

  return {
    text,
    visionCostHkd: estimateVisionCostHkd(usage.promptTokens, usage.completionTokens),
  };
}

export async function extractTextFromFile(
  filePath: string
): Promise<ExtractionResult> {
  const ext = extname(filePath).toLowerCase();
  const supportedImageExts = [".jpg", ".jpeg", ".png"];

  if (ext === ".pdf") {
    throw new Error("PDFs should be processed client-side via pdfjs-dist");
  } else if (ext === ".docx") {
    return extractFromDocx(filePath);
  } else if (ext === ".txt") {
    return extractFromTxt(filePath);
  } else if (supportedImageExts.includes(ext)) {
    const buffer = await readFile(filePath);
    return extractFromImage(filePath, buffer);
  }

  throw new Error(`Unsupported file type: ${ext}`);
}

import { SUPPORTED_EXTENSIONS } from "./constants";

export { SUPPORTED_EXTENSIONS };
