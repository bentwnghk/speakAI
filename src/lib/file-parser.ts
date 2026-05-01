import { readFile } from "fs/promises";
import { extname } from "path";

async function extractFromPdf(filePath: string): Promise<string> {
  const pdfParse = (await import("pdf-parse")).default;
  const buffer = await readFile(filePath);
  const data = await pdfParse(buffer);
  return data.text;
}

async function extractFromDocx(filePath: string): Promise<string> {
  const mammoth = await import("mammoth");
  const buffer = await readFile(filePath);
  const result = await mammoth.extractRawText({ buffer });
  return result.value;
}

async function extractFromTxt(filePath: string): Promise<string> {
  return readFile(filePath, "utf-8");
}

async function extractFromImage(
  filePath: string,
  buffer: Buffer
): Promise<string> {
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

  const { text } = await generateText({
    model: openai("gpt-4.1-mini"),
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

  return text;
}

export async function extractTextFromFile(
  filePath: string
): Promise<string> {
  const ext = extname(filePath).toLowerCase();
  const supportedImageExts = [".jpg", ".jpeg", ".png"];

  if (ext === ".pdf") {
    return extractFromPdf(filePath);
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
