import { NextResponse } from "next/server";
import { getWelcomeCredits } from "@/lib/db/credits";
import { estimateVisionCostHkd } from "@/lib/file-parser";

const AZURE_TTS_USD_PER_1M_CHARS =
  Number(process.env.AZURE_SPEECH_PRICE_USD_PER_1M_CHARS) || 16;

const HKD_PER_USD = 7.8;
const AVG_CHARS_PER_GENERATION = 3000;
const AVG_VISION_PROMPT_TOKENS = 1000;
const AVG_VISION_COMPLETION_TOKENS = 500;
const IMAGE_UPLOAD_RATIO = 0.3;

export function GET() {
  const welcomeCredits = getWelcomeCredits();

  const ttsCostPerGeneration =
    (AVG_CHARS_PER_GENERATION / 1_000_000) *
    AZURE_TTS_USD_PER_1M_CHARS *
    HKD_PER_USD;

  const visionCostPerGeneration = estimateVisionCostHkd(
    AVG_VISION_PROMPT_TOKENS,
    AVG_VISION_COMPLETION_TOKENS,
  );

  const avgCostPerGeneration =
    ttsCostPerGeneration * (1 - IMAGE_UPLOAD_RATIO) +
    (ttsCostPerGeneration + visionCostPerGeneration) * IMAGE_UPLOAD_RATIO;

  const approxGenerations = Math.floor(
    avgCostPerGeneration > 0 ? welcomeCredits / avgCostPerGeneration : 0,
  );

  return NextResponse.json({
    welcomeCredits,
    approxGenerations,
  });
}
