import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { generateTtsAudio, VOICE_MAP, estimateTtsCost, normalizeHeadingPunctuation } from "@/lib/tts";
import { db } from "@/lib/db";
import { generations } from "@/lib/db/schema";
import { eq, desc, count, gt, and, or, isNull } from "drizzle-orm";
import { deductCredits, getUserBalance } from "@/lib/db/credits";

function getExpiresAt(): Date {
  const days = parseInt(process.env.AUDIO_RETENTION_DAYS || "365", 10);
  return new Date(Date.now() + days * 24 * 60 * 60 * 1000);
}

export async function POST(request: NextRequest) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const userId = session.user.id;

  try {
    const body = (await request.json()) as {
      text: string;
      voice: string;
      speed: number;
      title?: string;
      visionCost?: number;
    };
    const { text, voice, speed, title, visionCost = 0 } = body;

    if (!text?.trim()) {
      return NextResponse.json(
        { error: "No text provided" },
        { status: 400 }
      );
    }

    if (!VOICE_MAP[voice]) {
      return NextResponse.json(
        { error: "Invalid voice selection" },
        { status: 400 }
      );
    }

    const processedText = normalizeHeadingPunctuation(text);

    const balance = await getUserBalance(userId);
    const estimatedTtsCost = estimateTtsCost(processedText);
    const estimatedTotalCost = estimatedTtsCost + visionCost;

    if (balance < estimatedTotalCost) {
      return NextResponse.json(
        {
          error: "Insufficient credits",
          creditsNeeded: estimatedTotalCost,
          currentBalance: balance,
        },
        { status: 402 }
      );
    }

    const result = await generateTtsAudio(processedText, voice, speed);

    const ttsCost = parseFloat(result.cost);
    const totalCostNum = Math.round((ttsCost + visionCost) * 100) / 100;
    const totalCost = totalCostNum.toFixed(2);

    const deduction = await deductCredits(
      userId,
      totalCostNum,
      `TTS generation${visionCost > 0 ? ` (+HK$${visionCost.toFixed(2)} vision)` : ""}: "${text.trim().slice(0, 50)}"`
    );

    if (!deduction.success) {
      return NextResponse.json(
        {
          error: deduction.error,
          creditsNeeded: totalCostNum,
          currentBalance: deduction.balance,
        },
        { status: 402 }
      );
    }

    const segmentsJson = result.segments.length > 0 ? JSON.stringify(result.segments) : null;

    const generationTitle =
      title?.trim() ||
      `${text.trim().slice(0, 30).replace(/\n/g, " ")}${text.trim().length > 30 ? "..." : ""}`;

    const [generation] = await db
      .insert(generations)
      .values({
        userId,
        title: generationTitle,
        transcript: text,
        voice: (VOICE_MAP[voice] || "nova") as
          | "nova"
          | "alloy"
          | "phoebe"
          | "adam"
          | "ava"
          | "ollie",
        speed,
        audioPath: result.audioPath,
        segments: segmentsJson,
        ttsCost: totalCost,
        expiresAt: getExpiresAt(),
      })
      .returning();

    return NextResponse.json({
      id: generation.id,
      title: generation.title,
      transcript: generation.transcript,
      voice,
      speed,
      audioUrl: `/api/audio/${generation.id}`,
      audioPath: generation.audioPath,
      segments: result.segments.length > 0 ? result.segments : undefined,
      ttsCost: totalCost,
      creditsUsed: totalCostNum,
      remainingCredits: deduction.balance,
      createdAt: generation.createdAt,
      expiresAt: generation.expiresAt,
    });
  } catch (error) {
    console.error("TTS generation error:", error);
    const message =
      error instanceof Error ? error.message : "Audio generation failed";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

export async function GET(request: NextRequest) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { searchParams } = new URL(request.url);
  const page = Math.max(1, Number(searchParams.get("page")) || 1);
  const limit = Math.min(
    100,
    Math.max(1, Number(searchParams.get("limit")) || 10)
  );
  const offset = (page - 1) * limit;

  const whereClause = and(
    eq(generations.userId, session.user.id),
    or(
      isNull(generations.expiresAt),
      gt(generations.expiresAt, new Date())
    )
  );

  const [userGenerations, totalResult] = await Promise.all([
    db
      .select()
      .from(generations)
      .where(whereClause)
      .orderBy(desc(generations.createdAt))
      .limit(limit)
      .offset(offset),
    db
      .select({ count: count() })
      .from(generations)
      .where(whereClause),
  ]);

  const total = totalResult[0]?.count ?? 0;

  return NextResponse.json({
    items: userGenerations.map((g) => ({
      id: g.id,
      title: g.title,
      transcript: g.transcript,
      voice: g.voice,
      speed: g.speed,
      audioUrl: `/api/audio/${g.id}`,
      segments: g.segments
        ? (JSON.parse(g.segments) as unknown[])
        : undefined,
      ttsCost: g.ttsCost,
      createdAt: g.createdAt,
      expiresAt: g.expiresAt,
    })),
    total,
    page,
    limit,
    totalPages: Math.ceil(total / limit),
  });
}
