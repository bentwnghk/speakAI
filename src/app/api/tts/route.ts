import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { generateTtsAudio, VOICE_MAP, estimateTtsCost, normalizeHeadingPunctuation } from "@/lib/tts";
import { db } from "@/lib/db";
import { generations } from "@/lib/db/schema";
import { eq, desc, count } from "drizzle-orm";
import { deductCredits, getUserBalance } from "@/lib/db/credits";

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
    };
    const { text, voice, speed, title } = body;

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
    const estimatedCost = estimateTtsCost(processedText);

    if (balance < estimatedCost) {
      return NextResponse.json(
        {
          error: "Insufficient credits",
          creditsNeeded: estimatedCost,
          currentBalance: balance,
        },
        { status: 402 }
      );
    }

    const result = await generateTtsAudio(processedText, voice, speed);

    const ttsCost = parseFloat(result.cost);
    const totalCost = ttsCost.toFixed(2);
    const totalCostNum = parseFloat(totalCost);

    const deduction = await deductCredits(
      userId,
      totalCostNum,
      `TTS generation: "${text.trim().slice(0, 50)}"`
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
          | "fable"
          | "echo"
          | "shimmer"
          | "onyx",
        speed,
        audioPath: result.audioPath,
        segments: segmentsJson,
        ttsCost: totalCost,
      })
      .returning();

    return NextResponse.json({
      id: generation.id,
      title: generation.title,
      transcript: generation.transcript,
      voice,
      speed,
      audioUrl: `/api/audio/${generation.id}`,
      segments: result.segments.length > 0 ? result.segments : undefined,
      ttsCost: totalCost,
      creditsUsed: totalCostNum,
      remainingCredits: deduction.balance,
      createdAt: generation.createdAt,
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

  const whereClause = eq(generations.userId, session.user.id);

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
    })),
    total,
    page,
    limit,
    totalPages: Math.ceil(total / limit),
  });
}
