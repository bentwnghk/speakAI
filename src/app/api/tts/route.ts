import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { generateTtsAudio, VOICE_MAP, alignAudio } from "@/lib/tts";
import { db } from "@/lib/db";
import { generations } from "@/lib/db/schema";
import { eq, desc, count } from "drizzle-orm";

export async function POST(request: NextRequest) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

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

    const result = await generateTtsAudio(text, voice, speed);

    const { segments, audioDurationSeconds } = await alignAudio(result.audioPath, text);

    const ttsCost = parseFloat(result.cost);
    const whisperCost = (audioDurationSeconds / 60) * 0.006 * 7.8;
    const totalCost = (ttsCost + whisperCost).toFixed(2);

    const segmentsJson = segments.length > 0 ? JSON.stringify(segments) : null;

    const generationTitle =
      title?.trim() ||
      `${text.trim().slice(0, 30).replace(/\n/g, " ")}${text.trim().length > 30 ? "..." : ""}`;

    const [generation] = await db
      .insert(generations)
      .values({
        userId: session.user.id,
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
      segments: segments.length > 0 ? segments : undefined,
      ttsCost: totalCost,
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
  const limit = Math.min(100, Math.max(1, Number(searchParams.get("limit")) || 10));
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
      segments: g.segments ? (JSON.parse(g.segments) as unknown[]) : undefined,
      ttsCost: g.ttsCost,
      createdAt: g.createdAt,
    })),
    total,
    page,
    limit,
    totalPages: Math.ceil(total / limit),
  });
}
