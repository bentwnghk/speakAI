import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { generateTtsAudio, VOICE_MAP, alignAudio } from "@/lib/tts";
import { db } from "@/lib/db";
import { generations } from "@/lib/db/schema";
import { eq, desc } from "drizzle-orm";

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

    const segments = await alignAudio(result.audioPath, text);
    const segmentsJson = segments.length > 0 ? JSON.stringify(segments) : null;

    const generationTitle =
      title?.trim() ||
      `Audio - ${text.trim().slice(0, 30).replace(/\n/g, " ")}${text.trim().length > 30 ? "..." : ""}`;

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
        ttsCost: result.cost,
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
      ttsCost: result.cost,
      createdAt: generation.createdAt,
    });
  } catch (error) {
    console.error("TTS generation error:", error);
    const message =
      error instanceof Error ? error.message : "Audio generation failed";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

export async function GET() {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const userGenerations = await db
    .select()
    .from(generations)
    .where(eq(generations.userId, session.user.id))
    .orderBy(desc(generations.createdAt));

  return NextResponse.json(
    userGenerations.map((g) => ({
      id: g.id,
      title: g.title,
      transcript: g.transcript,
      voice: g.voice,
      speed: g.speed,
      audioUrl: `/api/audio/${g.id}`,
      segments: g.segments ? (JSON.parse(g.segments) as unknown[]) : undefined,
      ttsCost: g.ttsCost,
      createdAt: g.createdAt,
    }))
  );
}
