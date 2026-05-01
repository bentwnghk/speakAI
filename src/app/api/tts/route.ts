import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { generateTtsAudio, VOICE_MAP } from "@/lib/tts";
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
    };
    const { text, voice, speed } = body;

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

    const now = new Date();
    const title = `Audio - ${now.toISOString().slice(0, 16).replace("T", " ")}`;

    const [generation] = await db
      .insert(generations)
      .values({
        userId: session.user.id,
        title,
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
      ttsCost: g.ttsCost,
      createdAt: g.createdAt,
    }))
  );
}
