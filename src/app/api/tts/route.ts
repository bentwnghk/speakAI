import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { generateTtsAudio, VOICE_MAP } from "@/lib/tts";
import { db } from "@/lib/db";
import { generations } from "@/lib/db/schema";
import { eq, desc } from "drizzle-orm";
import { writeFile } from "fs/promises";
import { join } from "path";
import { extractTextFromFile, SUPPORTED_EXTENSIONS } from "@/lib/file-parser";
import { extname } from "path";

export async function POST(request: NextRequest) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const formData = await request.formData();
    const inputMethod = formData.get("inputMethod") as string;
    const voice = formData.get("voice") as string;
    const speedStr = formData.get("speed") as string;
    const text = formData.get("text") as string;
    const files = formData.getAll("files") as File[];

    const speed = parseInt(speedStr || "100", 10);

    if (!VOICE_MAP[voice]) {
      return NextResponse.json(
        { error: "Invalid voice selection" },
        { status: 400 }
      );
    }

    let fullText = "";

    if (inputMethod === "upload" && files.length > 0) {
      const texts: string[] = [];
      const fileNames: string[] = [];

      for (const file of files) {
        const ext = extname(file.name).toLowerCase();
        if (!SUPPORTED_EXTENSIONS.includes(ext)) {
          return NextResponse.json(
            { error: `Unsupported file type: ${ext}` },
            { status: 400 }
          );
        }

        const bytes = await file.arrayBuffer();
        const buffer = Buffer.from(bytes);

        const tmpDir = await import("os").then(m => m.tmpdir());
        const tmpPath = join(tmpDir, `speakai-${Date.now()}-${file.name}`);
        await writeFile(tmpPath, buffer);

        try {
          const extracted = await extractTextFromFile(tmpPath);
          texts.push(extracted);
          fileNames.push(file.name.replace(ext, ""));
        } finally {
          const { unlink } = await import("fs/promises");
          await unlink(tmpPath).catch(() => {});
        }
      }

      fullText = texts.filter(Boolean).join("\n\n");
    } else if (inputMethod === "text" && text?.trim()) {
      fullText = text;
    } else {
      return NextResponse.json(
        { error: "No text or files provided" },
        { status: 400 }
      );
    }

    if (!fullText.trim()) {
      return NextResponse.json(
        { error: "No text content to process" },
        { status: 400 }
      );
    }

    const result = await generateTtsAudio(fullText, voice, speed);

    const now = new Date();
    const title = `Audio - ${now.toISOString().slice(0, 16).replace("T", " ")}`;

    const [generation] = await db
      .insert(generations)
      .values({
        userId: session.user.id,
        title,
        transcript: fullText,
        voice: (VOICE_MAP[voice] || "nova") as "nova" | "alloy" | "fable" | "echo" | "shimmer" | "onyx",
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
