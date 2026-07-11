import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { db } from "@/lib/db";
import { assessments } from "@/lib/db/schema";
import { deductCredits, refundCredits } from "@/lib/db/credits";
import { analyzeStress } from "@/lib/stress";
import { eq, desc, sql, or, isNull, gt, and } from "drizzle-orm";
import { z } from "zod";
import { mkdir, writeFile } from "fs/promises";
import { join } from "path";
import { nanoid } from "nanoid";

const ASSESSMENT_PRICE_USD_PER_HOUR = parseFloat(
  process.env.ASSESSMENT_PRICE_USD_PER_HOUR || "1.00"
);
const USD_TO_HKD = 7.8;

function calculateCost(durationMs: number): number {
  const cost = (durationMs / 3_600_000) * ASSESSMENT_PRICE_USD_PER_HOUR * USD_TO_HKD;
  return Math.round(Math.max(cost, 0.01) * 100) / 100;
}

function getExpiresAt(): Date {
  const days = parseInt(process.env.RECORDING_RETENTION_DAYS || process.env.AUDIO_RETENTION_DAYS || "365", 10);
  return new Date(Date.now() + days * 24 * 60 * 60 * 1000);
}

const saveSchema = z.object({
  referenceText: z.string().min(1).max(5000),
  recognizedText: z.string().min(1),
  durationMs: z.number().int().min(0),
  accuracyScore: z.number().min(0).max(100),
  fluencyScore: z.number().min(0).max(100),
  completenessScore: z.number().min(0).max(100),
  prosodyScore: z.number().min(0).max(100).nullable(),
  pronScore: z.number().min(0).max(100),
  words: z.array(z.any()),
  phonemes: z.array(z.array(z.any())).nullable().optional(),
});

export async function POST(request: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const userId = session.user.id;

  try {
    const formData = await request.formData();
    const dataField = formData.get("data");
    if (!dataField || typeof dataField !== "string") {
      return NextResponse.json(
        { error: "Missing data field" },
        { status: 400 }
      );
    }

    const body: unknown = JSON.parse(dataField);
    const data = saveSchema.parse(body);

    const audioFile = formData.get("audio") as File | null;

      const cost = calculateCost(data.durationMs);

      const stress = analyzeStress(data.words);

      const deductResult = await deductCredits(
      userId,
      cost,
      `Speaking assessment: "${data.referenceText.slice(0, 50)}${data.referenceText.length > 50 ? "..." : ""}"`
    );

    if (!deductResult.success) {
      return NextResponse.json(
        { error: "Insufficient credits", balance: deductResult.balance },
        { status: 402 }
      );
    }

    let audioPath: string | null = null;

    try {
      let savedId = "";

      if (audioFile && audioFile.size > 0) {
        const audioDir = join(process.cwd(), "data", "recording");
        await mkdir(audioDir, { recursive: true });
        const audioMimeType = (formData.get("audioMimeType") as string) || audioFile.type || "audio/webm";
        const ext = audioMimeType.includes("mp4") ? "mp4" : "webm";
        const filename = `${nanoid()}.${ext}`;
        audioPath = join("data", "recording", filename);
        const buffer = Buffer.from(await audioFile.arrayBuffer());
        await writeFile(join(process.cwd(), audioPath), buffer);
      }

      const [inserted] = await db
        .insert(assessments)
        .values({
          userId,
          referenceText: data.referenceText,
          recognizedText: data.recognizedText,
          durationMs: data.durationMs,
          accuracyScore: data.accuracyScore,
          fluencyScore: data.fluencyScore,
          completenessScore: data.completenessScore,
          prosodyScore: data.prosodyScore,
          pronScore: data.pronScore,
          words: data.words,
          phonemes: data.phonemes ?? null,
          stress,
          audioPath,
          cost,
          expiresAt: audioPath ? getExpiresAt() : null,
        })
        .returning({ id: assessments.id });

      savedId = inserted.id;

      return NextResponse.json({
        id: savedId,
        cost,
        balance: deductResult.balance,
        stress,
      });
    } catch (dbError) {
      await refundCredits(userId, cost, "Assessment save failed - refund");
      throw dbError;
    }
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: "Invalid data", details: error.errors },
        { status: 400 }
      );
    }
    console.error("Assessment save error:", error);
    return NextResponse.json(
      { error: "Failed to save assessment" },
      { status: 500 }
    );
  }
}

export async function GET(request: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { searchParams } = new URL(request.url);
  const page = parseInt(searchParams.get("page") || "1");
  const limit = parseInt(searchParams.get("limit") || "10");
  const offset = (page - 1) * limit;

  const notExpired = or(
    isNull(assessments.expiresAt),
    gt(assessments.expiresAt, new Date())
  );

  const [countResult] = await db
    .select({ count: sql<number>`count(*)::int` })
    .from(assessments)
    .where(and(eq(assessments.userId, session.user.id), notExpired));

  const items = await db
    .select({
      id: assessments.id,
      referenceText: assessments.referenceText,
      recognizedText: assessments.recognizedText,
      durationMs: assessments.durationMs,
      accuracyScore: assessments.accuracyScore,
      fluencyScore: assessments.fluencyScore,
      completenessScore: assessments.completenessScore,
      prosodyScore: assessments.prosodyScore,
      pronScore: assessments.pronScore,
      words: assessments.words,
      cost: assessments.cost,
      hasAudio: sql<boolean>`${assessments.audioPath} IS NOT NULL`,
      expiresAt: assessments.expiresAt,
      createdAt: assessments.createdAt,
    })
    .from(assessments)
    .where(and(eq(assessments.userId, session.user.id), notExpired))
    .orderBy(desc(assessments.createdAt))
    .limit(limit)
    .offset(offset);

  return NextResponse.json({
    items,
    total: countResult.count,
    page,
    limit,
  });
}
