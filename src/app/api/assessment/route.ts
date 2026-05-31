import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { db } from "@/lib/db";
import { assessments } from "@/lib/db/schema";
import { deductCredits, refundCredits } from "@/lib/db/credits";
import { eq, desc, sql } from "drizzle-orm";
import { z } from "zod";

const ASSESSMENT_COST_HKD = parseFloat(
  process.env.ASSESSMENT_COST_HKD || "0.50"
);

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
  syllables: z.array(z.array(z.any())).nullable().optional(),
});

export async function POST(request: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const userId = session.user.id;

  try {
    const body: unknown = await request.json();
    const data = saveSchema.parse(body);

    const deductResult = await deductCredits(
      userId,
      ASSESSMENT_COST_HKD,
      `Speaking assessment: "${data.referenceText.slice(0, 50)}${data.referenceText.length > 50 ? "..." : ""}"`
    );

    if (!deductResult.success) {
      return NextResponse.json(
        { error: "Insufficient credits", balance: deductResult.balance },
        { status: 402 }
      );
    }

    try {
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
          syllables: data.syllables ?? null,
          cost: ASSESSMENT_COST_HKD,
        })
        .returning({ id: assessments.id });

      return NextResponse.json({
        id: inserted.id,
        cost: ASSESSMENT_COST_HKD,
        balance: deductResult.balance,
      });
    } catch (dbError) {
      await refundCredits(userId, ASSESSMENT_COST_HKD, "Assessment save failed - refund");
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

  const [countResult] = await db
    .select({ count: sql<number>`count(*)::int` })
    .from(assessments)
    .where(eq(assessments.userId, session.user.id));

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
      createdAt: assessments.createdAt,
    })
    .from(assessments)
    .where(eq(assessments.userId, session.user.id))
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
