import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { isAdminEmail } from "@/lib/admin";
import { db } from "@/lib/db";
import { assessments } from "@/lib/db/schema";
import { eq, and, or, isNull, gt, sql } from "drizzle-orm";
import { deductCredits, refundCredits } from "@/lib/db/credits";
import { summarizeAssessment, generateFeedback } from "@/lib/feedback";
import type { WordResult } from "@/types/assessment";

export async function POST(
  request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id } = await params;
  const isAdmin = isAdminEmail(session.user.email);

  const conditions = [
    eq(assessments.id, id),
    ...(isAdmin ? [] : [eq(assessments.userId, session.user.id)]),
    ...(isAdmin ? [] : [or(isNull(assessments.expiresAt), gt(assessments.expiresAt, new Date()))]),
  ];

  const [row] = await db.select().from(assessments).where(and(...conditions));

  if (!row) {
    return NextResponse.json({ error: "Assessment not found" }, { status: 404 });
  }

  if (row.feedback) {
    return NextResponse.json({ feedback: row.feedback, cost: 0, cached: true });
  }

  let locale = "en";
  try {
    const body = (await request.json()) as { locale?: string };
    if (body?.locale === "zh-TW") locale = "zh-TW";
  } catch {
    /* no body */
  }

  const words = (row.words ?? []) as WordResult[];
  const summary = summarizeAssessment(
    words,
    {
      accuracy: row.accuracyScore,
      fluency: row.fluencyScore,
      completeness: row.completenessScore,
      prosody: row.prosodyScore,
      pron: row.pronScore,
    },
    row.referenceText,
    row.recognizedText,
    row.durationMs,
  );

  let feedbackCost = 0;
  let feedback;

  try {
    const result = await generateFeedback(summary, locale);
    feedback = result.feedback;
    feedbackCost = result.cost;
  } catch (error) {
    console.error("Feedback generation error:", error);
    const message =
      error instanceof Error ? error.message : "Failed to generate feedback";
    return NextResponse.json({ error: message }, { status: 502 });
  }

  const userId = session.user.id;
  const deduct = await deductCredits(
    userId,
    feedbackCost,
    `AI feedback for assessment: "${row.referenceText.slice(0, 50)}${row.referenceText.length > 50 ? "..." : ""}"`,
  );

  if (!deduct.success) {
    return NextResponse.json(
      { error: "Insufficient credits", balance: deduct.balance },
      { status: 402 },
    );
  }

  try {
    await db
      .update(assessments)
      .set({
        feedback,
        cost: sql`${assessments.cost} + ${feedbackCost}`,
      })
      .where(eq(assessments.id, id));

    return NextResponse.json({
      feedback,
      cost: feedbackCost,
      balance: deduct.balance,
      cached: false,
    });
  } catch (dbError) {
    await refundCredits(userId, feedbackCost, "Feedback save failed - refund");
    throw dbError;
  }
}
