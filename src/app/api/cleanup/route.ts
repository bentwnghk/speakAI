import { NextResponse } from "next/server";
import { db } from "@/lib/db";
import { generations, assessments } from "@/lib/db/schema";
import { lte, eq, and, isNotNull } from "drizzle-orm";
import { unlink } from "fs/promises";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const secret = process.env.CLEANUP_SECRET;
  if (secret) {
    const authHeader = request.headers.get("authorization");
    if (authHeader !== `Bearer ${secret}`) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
  }

  const now = new Date();
  let deletedGenerations = 0;
  let deletedAssessments = 0;

  const expired = await db
    .select()
    .from(generations)
    .where(lte(generations.expiresAt, now));

  for (const gen of expired) {
    try {
      await unlink(gen.audioPath).catch(() => {});
      await db.delete(generations).where(eq(generations.id, gen.id));
      deletedGenerations++;
    } catch {
      // skip individual errors
    }
  }

  const expiredAssessments = await db
    .select()
    .from(assessments)
    .where(and(lte(assessments.expiresAt, now), isNotNull(assessments.expiresAt)));

  for (const assessment of expiredAssessments) {
    try {
      if (assessment.audioPath) {
        await unlink(assessment.audioPath).catch(() => {});
      }
      await db.delete(assessments).where(eq(assessments.id, assessment.id));
      deletedAssessments++;
    } catch {
      // skip individual errors
    }
  }

  return NextResponse.json({
    deletedGenerations,
    deletedAssessments,
  });
}
