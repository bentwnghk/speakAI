import { NextResponse } from "next/server";
import { db } from "@/lib/db";
import { generations } from "@/lib/db/schema";
import { lte, eq } from "drizzle-orm";
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
  const expired = await db
    .select()
    .from(generations)
    .where(lte(generations.expiresAt, now));

  if (expired.length === 0) {
    return NextResponse.json({ deleted: 0 });
  }

  let deletedCount = 0;
  for (const gen of expired) {
    try {
      await unlink(gen.audioPath).catch(() => {});
      await db.delete(generations).where(eq(generations.id, gen.id));
      deletedCount++;
    } catch {
      // skip individual errors
    }
  }

  return NextResponse.json({ deleted: deletedCount });
}
