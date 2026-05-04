import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { db } from "@/lib/db";
import { generations } from "@/lib/db/schema";
import { eq, and, gt, or, isNull } from "drizzle-orm";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id } = await params;

  const [generation] = await db
    .select()
    .from(generations)
    .where(and(
      eq(generations.id, id),
      eq(generations.userId, session.user.id),
      or(isNull(generations.expiresAt), gt(generations.expiresAt, new Date()))
    ));

  if (!generation) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  return NextResponse.json({
    id: generation.id,
    title: generation.title,
    transcript: generation.transcript,
    voice: generation.voice,
    speed: generation.speed,
    audioUrl: `/api/audio/${generation.id}`,
    segments: generation.segments ? (JSON.parse(generation.segments) as unknown[]) : undefined,
    ttsCost: generation.ttsCost,
    createdAt: generation.createdAt,
    expiresAt: generation.expiresAt,
  });
}

export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id } = await params;
  const body = (await request.json()) as { title: string };
  const { title } = body;

  if (!title?.trim()) {
    return NextResponse.json({ error: "Title is required" }, { status: 400 });
  }

  const [updated] = await db
    .update(generations)
    .set({ title: title.trim() })
    .where(and(eq(generations.id, id), eq(generations.userId, session.user.id)))
    .returning();

  if (!updated) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  return NextResponse.json({ id: updated.id, title: updated.title });
}

export async function DELETE(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id } = await params;

  const [deleted] = await db
    .delete(generations)
    .where(and(eq(generations.id, id), eq(generations.userId, session.user.id)))
    .returning();

  if (!deleted) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  const { unlink } = await import("fs/promises");
  await unlink(deleted.audioPath).catch(() => {});

  return NextResponse.json({ success: true });
}
