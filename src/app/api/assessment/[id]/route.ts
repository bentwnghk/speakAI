import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { isAdminEmail } from "@/lib/admin";
import { db } from "@/lib/db";
import { assessments } from "@/lib/db/schema";
import { eq, and, or, isNull, gt } from "drizzle-orm";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
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

  const [row] = await db
    .select()
    .from(assessments)
    .where(and(...conditions));

  if (!row) {
    return NextResponse.json(
      { error: "Assessment not found" },
      { status: 404 }
    );
  }

  return NextResponse.json(row);
}

export async function DELETE(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id } = await params;

  const result = await db
    .delete(assessments)
    .where(
      and(eq(assessments.id, id), eq(assessments.userId, session.user.id))
    )
    .returning({ id: assessments.id });

  if (result.length === 0) {
    return NextResponse.json(
      { error: "Assessment not found" },
      { status: 404 }
    );
  }

  return NextResponse.json({ success: true });
}
