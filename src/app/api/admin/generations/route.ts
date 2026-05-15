import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { isAdminEmail } from "@/lib/admin";
import { db } from "@/lib/db";
import { generations, users } from "@/lib/db/schema";
import { desc, asc, sql, ilike, or } from "drizzle-orm";

export async function GET(req: NextRequest) {
  try {
    const session = await auth();
    if (!session?.user?.id || !isAdminEmail(session.user.email)) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }

    const { searchParams } = new URL(req.url);
    const sortBy = searchParams.get("sortBy") || "createdAt";
    const sortOrder = searchParams.get("sortOrder") || "desc";
    const search = searchParams.get("q") || "";

    const query = db
      .select({
        id: generations.id,
        userName: users.name,
        email: users.email,
        title: generations.title,
        voice: generations.voice,
        createdAt: generations.createdAt,
        ttsCost: generations.ttsCost,
      })
      .from(generations)
      .innerJoin(users, sql`${generations.userId} = ${users.id}`);

    const conditions = search.trim()
      ? [
          ilike(users.name, `%${search.trim()}%`),
          ilike(users.email, `%${search.trim()}%`),
          ilike(generations.title, `%${search.trim()}%`),
          ilike(generations.voice, `%${search.trim()}%`),
        ]
      : [];

    const orderColumn =
      sortBy === "userName"
        ? users.name
        : sortBy === "title"
          ? generations.title
          : sortBy === "voice"
            ? generations.voice
            : generations.createdAt;

    const orderFn = sortOrder === "asc" ? asc : desc;

    const rows = conditions.length
      ? await query
          .where(or(...conditions))
          .orderBy(orderFn(orderColumn))
      : await query.orderBy(orderFn(orderColumn));

    return NextResponse.json({ generations: rows });
  } catch (error) {
    console.error("Admin generations GET error:", error);
    return NextResponse.json(
      { error: "Failed to fetch generations." },
      { status: 500 }
    );
  }
}
