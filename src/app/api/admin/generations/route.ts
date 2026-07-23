import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { isAdminEmail } from "@/lib/admin";
import { db } from "@/lib/db";
import { generations, users } from "@/lib/db/schema";
import { desc, asc, sql, ilike, or, count } from "drizzle-orm";

const cumulativeSql = sql<string>`coalesce(sum(${generations.ttsCost}::numeric) over (partition by ${generations.userId} order by ${generations.createdAt} asc rows between unbounded preceding and current row), 0)`.as("cumulative_cost");

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
    const page = Math.max(1, parseInt(searchParams.get("page") || "1", 10));
    const perPage = Math.min(100, Math.max(1, parseInt(searchParams.get("perPage") || "20", 10)));

    const baseQuery = db
      .select({
        id: generations.id,
        userName: users.name,
        email: users.email,
        title: generations.title,
        voice: generations.voice,
        createdAt: generations.createdAt,
        ttsCost: generations.ttsCost,
        cumulativeCost: cumulativeSql,
      })
      .from(generations)
      .innerJoin(users, sql`${generations.userId} = ${users.id}`);

    const where = search.trim()
      ? or(
          ilike(users.name, `%${search.trim()}%`),
          ilike(users.email, `%${search.trim()}%`),
          ilike(generations.title, `%${search.trim()}%`),
          sql`${generations.voice}::text ilike ${`%${search.trim()}%`}`,
        )
      : undefined;

    const orderColumn =
      sortBy === "userName"
        ? users.name
        : sortBy === "title"
          ? generations.title
          : sortBy === "voice"
            ? generations.voice
            : generations.createdAt;

    const orderFn = sortOrder === "asc" ? asc : desc;

    const [rows, [{ total }]] = await Promise.all([
      where
        ? baseQuery.where(where).orderBy(orderFn(orderColumn)).limit(perPage).offset((page - 1) * perPage)
        : baseQuery.orderBy(orderFn(orderColumn)).limit(perPage).offset((page - 1) * perPage),
      db
        .select({ total: count() })
        .from(generations)
        .innerJoin(users, sql`${generations.userId} = ${users.id}`)
        .where(where),
    ]);

    return NextResponse.json({ generations: rows, total, page, perPage });
  } catch (error) {
    console.error("Admin generations GET error:", error);
    return NextResponse.json(
      { error: "Failed to fetch generations." },
      { status: 500 }
    );
  }
}
