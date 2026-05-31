import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { isAdminEmail } from "@/lib/admin";
import { db } from "@/lib/db";
import { assessments, users } from "@/lib/db/schema";
import { desc, asc, sql, ilike, or, count } from "drizzle-orm";

const cumulativeSql = sql<string>`coalesce(sum(${assessments.cost}) over (partition by ${assessments.userId} order by ${assessments.createdAt} asc rows between unbounded preceding and current row), 0)`.as("cumulative_cost");

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
        id: assessments.id,
        userName: users.name,
        email: users.email,
        referenceText: assessments.referenceText,
        durationMs: assessments.durationMs,
        pronScore: assessments.pronScore,
        accuracyScore: assessments.accuracyScore,
        fluencyScore: assessments.fluencyScore,
        completenessScore: assessments.completenessScore,
        prosodyScore: assessments.prosodyScore,
        cost: assessments.cost,
        createdAt: assessments.createdAt,
        cumulativeCost: cumulativeSql,
      })
      .from(assessments)
      .innerJoin(users, sql`${assessments.userId} = ${users.id}`);

    const where = search.trim()
      ? or(
          ilike(users.name, `%${search.trim()}%`),
          ilike(users.email, `%${search.trim()}%`),
          ilike(assessments.referenceText, `%${search.trim()}%`),
        )
      : undefined;

    const orderColumn =
      sortBy === "userName"
        ? users.name
        : sortBy === "pronScore"
          ? assessments.pronScore
          : sortBy === "cost"
            ? assessments.cost
            : assessments.createdAt;

    const orderFn = sortOrder === "asc" ? asc : desc;

    const [rows, [{ total }]] = await Promise.all([
      where
        ? baseQuery.where(where).orderBy(orderFn(orderColumn)).limit(perPage).offset((page - 1) * perPage)
        : baseQuery.orderBy(orderFn(orderColumn)).limit(perPage).offset((page - 1) * perPage),
      db
        .select({ total: count() })
        .from(assessments)
        .innerJoin(users, sql`${assessments.userId} = ${users.id}`)
        .where(where),
    ]);

    return NextResponse.json({ assessments: rows, total, page, perPage });
  } catch (error) {
    console.error("Admin assessments GET error:", error);
    return NextResponse.json(
      { error: "Failed to fetch assessments." },
      { status: 500 }
    );
  }
}
