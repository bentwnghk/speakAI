import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { isAdminEmail } from "@/lib/admin";
import { db } from "@/lib/db";
import { signInLogs, users } from "@/lib/db/schema";
import { desc, asc, sql, count } from "drizzle-orm";

export async function GET(req: NextRequest) {
  try {
    const session = await auth();
    if (!session?.user?.id || !isAdminEmail(session.user.email)) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }

    const { searchParams } = new URL(req.url);
    const sortBy = searchParams.get("sortBy") || "createdAt";
    const sortOrder = searchParams.get("sortOrder") || "desc";
    const page = Math.max(1, parseInt(searchParams.get("page") || "1", 10));
    const perPage = Math.min(100, Math.max(1, parseInt(searchParams.get("perPage") || "20", 10)));

    const orderColumn =
      sortBy === "userName"
        ? users.name
        : signInLogs.createdAt;

    const orderFn = sortOrder === "asc" ? asc : desc;

    const [rows, [{ total }]] = await Promise.all([
      db
        .select({
          id: signInLogs.id,
          userName: users.name,
          email: users.email,
          provider: signInLogs.provider,
          createdAt: signInLogs.createdAt,
        })
        .from(signInLogs)
        .innerJoin(users, sql`${signInLogs.userId} = ${users.id}`)
        .orderBy(orderFn(orderColumn))
        .limit(perPage)
        .offset((page - 1) * perPage),
      db
        .select({ total: count() })
        .from(signInLogs)
        .innerJoin(users, sql`${signInLogs.userId} = ${users.id}`),
    ]);

    return NextResponse.json({ signIns: rows, total, page, perPage });
  } catch (error) {
    console.error("Admin sign-ins GET error:", error);
    return NextResponse.json(
      { error: "Failed to fetch sign-in logs." },
      { status: 500 }
    );
  }
}
