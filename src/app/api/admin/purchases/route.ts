import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { isAdminEmail } from "@/lib/admin";
import { db } from "@/lib/db";
import { purchases, users } from "@/lib/db/schema";
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
        : sortBy === "planName"
          ? purchases.planName
          : sortBy === "amountHKD"
            ? purchases.amountHKD
            : purchases.createdAt;

    const orderFn = sortOrder === "asc" ? asc : desc;

    const [rows, [{ total }]] = await Promise.all([
      db
        .select({
          id: purchases.id,
          userName: users.name,
          email: users.email,
          planName: purchases.planName,
          creditsAmount: purchases.creditsAmount,
          amountHKD: purchases.amountHKD,
          status: purchases.status,
          createdAt: purchases.createdAt,
        })
        .from(purchases)
        .innerJoin(users, sql`${purchases.userId} = ${users.id}`)
        .orderBy(orderFn(orderColumn))
        .limit(perPage)
        .offset((page - 1) * perPage),
      db
        .select({ total: count() })
        .from(purchases)
        .innerJoin(users, sql`${purchases.userId} = ${users.id}`),
    ]);

    return NextResponse.json({ purchases: rows, total, page, perPage });
  } catch (error) {
    console.error("Admin purchases GET error:", error);
    return NextResponse.json(
      { error: "Failed to fetch purchases." },
      { status: 500 }
    );
  }
}
