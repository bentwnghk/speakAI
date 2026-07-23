import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { getUserPurchases } from "@/lib/db/credits";

export async function GET() {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    const purchases = await getUserPurchases(session.user.id);
    return NextResponse.json({ purchases });
  } catch (error) {
    console.error("Get purchases error:", error);
    return NextResponse.json(
      { error: "Failed to get purchases" },
      { status: 500 }
    );
  }
}
