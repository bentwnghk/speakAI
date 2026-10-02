import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { getUserTransactions } from "@/lib/db/credits";

export async function GET() {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    const transactions = await getUserTransactions(session.user.id);
    return NextResponse.json({ transactions });
  } catch (error) {
    console.error("Get transactions error:", error);
    return NextResponse.json(
      { error: "Failed to get transactions" },
      { status: 500 }
    );
  }
}
