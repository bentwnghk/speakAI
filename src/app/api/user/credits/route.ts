import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { getUserBalance } from "@/lib/db/credits";

export async function GET() {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    const balance = await getUserBalance(session.user.id);
    return NextResponse.json({ balance });
  } catch (error) {
    console.error("Get credits error:", error);
    return NextResponse.json(
      { error: "Failed to get credits" },
      { status: 500 }
    );
  }
}
