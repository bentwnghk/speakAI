import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { estimateTtsCost } from "@/lib/tts";
import { synthesizeWord } from "@/lib/word-tts";
import { deductCredits } from "@/lib/db/credits";

export async function POST(request: NextRequest) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const userId = session.user.id;

  try {
    const body = (await request.json()) as { word: string };
    const { word } = body;

    if (!word?.trim()) {
      return NextResponse.json({ error: "No word provided" }, { status: 400 });
    }

    const cleanWord = word.trim().replace(/[^a-zA-Z0-9''-]/g, "").slice(0, 100);
    if (!cleanWord) {
      return NextResponse.json({ error: "Invalid word" }, { status: 400 });
    }

    const estimatedCost = estimateTtsCost(cleanWord);
    const deduction = await deductCredits(
      userId,
      estimatedCost,
      `Word pronunciation: "${cleanWord}"`,
    );

    if (!deduction.success) {
      return NextResponse.json(
        {
          error: "Insufficient credits",
          creditsNeeded: estimatedCost,
          currentBalance: deduction.balance,
        },
        { status: 402 },
      );
    }

    const audioBuffer = await synthesizeWord(cleanWord);

    return new NextResponse(audioBuffer, {
      headers: {
        "Content-Type": "audio/mpeg",
        "Content-Length": String(audioBuffer.byteLength),
        "X-Credits-Used": estimatedCost.toFixed(4),
        "X-Remaining-Credits": String(deduction.balance),
      },
    });
  } catch (error) {
    console.error("Word TTS error:", error);
    const message =
      error instanceof Error ? error.message : "Word pronunciation failed";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
