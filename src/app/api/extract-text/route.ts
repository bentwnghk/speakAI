import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { extractTextFromFile, SUPPORTED_EXTENSIONS } from "@/lib/file-parser";
import { writeFile, mkdir, unlink } from "fs/promises";
import { join } from "path";
import { extname } from "path";

export async function POST(request: NextRequest) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const formData = await request.formData();
    const file = formData.get("file") as File;

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 });
    }

    const ext = extname(file.name).toLowerCase();
    if (!SUPPORTED_EXTENSIONS.includes(ext)) {
      return NextResponse.json(
        { error: `Unsupported file type: ${ext}` },
        { status: 400 }
      );
    }

    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    const tmpDir = await import("os").then(m => m.tmpdir());
    const tmpPath = join(tmpDir, `speakai-${Date.now()}-${file.name}`);
    await writeFile(tmpPath, buffer);

    try {
      const text = await extractTextFromFile(tmpPath);
      return NextResponse.json({ text, fileName: file.name });
    } finally {
      await unlink(tmpPath).catch(() => {});
    }
  } catch (error) {
    console.error("Text extraction error:", error);
    const message =
      error instanceof Error ? error.message : "Text extraction failed";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
