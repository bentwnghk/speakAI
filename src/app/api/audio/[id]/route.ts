import { NextRequest } from "next/server";
import { auth } from "@/lib/auth";
import { db } from "@/lib/db";
import { generations } from "@/lib/db/schema";
import { eq, and } from "drizzle-orm";
import { readFile } from "fs/promises";
import { join } from "path";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const session = await auth();
  if (!session?.user?.id) {
    return new Response("Unauthorized", { status: 401 });
  }

  const { id } = await params;

  const [generation] = await db
    .select()
    .from(generations)
    .where(and(eq(generations.id, id), eq(generations.userId, session.user.id)));

  if (!generation) {
    return new Response("Not found", { status: 404 });
  }

  const absolutePath = join(process.cwd(), generation.audioPath);

  let stat;
  try {
    const { stat: s } = await import("fs/promises");
    stat = await s(absolutePath);
  } catch {
    return new Response("Audio file not found", { status: 404 });
  }

  const fileSize = stat.size;
  const rangeHeader = request.headers.get("Range");

  if (rangeHeader) {
    const match = /bytes=(\d+)-(\d*)/.exec(rangeHeader);
    if (!match) {
      return new Response("Invalid range", { status: 416 });
    }
    const start = parseInt(match[1], 10);
    const end = match[2] ? Math.min(parseInt(match[2], 10), fileSize - 1) : fileSize - 1;
    if (start >= fileSize || end >= fileSize || start > end) {
      return new Response(null, {
        status: 416,
        headers: { "Content-Range": `bytes */${fileSize}` },
      });
    }
    const chunkSize = end - start + 1;
    const fd = await (await import("fs/promises")).open(absolutePath, "r");
    const buf = Buffer.alloc(chunkSize);
    await fd.read(buf, 0, chunkSize, start);
    await fd.close();
    return new Response(buf, {
      status: 206,
      headers: {
        "Content-Type": "audio/mpeg",
        "Content-Length": chunkSize.toString(),
        "Content-Range": `bytes ${start}-${end}/${fileSize}`,
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, max-age=86400",
      },
    });
  }

  try {
    const audioBuffer = await readFile(absolutePath);
    return new Response(audioBuffer, {
      headers: {
        "Content-Type": "audio/mpeg",
        "Content-Length": audioBuffer.length.toString(),
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, max-age=86400",
      },
    });
  } catch {
    return new Response("Audio file not found", { status: 404 });
  }
}
