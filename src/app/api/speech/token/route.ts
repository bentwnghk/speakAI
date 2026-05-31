import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";

interface AzureEndpoint {
  key: string;
  region: string;
}

function loadAzureEndpoints(): AzureEndpoint[] {
  const endpoints: AzureEndpoint[] = [];
  for (let i = 1; i <= 20; i++) {
    const key = process.env[`AZURE_SPEECH_KEY_${i}`];
    const region = process.env[`AZURE_SPEECH_REGION_${i}`];
    if (key && region) endpoints.push({ key, region });
  }
  if (endpoints.length === 0) {
    const key = process.env.AZURE_SPEECH_KEY;
    const region = process.env.AZURE_SPEECH_REGION;
    if (key && region) endpoints.push({ key, region });
  }
  return endpoints;
}

const AZURE_ENDPOINTS = loadAzureEndpoints();
let rrCursor = 0;

export async function GET() {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  if (AZURE_ENDPOINTS.length === 0) {
    return NextResponse.json(
      { error: "No Azure Speech endpoints configured" },
      { status: 500 }
    );
  }

  const ep = AZURE_ENDPOINTS[rrCursor % AZURE_ENDPOINTS.length];
  rrCursor++;

  try {
    const res = await fetch(
      `https://${ep.region}.api.cognitive.microsoft.com/sts/v1.0/issueToken`,
      {
        method: "POST",
        headers: {
          "Ocp-Apim-Subscription-Key": ep.key,
          "Content-Length": "0",
        },
      }
    );

    if (!res.ok) {
      throw new Error(`Token request failed: ${res.status}`);
    }

    const token = await res.text();
    return NextResponse.json({ token, region: ep.region });
  } catch (error) {
    console.error("Failed to issue speech token:", error);
    return NextResponse.json(
      { error: "Failed to issue speech token" },
      { status: 500 }
    );
  }
}
