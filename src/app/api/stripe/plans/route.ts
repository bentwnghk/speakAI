import { NextResponse } from "next/server";
import { getCreditPlans } from "@/lib/stripe";

export function GET() {
  return NextResponse.json({ plans: getCreditPlans() });
}
