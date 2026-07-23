import { NextRequest, NextResponse } from "next/server";
import { getStripe, getPlanByKey } from "@/lib/stripe";
import { addCreditsFromPurchase, markPurchaseFailed } from "@/lib/db/credits";

export async function POST(req: NextRequest) {
  const body = await req.text();
  const signature = req.headers.get("stripe-signature");
  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;

  if (!signature || !webhookSecret) {
    return NextResponse.json(
      { error: "Missing signature or secret" },
      { status: 400 }
    );
  }

  let event;
  try {
    const stripe = getStripe();
    event = stripe.webhooks.constructEvent(body, signature, webhookSecret);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Invalid signature";
    console.error("Webhook signature verification failed:", message);
    return NextResponse.json({ error: message }, { status: 400 });
  }

  try {
    switch (event.type) {
      case "checkout.session.completed": {
        const session = event.data.object;
        const userId = session.metadata?.userId;
        const planKey = session.metadata?.planKey;
        const credits = parseFloat(session.metadata?.credits || "0");
        const amountHKD = parseFloat(session.metadata?.amountHKD || "0");

        if (!userId || !credits) {
          console.error(
            "Missing metadata in checkout session:",
            session.id
          );
          break;
        }

        await addCreditsFromPurchase(
          userId,
          credits,
          session.id,
          (session.payment_intent as string | null) ?? null,
          planKey ? (getPlanByKey(planKey)?.label || planKey) : "unknown",
          amountHKD
        );
        break;
      }
      case "checkout.session.expired": {
        const session = event.data.object;
        await markPurchaseFailed(session.id);
        break;
      }
      default:
        console.log(`Unhandled event type: ${event.type}`);
    }
  } catch (error) {
    console.error("Webhook handler error:", error);
    return NextResponse.json(
      { error: "Webhook handler failed" },
      { status: 500 }
    );
  }

  return NextResponse.json({ received: true });
}
