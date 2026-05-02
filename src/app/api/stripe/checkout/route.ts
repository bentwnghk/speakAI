import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { getStripe, getPlanByKey } from "@/lib/stripe";
import { createPendingPurchase } from "@/lib/db/credits";

interface CheckoutBody {
  planKey: string;
}

export async function POST(req: NextRequest) {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { planKey } = (await req.json()) as CheckoutBody;
    const plan = getPlanByKey(planKey);
    if (!plan) {
      return NextResponse.json({ error: "Invalid plan" }, { status: 400 });
    }

    const baseUrl = process.env.AUTH_URL || "http://localhost:3000";
    const stripe = getStripe();

    const checkoutSession = await stripe.checkout.sessions.create({
      mode: "payment",
      payment_method_types: ["card"],
      line_items: [
        {
          price_data: {
            currency: "hkd",
            unit_amount: Math.round(plan.priceHKD * 100),
            product_data: {
              name: `${plan.label} — ${plan.credits} Credits`,
              description: `${plan.credits} credits for Mr.🆖 SpeakAI audio generations`,
            },
          },
          quantity: 1,
        },
      ],
      metadata: {
        userId: session.user.id,
        planKey: plan.key,
        credits: plan.credits.toString(),
        amountHKD: plan.priceHKD.toString(),
      },
      success_url: `${baseUrl}/credits?success=true&session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${baseUrl}/credits?canceled=true`,
    });

    await createPendingPurchase(
      session.user.id,
      checkoutSession.id,
      plan.label,
      plan.credits,
      plan.priceHKD
    );

    return NextResponse.json({
      sessionId: checkoutSession.id,
      url: checkoutSession.url,
    });
  } catch (error: unknown) {
    const message =
      error instanceof Error
        ? error.message
        : "Failed to create checkout session";
    console.error("Checkout error:", error);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
