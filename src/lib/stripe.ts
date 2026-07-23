import Stripe from "stripe";

let _stripe: Stripe | null = null;

export function getStripe(): Stripe {
  if (!_stripe) {
    _stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
      apiVersion: "2026-04-22.dahlia",
    });
  }
  return _stripe;
}

export interface CreditPlan {
  key: string;
  label: string;
  credits: number;
  priceHKD: number;
  highlight?: boolean;
}

function getPlans(): CreditPlan[] {
  return [
    {
      key: "plan_a",
      label: "Starter",
      credits: parseFloat(process.env.STRIPE_PLAN_A_CREDITS || "15"),
      priceHKD: parseFloat(process.env.STRIPE_PLAN_A_PRICE_HKD || "15"),
    },
    {
      key: "plan_b",
      label: "Best Value",
      credits: parseFloat(process.env.STRIPE_PLAN_B_CREDITS || "50"),
      priceHKD: parseFloat(process.env.STRIPE_PLAN_B_PRICE_HKD || "45"),
      highlight: true,
    },
  ];
}

export function getCreditPlans(): CreditPlan[] {
  return getPlans();
}

export function getPlanByKey(key: string): CreditPlan | undefined {
  return getPlans().find((p) => p.key === key);
}
