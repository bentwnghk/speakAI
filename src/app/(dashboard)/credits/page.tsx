"use client";

import { useState, useEffect, useCallback } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { useCredits } from "@/hooks/use-credits";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  CheckCircle,
  XCircle,
  Loader2,
  Coins,
  ShoppingCart,
  Zap,
  Package,
} from "lucide-react";

interface PlanConfig {
  key: string;
  label: string;
  credits: number;
  priceHKD: number;
  highlight?: boolean;
}

interface PurchaseRecord {
  id: string;
  planName: string;
  creditsAmount: number;
  amountHKD: number;
  status: string;
  createdAt: string;
}

interface PlansResponse {
  plans: PlanConfig[];
}

interface PurchasesResponse {
  purchases: PurchaseRecord[];
}

interface CheckoutResponse {
  url?: string;
  error?: string;
}

export default function CreditsPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { balance, refreshBalance } = useCredits();
  const [plans, setPlans] = useState<PlanConfig[]>([]);
  const [loading, setLoading] = useState<string | null>(null);
  const [purchases, setPurchases] = useState<PurchaseRecord[]>([]);

  const isSuccess = searchParams.get("success") === "true";
  const isCanceled = searchParams.get("canceled") === "true";

  useEffect(() => {
    if (isSuccess) {
      void refreshBalance();
      const timer = setTimeout(() => {
        router.replace("/credits");
      }, 8000);
      return () => clearTimeout(timer);
    }
  }, [isSuccess, refreshBalance, router]);

  useEffect(() => {
    fetch("/api/stripe/plans")
      .then((res) => res.json())
      .then((data: PlansResponse) => {
        setPlans(data.plans || []);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetch("/api/user/purchases")
      .then((res) => res.json())
      .then((data: PurchasesResponse) => setPurchases(data.purchases || []))
      .catch(() => {});
  }, []);

  const handlePurchase = useCallback(async (planKey: string) => {
    setLoading(planKey);
    try {
      const res = await fetch("/api/stripe/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ planKey }),
      });
      const data = (await res.json()) as CheckoutResponse;
      if (data.url) {
        window.location.href = data.url;
      } else {
        throw new Error(data.error || "Failed to create checkout session");
      }
    } catch (err) {
      alert(err instanceof Error ? err.message : "Purchase failed");
    } finally {
      setLoading(null);
    }
  }, []);

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-bold">Credits</h1>
        <p className="text-muted-foreground mt-2">
          Purchase credits to generate audio
        </p>
      </div>

      {isSuccess && (
        <div className="mb-6 flex items-center gap-3 rounded-lg border border-green-200 bg-green-50 p-4 dark:border-green-900 dark:bg-green-950">
          <CheckCircle className="h-5 w-5 text-green-600 dark:text-green-400" />
          <div>
            <p className="font-medium text-green-800 dark:text-green-200">
              Payment successful!
            </p>
            <p className="text-sm text-green-700 dark:text-green-300">
              Your credits have been added to your account.
            </p>
          </div>
        </div>
      )}

      {isCanceled && (
        <div className="mb-6 flex items-center gap-3 rounded-lg border border-yellow-200 bg-yellow-50 p-4 dark:border-yellow-900 dark:bg-yellow-950">
          <XCircle className="h-5 w-5 text-yellow-600 dark:text-yellow-400" />
          <p className="text-sm text-yellow-700 dark:text-yellow-300">
            Payment was canceled. No charges were made.
          </p>
        </div>
      )}

      <div className="mb-8 flex items-center justify-center gap-2 rounded-lg bg-muted p-4">
        <Coins className="h-5 w-5" />
        <span className="text-lg font-semibold">
          HK${balance !== null ? balance.toFixed(2) : "..."}
        </span>
        <span className="text-muted-foreground">remaining</span>
      </div>

      <div className="mb-8 grid gap-6 pt-4 md:grid-cols-2">
        {plans.map((plan) => {
          const starterPlan = plans.find((p) => !p.highlight);
          const savedPct = starterPlan
            ? Math.round(
                (1 -
                  plan.priceHKD /
                    ((plan.credits * starterPlan.priceHKD) /
                      starterPlan.credits)) *
                  100
              )
            : 0;

          return (
            <Card
              key={plan.key}
              className={`relative pt-6 ${
                plan.highlight
                  ? "border-primary !overflow-visible shadow-lg scale-[1.02]"
                  : ""
              }`}
            >
              {plan.highlight && (
                <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                  <Badge className="bg-primary px-3 py-1 text-primary-foreground">
                    <Zap className="mr-1 h-3 w-3" />
                    Best Value
                  </Badge>
                </div>
              )}
              <CardHeader className="items-center pb-2 text-center">
                <div className="mb-2">
                  {plan.highlight ? (
                    <Zap className="h-8 w-8 text-primary" />
                  ) : (
                    <Package className="h-8 w-8 text-muted-foreground" />
                  )}
                </div>
                <CardTitle className="text-xl">{plan.label}</CardTitle>
                <CardDescription>
                  {plan.credits} Credits (HK${plan.credits.toFixed(2)})
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 text-center">
                <div>
                  <span className="text-4xl font-bold">
                    HK${plan.priceHKD}
                  </span>
                </div>
                <p className="text-sm text-muted-foreground">
                  HK${(plan.priceHKD / plan.credits).toFixed(2)} per credit
                </p>
                {plan.highlight && savedPct > 0 && (
                  <p className="text-sm font-medium text-primary">
                    Save {savedPct}% compared to Starter
                  </p>
                )}
                <Button
                  className="w-full"
                  variant={plan.highlight ? "default" : "outline"}
                  size="lg"
                  onClick={() => void handlePurchase(plan.key)}
                  disabled={loading !== null}
                >
                  {loading === plan.key ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Redirecting...
                    </>
                  ) : (
                    <>
                      <ShoppingCart className="mr-2 h-4 w-4" />
                      Buy {plan.credits} Credits
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {purchases.length > 0 && (
        <>
          <Separator className="my-8" />
          <div>
            <h2 className="mb-4 text-xl font-semibold">Purchase History</h2>
            <div className="rounded-md border">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="p-3 text-left font-medium">Date</th>
                    <th className="p-3 text-left font-medium">Package</th>
                    <th className="p-3 text-right font-medium">Amount</th>
                    <th className="p-3 text-right font-medium">Credits</th>
                    <th className="p-3 text-right font-medium">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {purchases.map((p) => (
                    <tr key={p.id} className="border-b last:border-0">
                      <td className="p-3">
                        {new Date(p.createdAt).toLocaleDateString("en-HK", {
                          timeZone: "Asia/Hong_Kong",
                          year: "numeric",
                          month: "short",
                          day: "numeric",
                        })}
                      </td>
                      <td className="p-3">{p.planName}</td>
                      <td className="p-3 text-right">
                        HK${p.amountHKD.toFixed(2)}
                      </td>
                      <td className="p-3 text-right">
                        {p.creditsAmount.toFixed(2)}
                      </td>
                      <td className="p-3 text-right">
                        <Badge
                          variant={
                            p.status === "completed"
                              ? "default"
                              : p.status === "pending"
                                ? "secondary"
                                : "destructive"
                          }
                        >
                          {p.status}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
