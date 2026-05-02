import { db } from "@/lib/db";
import { credits, creditTransactions, purchases } from "./schema";
import { eq, sql } from "drizzle-orm";

const WELCOME_CREDITS = parseFloat(process.env.WELCOME_CREDITS || "3");

export function getWelcomeCredits() {
  return WELCOME_CREDITS;
}

export async function getUserBalance(userId: string): Promise<number> {
  const [row] = await db
    .select({ balance: credits.balance })
    .from(credits)
    .where(eq(credits.userId, userId));
  return row?.balance ?? 0;
}

export async function ensureCreditsRecord(userId: string): Promise<void> {
  const existing = await db
    .select({ userId: credits.userId })
    .from(credits)
    .where(eq(credits.userId, userId));

  if (existing.length === 0) {
    await db.insert(credits).values({
      userId,
      balance: WELCOME_CREDITS,
    });
    await db.insert(creditTransactions).values({
      userId,
      amount: WELCOME_CREDITS,
      type: "welcome_bonus",
      description: `Welcome bonus: ${WELCOME_CREDITS} free credits`,
    });
  }
}

export async function deductCredits(
  userId: string,
  amount: number,
  description: string
): Promise<{
  success: boolean;
  balance: number;
  transactionId?: string;
  error?: string;
}> {
  const [row] = await db
    .select({ balance: credits.balance })
    .from(credits)
    .where(eq(credits.userId, userId));

  if (!row) {
    return { success: false, balance: 0, error: "No credits record found" };
  }

  if (row.balance < amount) {
    return {
      success: false,
      balance: row.balance,
      error: "Insufficient credits",
    };
  }

  await db
    .update(credits)
    .set({
      balance: sql`${credits.balance} - ${amount}`,
      updatedAt: new Date(),
    })
    .where(eq(credits.userId, userId));

  const [txn] = await db
    .insert(creditTransactions)
    .values({
      userId,
      amount: -amount,
      type: "generation",
      description,
    })
    .returning({ id: creditTransactions.id });

  return {
    success: true,
    balance: row.balance - amount,
    transactionId: txn.id,
  };
}

export async function refundCredits(
  userId: string,
  amount: number,
  description: string
): Promise<void> {
  await db
    .update(credits)
    .set({
      balance: sql`${credits.balance} + ${amount}`,
      updatedAt: new Date(),
    })
    .where(eq(credits.userId, userId));

  await db.insert(creditTransactions).values({
    userId,
    amount,
    type: "refund",
    description,
  });
}

export async function createPendingPurchase(
  userId: string,
  stripeSessionId: string,
  planName: string,
  creditsAmount: number,
  amountHKD: number
): Promise<void> {
  await db.insert(purchases).values({
    userId,
    stripeSessionId,
    planName,
    creditsAmount,
    amountHKD,
    status: "pending",
  });
}

export async function addCreditsFromPurchase(
  userId: string,
  amount: number,
  stripeSessionId: string,
  stripePaymentIntentId: string | null,
  planName: string,
  amountHKD: number
): Promise<void> {
  await db.transaction(async (tx) => {
    const [existing] = await tx
      .select({ id: purchases.id })
      .from(purchases)
      .where(eq(purchases.stripeSessionId, stripeSessionId));

    if (existing) {
      await tx
        .update(purchases)
        .set({ status: "completed", stripePaymentIntentId })
        .where(eq(purchases.stripeSessionId, stripeSessionId));
    } else {
      await tx.insert(purchases).values({
        userId,
        stripeSessionId,
        stripePaymentIntentId,
        planName,
        creditsAmount: amount,
        amountHKD,
        status: "completed",
      });
    }

    await tx
      .update(credits)
      .set({
        balance: sql`${credits.balance} + ${amount}`,
        updatedAt: new Date(),
      })
      .where(eq(credits.userId, userId));

    await tx.insert(creditTransactions).values({
      userId,
      amount,
      type: "purchase",
      description: `Purchased ${amount} credits (${planName})`,
      stripeSessionId,
    });
  });
}

export async function markPurchaseFailed(
  stripeSessionId: string
): Promise<void> {
  await db
    .update(purchases)
    .set({ status: "failed" })
    .where(eq(purchases.stripeSessionId, stripeSessionId));
}

export async function getUserPurchases(userId: string) {
  return db
    .select()
    .from(purchases)
    .where(eq(purchases.userId, userId))
    .orderBy(sql`${purchases.createdAt} DESC`);
}
