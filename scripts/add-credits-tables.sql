-- Add credits, credit_transactions, and purchases tables

CREATE TABLE IF NOT EXISTS "credits" (
  "userId" TEXT NOT NULL PRIMARY KEY REFERENCES "user"("id") ON DELETE CASCADE,
  "balance" REAL NOT NULL DEFAULT 0,
  "updatedAt" TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS "credit_transactions" (
  "id" TEXT NOT NULL PRIMARY KEY DEFAULT gen_random_uuid(),
  "userId" TEXT NOT NULL REFERENCES "user"("id") ON DELETE CASCADE,
  "amount" REAL NOT NULL,
  "type" VARCHAR(30) NOT NULL,
  "description" TEXT,
  "stripeSessionId" TEXT,
  "createdAt" TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS "purchases" (
  "id" TEXT NOT NULL PRIMARY KEY DEFAULT gen_random_uuid(),
  "userId" TEXT NOT NULL REFERENCES "user"("id") ON DELETE CASCADE,
  "stripeSessionId" TEXT NOT NULL UNIQUE,
  "stripePaymentIntentId" TEXT,
  "planName" VARCHAR(20) NOT NULL,
  "creditsAmount" REAL NOT NULL,
  "amountHKD" REAL NOT NULL,
  "status" VARCHAR(20) NOT NULL DEFAULT 'pending',
  "createdAt" TIMESTAMP NOT NULL DEFAULT NOW()
);
