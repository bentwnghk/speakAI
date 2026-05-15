BEGIN;

DROP TABLE IF EXISTS "generation" CASCADE;
DROP TABLE IF EXISTS "user_settings" CASCADE;
DROP TABLE IF EXISTS "session" CASCADE;
DROP TABLE IF EXISTS "account" CASCADE;
DROP TABLE IF EXISTS "verification_token" CASCADE;
DROP TABLE IF EXISTS "user" CASCADE;
DROP TYPE IF EXISTS voice CASCADE;

CREATE TYPE voice AS ENUM ('nova', 'alloy', 'phoebe', 'adam', 'ava', 'ollie');

CREATE TABLE "user" (
  "id" text PRIMARY KEY DEFAULT gen_random_uuid(),
  "name" text,
  "email" text NOT NULL UNIQUE,
  "emailVerified" timestamp,
  "image" text
);

CREATE TABLE "account" (
  "id" text PRIMARY KEY DEFAULT gen_random_uuid(),
  "userId" text NOT NULL REFERENCES "user"("id") ON DELETE CASCADE,
  "type" text NOT NULL,
  "provider" text NOT NULL,
  "providerAccountId" text NOT NULL,
  "refresh_token" text,
  "access_token" text,
  "expires_at" integer,
  "token_type" text,
  "scope" text,
  "id_token" text,
  "session_state" text
);

CREATE TABLE "session" (
  "id" text PRIMARY KEY DEFAULT gen_random_uuid(),
  "sessionToken" text NOT NULL UNIQUE,
  "userId" text NOT NULL REFERENCES "user"("id") ON DELETE CASCADE,
  "expires" timestamp NOT NULL
);

CREATE TABLE "verification_token" (
  "identifier" text NOT NULL,
  "token" text NOT NULL UNIQUE,
  "expires" timestamp NOT NULL
);

CREATE TABLE "generation" (
  "id" text PRIMARY KEY DEFAULT gen_random_uuid(),
  "userId" text NOT NULL REFERENCES "user"("id") ON DELETE CASCADE,
  "title" varchar(500) NOT NULL,
  "transcript" text NOT NULL,
  "voice" voice NOT NULL DEFAULT 'nova',
  "speed" integer NOT NULL DEFAULT 100,
  "audioPath" text NOT NULL,
  "segments" text,
  "ttsCost" text,
  "createdAt" timestamp NOT NULL DEFAULT now(),
  "expiresAt" timestamp
);

CREATE INDEX "account_provider_idx" ON "account"("provider", "providerAccountId");
CREATE INDEX "verification_token_idx" ON "verification_token"("identifier", "token");

CREATE TABLE "credits" (
  "userId" TEXT NOT NULL PRIMARY KEY REFERENCES "user"("id") ON DELETE CASCADE,
  "balance" REAL NOT NULL DEFAULT 0,
  "updatedAt" TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE "credit_transactions" (
  "id" TEXT NOT NULL PRIMARY KEY DEFAULT gen_random_uuid(),
  "userId" TEXT NOT NULL REFERENCES "user"("id") ON DELETE CASCADE,
  "amount" REAL NOT NULL,
  "type" VARCHAR(30) NOT NULL,
  "description" TEXT,
  "stripeSessionId" TEXT,
  "createdAt" TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE "purchases" (
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

CREATE TABLE "user_settings" (
  "userId" TEXT NOT NULL PRIMARY KEY REFERENCES "user"("id") ON DELETE CASCADE,
  "theme" VARCHAR(10) NOT NULL DEFAULT 'system',
  "locale" VARCHAR(10) NOT NULL DEFAULT 'en',
  "updatedAt" TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE "sign_in_logs" (
  "id" TEXT NOT NULL PRIMARY KEY DEFAULT gen_random_uuid(),
  "userId" TEXT NOT NULL REFERENCES "user"("id") ON DELETE CASCADE,
  "provider" TEXT NOT NULL DEFAULT 'google',
  "createdAt" TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX "sign_in_logs_userId_idx" ON "sign_in_logs"("userId");
CREATE INDEX "sign_in_logs_createdAt_idx" ON "sign_in_logs"("createdAt");

COMMIT;
