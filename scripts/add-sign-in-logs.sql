CREATE TABLE IF NOT EXISTS "sign_in_logs" (
  "id" TEXT NOT NULL PRIMARY KEY DEFAULT gen_random_uuid(),
  "userId" TEXT NOT NULL REFERENCES "user"("id") ON DELETE CASCADE,
  "provider" TEXT NOT NULL DEFAULT 'google',
  "createdAt" TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS "sign_in_logs_userId_idx" ON "sign_in_logs"("userId");
CREATE INDEX IF NOT EXISTS "sign_in_logs_createdAt_idx" ON "sign_in_logs"("createdAt");
