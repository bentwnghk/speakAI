-- Add expiresAt column to generation table for audio retention policy
ALTER TABLE "generation" ADD COLUMN IF NOT EXISTS "expiresAt" TIMESTAMP;

-- Backfill existing rows: set expiresAt = createdAt + 365 days (or AUDIO_RETENTION_DAYS if custom)
-- Default uses 365 days; adjust the interval if your deployment uses a different retention period.
UPDATE "generation"
SET "expiresAt" = "createdAt" + INTERVAL '365 days'
WHERE "expiresAt" IS NULL;
