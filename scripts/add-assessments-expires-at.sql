ALTER TABLE assessments ADD COLUMN IF NOT EXISTS "expiresAt" TIMESTAMP;

UPDATE assessments
SET "expiresAt" = "createdAt" + INTERVAL '365 days'
WHERE "expiresAt" IS NULL AND "audioPath" IS NOT NULL;
