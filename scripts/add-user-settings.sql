-- User settings table for theme and locale preferences
CREATE TABLE IF NOT EXISTS "user_settings" (
  "userId" TEXT NOT NULL PRIMARY KEY REFERENCES "user"("id") ON DELETE CASCADE,
  "theme" VARCHAR(10) NOT NULL DEFAULT 'system',
  "locale" VARCHAR(10) NOT NULL DEFAULT 'en',
  "updatedAt" TIMESTAMP NOT NULL DEFAULT NOW()
);
