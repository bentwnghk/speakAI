BEGIN;

CREATE TYPE voice AS ENUM ('nova', 'alloy', 'fable', 'echo', 'shimmer', 'onyx');

CREATE TABLE IF NOT EXISTS "user" (
  "id" text PRIMARY KEY DEFAULT gen_random_uuid(),
  "name" text,
  "email" text NOT NULL UNIQUE,
  "email_verified" timestamp,
  "image" text
);

CREATE TABLE IF NOT EXISTS "account" (
  "id" text PRIMARY KEY DEFAULT gen_random_uuid(),
  "user_id" text NOT NULL REFERENCES "user"("id") ON DELETE CASCADE,
  "type" text NOT NULL,
  "provider" text NOT NULL,
  "provider_account_id" text NOT NULL,
  "refresh_token" text,
  "access_token" text,
  "expires_at" integer,
  "token_type" text,
  "scope" text,
  "id_token" text,
  "session_state" text
);

CREATE TABLE IF NOT EXISTS "session" (
  "id" text PRIMARY KEY DEFAULT gen_random_uuid(),
  "session_token" text NOT NULL UNIQUE,
  "user_id" text NOT NULL REFERENCES "user"("id") ON DELETE CASCADE,
  "expires" timestamp NOT NULL
);

CREATE TABLE IF NOT EXISTS "verification_token" (
  "identifier" text NOT NULL,
  "token" text NOT NULL UNIQUE,
  "expires" timestamp NOT NULL
);

CREATE TABLE IF NOT EXISTS "generation" (
  "id" text PRIMARY KEY DEFAULT gen_random_uuid(),
  "user_id" text NOT NULL REFERENCES "user"("id") ON DELETE CASCADE,
  "title" varchar(500) NOT NULL,
  "transcript" text NOT NULL,
  "voice" voice NOT NULL DEFAULT 'nova',
  "speed" integer NOT NULL DEFAULT 100,
  "audio_path" text NOT NULL,
  "tts_cost" text,
  "created_at" timestamp NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS "account_provider_idx" ON "account"("provider", "provider_account_id");
CREATE INDEX IF NOT EXISTS "verification_token_idx" ON "verification_token"("identifier", "token");

COMMIT;
