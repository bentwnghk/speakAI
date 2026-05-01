-- Migration: Add segments column for karaoke timing data
-- Date: 2026-05-01
-- Description: Adds a nullable JSON text column to store Whisper STT word-level
--              timestamps grouped into sentence segments for karaoke playback sync.

BEGIN;

ALTER TABLE "generation" ADD COLUMN IF NOT EXISTS "segments" text;

COMMIT;
