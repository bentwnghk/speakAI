-- Fix assessment audioPath values that were incorrectly stored as absolute
-- filesystem paths instead of relative paths.
--
-- Affected rows: any assessment where audioPath starts with '/' (absolute).
-- These were written by a bug in POST /api/assessment that stored
-- join(process.cwd(), "data", "recording", filename) instead of
-- join("data", "recording", filename).
--
-- The relative path is always the trailing "data/recording/<filename>.webm"
-- segment, so we extract it with a regex regardless of what the leading
-- prefix was.
UPDATE assessments
SET    "audioPath" = regexp_replace("audioPath", '^.*(data/recording/[^/]+)$', '\1')
WHERE  "audioPath" IS NOT NULL
  AND  "audioPath" LIKE '/%';
