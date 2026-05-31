CREATE TABLE IF NOT EXISTS assessments (
  id TEXT NOT NULL PRIMARY KEY DEFAULT gen_random_uuid()::text,
  userId TEXT NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
  referenceText TEXT NOT NULL,
  recognizedText TEXT NOT NULL,
  durationMs INTEGER NOT NULL,
  accuracyScore REAL NOT NULL,
  fluencyScore REAL NOT NULL,
  completenessScore REAL NOT NULL,
  prosodyScore REAL,
  pronScore REAL NOT NULL,
  words JSONB NOT NULL,
  phonemes JSONB,
  syllables JSONB,
  cost REAL NOT NULL,
  "createdAt" TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS assessments_user_idx ON assessments("userId");
