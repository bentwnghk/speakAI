-- Update voice enum: replace old values with new ones
-- Must be done in dependency order and without invalidating the enum

-- Rename old values to temporary names first to avoid conflicts
ALTER TYPE voice RENAME VALUE 'fable' TO 'phoebe';
ALTER TYPE voice RENAME VALUE 'echo' TO 'adam';
ALTER TYPE voice RENAME VALUE 'shimmer' TO 'ava';
ALTER TYPE voice RENAME VALUE 'onyx' TO 'ollie';
