-- The episode that closed a fact through a current fact that already held the new statement,
-- instead of through a fact it created. Purging that episode reopens the fact, as purging the
-- creator of a normal successor does. NULL for every other close. Row-level security and the
-- grants of 0001 cover the table, so they cover this column too.
ALTER TABLE facts ADD COLUMN closed_by_episode uuid REFERENCES episodes (id) ON DELETE SET NULL;
CREATE INDEX facts_closed_by_episode_idx ON facts (closed_by_episode)
    WHERE closed_by_episode IS NOT NULL;
