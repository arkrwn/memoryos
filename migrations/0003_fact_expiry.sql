-- When a fact stops being true without anything replacing it, such as "on leave until Friday". From
-- then on the fact is not current: recall, candidates and matching skip it as they skip a closed
-- fact, and it stays in history and in the past facts of its scope. Unlike `valid_to`, which a
-- supersede or a purge rewrites, this is the commit's own statement and is never changed. NULL means
-- no expiry. Row-level security and the grants of 0001 cover the table, so they cover this column.
ALTER TABLE facts ADD COLUMN expires_at timestamptz;
ALTER TABLE facts ADD CONSTRAINT facts_expires_after_valid_from
    CHECK (expires_at IS NULL OR expires_at > valid_from);

-- 0001's text search with the expiry filter added, so text and vector search agree on what is
-- current. The signature is unchanged, so the owner, SECURITY DEFINER and the grants carry over;
-- they are restated below to keep this file the reference for them.
CREATE OR REPLACE FUNCTION search_fact_ids_text(p_scopes text[], p_query text, p_as_of timestamptz, p_known_at timestamptz, p_limit bigint)
RETURNS TABLE (id uuid, rank real)
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    q tsquery := websearch_to_tsquery('simple', p_query);
BEGIN
    RETURN QUERY EXECUTE $sql$
        SELECT c.id, ts_rank(c.statement_tsv, $1) AS rank
        FROM (
            SELECT f.id, f.statement_tsv FROM facts f
            WHERE f.tenant = current_setting('app.tenant', true)
              AND f.scope = ANY ($2)
              AND f.status = 'active'
              AND f.statement_tsv @@ $1
              AND f.valid_from <= $3 AND (f.valid_to IS NULL OR f.valid_to > $3)
              AND (f.expires_at IS NULL OR f.expires_at > $3)
              AND f.recorded_at <= $4
              AND (f.invalidated_at IS NULL OR f.invalidated_at > $4)
            ORDER BY f.id DESC
            LIMIT GREATEST(1000, $5)
        ) c
        ORDER BY rank DESC, c.id
        LIMIT $5
    $sql$ USING q, p_scopes, p_as_of, p_known_at, p_limit;
END;
$$;

REVOKE ALL ON FUNCTION search_fact_ids_text(text[], text, timestamptz, timestamptz, bigint) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION search_fact_ids_text(text[], text, timestamptz, timestamptz, bigint) TO memoryos_app;
