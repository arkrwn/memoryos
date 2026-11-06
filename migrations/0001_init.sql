CREATE EXTENSION IF NOT EXISTS vector;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'memoryos_app') THEN
        CREATE ROLE memoryos_app NOLOGIN;
    END IF;
END
$$;

CREATE TABLE episodes (
    id              uuid PRIMARY KEY,
    tenant          text NOT NULL,
    scope           text NOT NULL,
    source          text NOT NULL,
    trust           text NOT NULL CHECK (trust IN ('external', 'agent', 'user', 'system')),
    content         text NOT NULL,
    metadata        jsonb NOT NULL DEFAULT '{}',
    occurred_at     timestamptz NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    processed_at    timestamptz,
    idempotency_key text,
    UNIQUE (tenant, idempotency_key)
);
CREATE INDEX episodes_scope_idx ON episodes (tenant, scope, created_at DESC);

CREATE TABLE entities (
    id              uuid PRIMARY KEY,
    tenant          text NOT NULL,
    scope           text NOT NULL,
    kind            text NOT NULL,
    name            text NOT NULL,
    normalized_name text NOT NULL,
    aliases         text[] NOT NULL DEFAULT '{}',
    -- `aliases` under normalize_name, so lookups do not need that function in SQL.
    alias_keys      text[] NOT NULL DEFAULT '{}',
    embedding       vector(384) NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant, scope, kind, normalized_name)
);
CREATE INDEX entities_embedding_idx ON entities USING hnsw (embedding vector_cosine_ops);

CREATE TABLE facts (
    id             uuid PRIMARY KEY,
    tenant         text NOT NULL,
    scope          text NOT NULL,
    subject_id     uuid NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
    predicate      text NOT NULL,
    object_id      uuid REFERENCES entities (id) ON DELETE CASCADE,
    statement      text NOT NULL,
    embedding      vector(384) NOT NULL,
    trust          text NOT NULL CHECK (trust IN ('external', 'agent', 'user', 'system')),
    status         text NOT NULL CHECK (status IN ('active', 'quarantined', 'rejected')),
    valid_from     timestamptz NOT NULL,
    valid_to       timestamptz,
    recorded_at    timestamptz NOT NULL DEFAULT now(),
    invalidated_at timestamptz,
    superseded_by  uuid REFERENCES facts (id) ON DELETE SET NULL,
    -- Set on a quarantined fact: the fact that approval will close.
    pending_supersede_of uuid REFERENCES facts (id) ON DELETE SET NULL,
    statement_tsv  tsvector GENERATED ALWAYS AS (to_tsvector('simple', statement)) STORED,
    CHECK (valid_to IS NULL OR valid_to >= valid_from)
);
CREATE INDEX facts_scope_idx ON facts (tenant, scope, status);
CREATE INDEX facts_subject_idx ON facts (tenant, subject_id, predicate);
-- Graph walks read a hub entity's newest edges and stop at their limit. Ids are UUIDv7.
CREATE INDEX facts_subject_recent_idx ON facts (tenant, subject_id, id DESC);
CREATE INDEX facts_object_recent_idx ON facts (tenant, object_id, id DESC);
CREATE INDEX facts_tsv_idx ON facts USING gin (statement_tsv);
CREATE INDEX facts_embedding_idx ON facts USING hnsw (embedding vector_cosine_ops);
CREATE INDEX facts_superseded_by_idx ON facts (superseded_by) WHERE superseded_by IS NOT NULL;
CREATE INDEX facts_pending_supersede_idx ON facts (pending_supersede_of) WHERE pending_supersede_of IS NOT NULL;

CREATE TABLE fact_sources (
    fact_id    uuid NOT NULL REFERENCES facts (id) ON DELETE CASCADE,
    episode_id uuid NOT NULL REFERENCES episodes (id) ON DELETE CASCADE,
    tenant     text NOT NULL,
    PRIMARY KEY (fact_id, episode_id)
);
CREATE INDEX fact_sources_episode_idx ON fact_sources (episode_id);

CREATE TABLE blocks (
    id         uuid PRIMARY KEY,
    tenant     text NOT NULL,
    agent_id   text NOT NULL,
    label      text NOT NULL,
    content    text NOT NULL,
    max_chars  integer NOT NULL CHECK (max_chars > 0),
    version    bigint NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant, agent_id, label)
);

CREATE TABLE block_versions (
    block_id  uuid NOT NULL REFERENCES blocks (id) ON DELETE CASCADE,
    tenant    text NOT NULL,
    version   bigint NOT NULL,
    content   text NOT NULL,
    edited_by text NOT NULL,
    edited_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (block_id, version)
);

CREATE TABLE audit_events (
    id         uuid PRIMARY KEY,
    tenant     text NOT NULL,
    actor      text NOT NULL,
    action     text NOT NULL,
    target_id  uuid,
    detail     jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX audit_events_tenant_idx ON audit_events (tenant, created_at DESC);

-- Not tenant-restricted: keys are read before the tenant is known.
CREATE TABLE api_keys (
    id         uuid PRIMARY KEY,
    tenant     text NOT NULL,
    name       text NOT NULL,
    key_hash   text NOT NULL UNIQUE,
    scopes     text[] NOT NULL,
    max_trust  text NOT NULL CHECK (max_trust IN ('external', 'agent', 'user', 'system')),
    is_admin   boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    revoked_at timestamptz
);

-- Row-level security. The store runs `SET LOCAL ROLE memoryos_app` and sets `app.tenant`
-- in each tenant transaction, so policies apply even if the login role is a superuser.
DO $$
DECLARE
    t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['episodes', 'entities', 'facts', 'fact_sources', 'blocks',
                             'block_versions', 'audit_events']
    LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
        EXECUTE format(
            'CREATE POLICY tenant_isolation ON %I TO memoryos_app
             USING (tenant = current_setting(''app.tenant'', true))
             WITH CHECK (tenant = current_setting(''app.tenant'', true))', t);
        EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON %I TO memoryos_app', t);
    END LOOP;
END
$$;

-- Text search under RLS cannot use GIN indexes, because `@@` is not leakproof (measured on
-- 200k facts: 0.5 ms with the index, 29 ms without). This function runs as the table owner and
-- takes the tenant from app.tenant itself, so callers cannot pass another tenant. It returns
-- ids only; callers load rows under RLS. FORCE is off on this table so that a non-superuser owner
-- can run it. Tenant transactions still use memoryos_app, which stays subject to policy.
ALTER TABLE facts NO FORCE ROW LEVEL SECURITY;

CREATE FUNCTION search_fact_ids_text(p_scopes text[], p_query text, p_as_of timestamptz, p_known_at timestamptz, p_limit bigint)
RETURNS TABLE (id uuid, rank real)
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = public
AS $$
    SELECT f.id, ts_rank(f.statement_tsv, q) AS rank
    FROM facts f, websearch_to_tsquery('simple', p_query) q
    WHERE f.tenant = current_setting('app.tenant', true)
      AND f.scope = ANY (p_scopes)
      AND f.status = 'active'
      AND f.statement_tsv @@ q
      AND f.valid_from <= p_as_of AND (f.valid_to IS NULL OR f.valid_to > p_as_of)
      AND f.recorded_at <= p_known_at
      AND (f.invalidated_at IS NULL OR f.invalidated_at > p_known_at)
    ORDER BY rank DESC, f.id
    LIMIT p_limit;
$$;

REVOKE ALL ON FUNCTION search_fact_ids_text(text[], text, timestamptz, timestamptz, bigint) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION search_fact_ids_text(text[], text, timestamptz, timestamptz, bigint) TO memoryos_app;
