-- SQLite equivalent of PostgreSQL migrations 0001-0003. UUIDs are canonical lowercase
-- hyphenated TEXT; timestamps are Unix microseconds; JSON values and arrays are JSON TEXT.

CREATE TABLE episodes (
    id              TEXT PRIMARY KEY CHECK (length(id) = 36),
    tenant          TEXT NOT NULL,
    scope           TEXT NOT NULL,
    source          TEXT NOT NULL,
    trust           TEXT NOT NULL CHECK (trust IN ('external', 'agent', 'user', 'system')),
    content         TEXT NOT NULL,
    metadata        TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(metadata)),
    occurred_at     INTEGER NOT NULL,
    created_at      INTEGER NOT NULL,
    processed_at    INTEGER,
    idempotency_key TEXT,
    UNIQUE (tenant, idempotency_key),
    UNIQUE (tenant, id)
);
CREATE INDEX episodes_scope_idx ON episodes (tenant, scope, created_at DESC);

CREATE TABLE entities (
    row_id          INTEGER PRIMARY KEY,
    id              TEXT NOT NULL UNIQUE CHECK (length(id) = 36),
    tenant          TEXT NOT NULL,
    scope           TEXT NOT NULL,
    kind            TEXT NOT NULL,
    name            TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    aliases         TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(aliases) AND json_type(aliases) = 'array'),
    alias_keys      TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(alias_keys) AND json_type(alias_keys) = 'array'),
    created_at      INTEGER NOT NULL,
    UNIQUE (tenant, scope, kind, normalized_name),
    UNIQUE (tenant, id)
);
CREATE INDEX entities_scope_idx ON entities (tenant, scope, kind, normalized_name);

CREATE TABLE facts (
    row_id               INTEGER PRIMARY KEY,
    id                   TEXT NOT NULL UNIQUE CHECK (length(id) = 36),
    tenant               TEXT NOT NULL,
    scope                TEXT NOT NULL,
    subject_id           TEXT NOT NULL CHECK (length(subject_id) = 36),
    predicate            TEXT NOT NULL,
    object_id            TEXT CHECK (object_id IS NULL OR length(object_id) = 36),
    statement            TEXT NOT NULL,
    trust                TEXT NOT NULL CHECK (trust IN ('external', 'agent', 'user', 'system')),
    status               TEXT NOT NULL CHECK (status IN ('active', 'quarantined', 'rejected')),
    valid_from           INTEGER NOT NULL,
    valid_to             INTEGER,
    expires_at           INTEGER,
    recorded_at          INTEGER NOT NULL,
    invalidated_at       INTEGER,
    superseded_by        TEXT REFERENCES facts (id) ON DELETE SET NULL,
    pending_supersede_of TEXT REFERENCES facts (id) ON DELETE SET NULL,
    closed_by_episode    TEXT REFERENCES episodes (id) ON DELETE SET NULL,
    UNIQUE (tenant, id),
    FOREIGN KEY (tenant, subject_id) REFERENCES entities (tenant, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant, object_id) REFERENCES entities (tenant, id) ON DELETE CASCADE,
    CHECK (valid_to IS NULL OR valid_to >= valid_from),
    CHECK (expires_at IS NULL OR expires_at > valid_from)
);
CREATE INDEX facts_scope_idx ON facts (tenant, scope, status);
CREATE INDEX facts_subject_idx ON facts (tenant, subject_id, predicate);
CREATE INDEX facts_subject_recent_idx ON facts (tenant, subject_id, id DESC);
CREATE INDEX facts_object_recent_idx ON facts (tenant, object_id, id DESC);
CREATE INDEX facts_superseded_by_idx ON facts (tenant, superseded_by) WHERE superseded_by IS NOT NULL;
CREATE INDEX facts_pending_supersede_idx ON facts (tenant, pending_supersede_of) WHERE pending_supersede_of IS NOT NULL;
CREATE INDEX facts_closed_by_episode_idx ON facts (tenant, closed_by_episode) WHERE closed_by_episode IS NOT NULL;

CREATE TABLE fact_sources (
    fact_id    TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    tenant     TEXT NOT NULL,
    PRIMARY KEY (fact_id, episode_id),
    FOREIGN KEY (tenant, fact_id) REFERENCES facts (tenant, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant, episode_id) REFERENCES episodes (tenant, id) ON DELETE CASCADE
);
CREATE INDEX fact_sources_episode_idx ON fact_sources (tenant, episode_id, fact_id);

CREATE TABLE entity_alias_sources (
    entity_id  TEXT NOT NULL,
    alias_key  TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    tenant     TEXT NOT NULL,
    alias      TEXT NOT NULL,
    PRIMARY KEY (entity_id, alias_key, episode_id),
    FOREIGN KEY (tenant, entity_id) REFERENCES entities (tenant, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant, episode_id) REFERENCES episodes (tenant, id) ON DELETE CASCADE
);
CREATE INDEX entity_alias_sources_episode_idx ON entity_alias_sources (tenant, episode_id);

CREATE TABLE blocks (
    id         TEXT PRIMARY KEY CHECK (length(id) = 36),
    tenant     TEXT NOT NULL,
    agent_id   TEXT NOT NULL,
    label      TEXT NOT NULL,
    content    TEXT NOT NULL,
    max_chars  INTEGER NOT NULL CHECK (max_chars > 0),
    version    INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    UNIQUE (tenant, agent_id, label),
    UNIQUE (tenant, id)
);

CREATE TABLE block_versions (
    block_id  TEXT NOT NULL,
    tenant    TEXT NOT NULL,
    version   INTEGER NOT NULL,
    content   TEXT NOT NULL,
    edited_by TEXT NOT NULL,
    edited_at INTEGER NOT NULL,
    PRIMARY KEY (block_id, version),
    FOREIGN KEY (tenant, block_id) REFERENCES blocks (tenant, id) ON DELETE CASCADE
);
CREATE INDEX block_versions_tenant_idx ON block_versions (tenant, block_id, version DESC);

CREATE TABLE audit_events (
    id         TEXT PRIMARY KEY CHECK (length(id) = 36),
    tenant     TEXT NOT NULL,
    actor      TEXT NOT NULL,
    action     TEXT NOT NULL,
    target_id  TEXT CHECK (target_id IS NULL OR length(target_id) = 36),
    detail     TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(detail) AND json_type(detail) = 'object'),
    created_at INTEGER NOT NULL
);
CREATE INDEX audit_events_tenant_idx ON audit_events (tenant, created_at DESC);

CREATE TABLE api_keys (
    id         TEXT PRIMARY KEY CHECK (length(id) = 36),
    tenant     TEXT NOT NULL,
    name       TEXT NOT NULL,
    key_hash   TEXT NOT NULL UNIQUE CHECK (length(key_hash) = 64),
    scopes     TEXT NOT NULL CHECK (json_valid(scopes) AND json_type(scopes) = 'array'),
    max_trust  TEXT NOT NULL CHECK (max_trust IN ('external', 'agent', 'user', 'system')),
    is_admin   INTEGER NOT NULL DEFAULT 0 CHECK (is_admin IN (0, 1)),
    created_at INTEGER NOT NULL,
    revoked_at INTEGER
);

-- sqlite-vec uses the base tables' integer rowids. entities.row_id and facts.row_id are INTEGER
-- PRIMARY KEY aliases, so VACUUM cannot renumber them and detach the vectors. Base and vector rows
-- are always written or removed in one transaction. Searches scan vec0 exactly and apply every
-- tenant/scope/time filter
-- before ordering, rather than relying on an approximate pre-filtered KNN query.
CREATE VIRTUAL TABLE entity_embeddings USING vec0(embedding float[384]);
CREATE VIRTUAL TABLE fact_embeddings USING vec0(embedding float[384]);

-- External-content FTS stays synchronized through triggers, including direct SQL writes. The
-- store additionally binds a safely quoted MATCH expression and filters the joined base row.
CREATE VIRTUAL TABLE facts_fts USING fts5(
    statement,
    content='facts',
    content_rowid='rowid',
    tokenize='unicode61'
);

CREATE TRIGGER facts_fts_insert AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts (rowid, statement) VALUES (new.rowid, new.statement);
END;
CREATE TRIGGER facts_fts_delete AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts (facts_fts, rowid, statement) VALUES ('delete', old.rowid, old.statement);
END;
CREATE TRIGGER facts_fts_update AFTER UPDATE OF statement ON facts BEGIN
    INSERT INTO facts_fts (facts_fts, rowid, statement) VALUES ('delete', old.rowid, old.statement);
    INSERT INTO facts_fts (rowid, statement) VALUES (new.rowid, new.statement);
END;
