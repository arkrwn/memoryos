-- Which episode earned which alias, so purging the episode can take the alias back. Numbered after
-- task T16 and not 0002, so it cannot collide with a migration written at the same time.
CREATE TABLE entity_alias_sources (
    entity_id  uuid NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
    alias_key  text NOT NULL,
    episode_id uuid NOT NULL REFERENCES episodes (id) ON DELETE CASCADE,
    tenant     text NOT NULL,
    alias      text NOT NULL,
    PRIMARY KEY (entity_id, alias_key, episode_id)
);
CREATE INDEX entity_alias_sources_episode_idx ON entity_alias_sources (episode_id);

ALTER TABLE entity_alias_sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_alias_sources FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON entity_alias_sources TO memoryos_app
    USING (tenant = current_setting('app.tenant', true))
    WITH CHECK (tenant = current_setting('app.tenant', true));
GRANT SELECT, INSERT, UPDATE, DELETE ON entity_alias_sources TO memoryos_app;
