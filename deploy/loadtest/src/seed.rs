//! Creates one database and fills it through COPY, then builds the migration's indexes. All indexes
//! come from `pg_indexes` after migrating, so they cannot drift from `migrations/0001_init.sql`.

use std::fmt::Write as _;
use std::time::Instant;

use chrono::{DateTime, Duration, TimeZone, Utc};
use memoryos_store::PgStore;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use sqlx::postgres::PgPoolOptions;
use sqlx::{Connection, PgConnection, Row};
use uuid::Uuid;

use crate::data::{self, ScopeSpec, Topics, Vocab, HUB_EDGES};

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

const CHUNK: usize = 8 * 1024 * 1024;
const SPAN_DAYS: i64 = 730;
const FACTS_PER_EPISODE: usize = 3;
const FACTS_PER_ENTITY: usize = 6;

/// The newest fact's time. Fixed, so that a run can be reproduced, and in the past, so that every
/// fact is visible to a read at the present.
pub fn timeline_end() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 10, 1, 0, 0, 0).single().expect("valid date")
}

pub struct Options {
    pub admin_url: String,
    pub database: String,
    pub shape: String,
    pub facts: usize,
    pub seed: u64,
    pub maintenance_work_mem: String,
}

pub async fn run(options: &Options) -> Res<()> {
    let specs = data::shape(&options.shape, options.facts).map_err(|e| e.to_string())?;
    let total: usize = specs.iter().map(|spec| spec.facts).sum();
    println!("shape {}: {total} facts in {} scopes", options.shape, specs.len());

    let mut admin = PgConnection::connect(&options.admin_url).await?;
    sqlx::query(&format!("CREATE DATABASE {}", ident(&options.database)?))
        .execute(&mut admin)
        .await?;
    let url = database_url(&options.admin_url, &options.database);
    PgStore::connect(&url, 2).await?.migrate().await?;

    let pool = PgPoolOptions::new().max_connections(1).connect(&url).await?;
    let mut conn = pool.acquire().await?;
    let indexes = drop_indexes(&mut conn).await?;
    // Superuser only. Foreign keys are skipped because generated rows refer to later rows.
    sqlx::query("SET session_replication_role = replica")
        .execute(&mut *conn)
        .await?;

    let started = Instant::now();
    load(&mut conn, options, &specs).await?;
    println!("copied in {:.0}s", started.elapsed().as_secs_f64());

    sqlx::query("SET session_replication_role = DEFAULT")
        .execute(&mut *conn)
        .await?;
    sqlx::query(&format!(
        "SET maintenance_work_mem = '{}'",
        options.maintenance_work_mem.replace('\'', "")
    ))
    .execute(&mut *conn)
    .await?;
    sqlx::query("SET max_parallel_maintenance_workers = 7")
        .execute(&mut *conn)
        .await?;
    for (name, definition) in &indexes {
        let started = Instant::now();
        sqlx::query(definition).execute(&mut *conn).await?;
        println!("index {name} in {:.0}s", started.elapsed().as_secs_f64());
    }
    let started = Instant::now();
    sqlx::query("VACUUM (ANALYZE)").execute(&mut *conn).await?;
    sqlx::query("CHECKPOINT").execute(&mut *conn).await?;
    println!("vacuum and checkpoint in {:.0}s", started.elapsed().as_secs_f64());
    let size: String = sqlx::query_scalar("SELECT pg_size_pretty(pg_database_size(current_database()))")
        .fetch_one(&mut *conn)
        .await?;
    println!("database size {size}");
    Ok(())
}

fn ident(name: &str) -> Res<String> {
    if name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') {
        Ok(format!("\"{name}\""))
    } else {
        Err(format!("bad database name {name}").into())
    }
}

/// Replaces the path of a `postgres://` URL.
pub fn database_url(admin_url: &str, database: &str) -> String {
    let base = admin_url.rsplit_once('/').map_or(admin_url, |(base, _)| base);
    format!("{base}/{database}")
}

/// Drops every index except primary keys and unique constraints, and returns their definitions.
async fn drop_indexes(conn: &mut PgConnection) -> Res<Vec<(String, String)>> {
    let rows = sqlx::query(
        "SELECT indexname, indexdef FROM pg_indexes \
         WHERE schemaname = 'public' AND tablename IN ('facts', 'entities', 'episodes', 'fact_sources') \
           AND indexname NOT IN (SELECT conname FROM pg_constraint) \
         ORDER BY indexname",
    )
    .fetch_all(&mut *conn)
    .await?;
    let mut indexes = Vec::new();
    for row in rows {
        let name: String = row.try_get("indexname")?;
        let definition: String = row.try_get("indexdef")?;
        sqlx::query(&format!("DROP INDEX \"{name}\"")).execute(&mut *conn).await?;
        indexes.push((name, definition));
    }
    // HNSW last, so a failed build costs the least.
    indexes.sort_by_key(|(_, definition)| definition.contains("hnsw"));
    Ok(indexes)
}

struct Copy<'a> {
    copy: sqlx::postgres::PgCopyIn<&'a mut PgConnection>,
    buffer: String,
    rows: u64,
}

impl<'a> Copy<'a> {
    async fn start(conn: &'a mut PgConnection, statement: &str) -> Res<Copy<'a>> {
        Ok(Copy {
            copy: conn.copy_in_raw(statement).await?,
            buffer: String::with_capacity(CHUNK + 1024 * 1024),
            rows: 0,
        })
    }

    async fn row_done(&mut self) -> Res<()> {
        self.rows += 1;
        if self.buffer.len() >= CHUNK {
            self.copy.send(self.buffer.as_bytes()).await?;
            self.buffer.clear();
        }
        Ok(())
    }

    async fn finish(mut self) -> Res<u64> {
        if !self.buffer.is_empty() {
            self.copy.send(self.buffer.as_bytes()).await?;
        }
        self.copy.finish().await?;
        Ok(self.rows)
    }
}

/// Everything known about a fact before its row is written.
struct Meta {
    id: Uuid,
    scope: u32,
    subject: Uuid,
    object: Option<Uuid>,
    at_millis: i64,
    valid_to_millis: Option<i64>,
    superseded_by: Option<Uuid>,
    episode: Uuid,
}

fn v7(millis: i64, rng: &mut SmallRng) -> Uuid {
    let bytes: [u8; 10] = rng.gen();
    uuid::Builder::from_unix_timestamp_millis(millis as u64, &bytes).into_uuid()
}

fn time_text(millis: i64) -> String {
    Utc.timestamp_millis_opt(millis)
        .single()
        .expect("valid time")
        .format("%Y-%m-%d %H:%M:%S%.3f+00")
        .to_string()
}

fn write_vector(out: &mut String, vector: &[f32]) {
    out.push('[');
    for (i, value) in vector.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "{value:.5}");
    }
    out.push(']');
}

fn optional_uuid(id: Option<Uuid>) -> String {
    id.map_or_else(|| "\\N".to_owned(), |id| id.to_string())
}

async fn load(conn: &mut PgConnection, options: &Options, specs: &[ScopeSpec]) -> Res<()> {
    let mut rng = SmallRng::seed_from_u64(options.seed);
    let vocab = Vocab::new();
    let topics = Topics::new(options.seed ^ 0x5eed);
    let end = timeline_end();
    let start = end - Duration::days(SPAN_DAYS);
    let total: usize = specs.iter().map(|spec| spec.facts).sum();

    // Entities, per scope. Scope order in the heap is mixed, as it is under real traffic.
    let mut entities: Vec<Vec<Uuid>> = Vec::with_capacity(specs.len());
    let mut hub = None;
    let mut copy = Copy::start(
        conn,
        "COPY entities (id, tenant, scope, kind, name, normalized_name, embedding, created_at) FROM STDIN",
    )
    .await?;
    for spec in specs {
        let count = (spec.facts / FACTS_PER_ENTITY).max(3);
        let mut ids = Vec::with_capacity(count);
        for index in 0..count {
            let at = start.timestamp_millis() + rng.gen_range(0..SPAN_DAYS * 86_400_000);
            let id = v7(at, &mut rng);
            let is_hub = spec.hub_edges > 0 && index == 0;
            let name = if is_hub { "hub".to_owned() } else { format!("e{index}") };
            if is_hub {
                hub = Some(id);
            }
            let _ = write!(
                copy.buffer,
                "{id}\t{}\t{}\tthing\t{name}\t{name}\t",
                spec.tenant, spec.scope
            );
            write_vector(&mut copy.buffer, &data::random_unit(&mut rng));
            let _ = writeln!(copy.buffer, "\t{}", time_text(at));
            copy.row_done().await?;
            ids.push(id);
        }
        entities.push(ids);
    }
    println!("entities {}", copy.finish().await?);

    // One slot per fact, shuffled, so scopes interleave over time.
    let mut slots: Vec<u32> = Vec::with_capacity(total);
    for (index, spec) in specs.iter().enumerate() {
        slots.extend(std::iter::repeat_n(index as u32, spec.facts));
    }
    slots.shuffle(&mut rng);

    // Which of a hub scope's facts touch the hub. Half have it as subject, half as object.
    let mut remaining_hub: Vec<usize> = specs.iter().map(|spec| spec.hub_edges).collect();
    let mut remaining_scope: Vec<usize> = specs.iter().map(|spec| spec.facts).collect();
    let mut last_open: Vec<Option<usize>> = vec![None; specs.len()];
    let mut open_episode: Vec<(Uuid, usize)> = vec![(Uuid::nil(), FACTS_PER_EPISODE); specs.len()];
    let span_millis = (end - start).num_milliseconds();
    let mut metas: Vec<Meta> = Vec::with_capacity(total);
    let mut hub_count = 0_usize;
    for (position, &slot) in slots.iter().enumerate() {
        let scope = slot as usize;
        let at_millis = start.timestamp_millis() + (position as i64 * span_millis) / total as i64;
        let id = v7(at_millis, &mut rng);
        let ids = &entities[scope];
        // Square of a uniform draw: a few entities get many edges, most get few.
        let draw = |rng: &mut SmallRng| {
            let u: f64 = rng.gen();
            ids[((u * u * ids.len() as f64) as usize).min(ids.len() - 1)]
        };
        let mut subject = draw(&mut rng);
        let mut object = rng.gen_bool(0.6).then(|| draw(&mut rng));
        if remaining_hub[scope] > 0 && rng.gen_range(0..remaining_scope[scope]) < remaining_hub[scope] {
            let hub_id = hub.expect("the hub scope created a hub");
            if hub_count % 2 == 0 {
                subject = hub_id;
            } else {
                object = Some(hub_id);
            }
            hub_count += 1;
            remaining_hub[scope] -= 1;
        }
        remaining_scope[scope] -= 1;

        // About a fifth of facts close the fact before them in the same scope.
        if rng.gen_bool(0.2) {
            if let Some(previous) = last_open[scope] {
                metas[previous].valid_to_millis = Some(at_millis);
                metas[previous].superseded_by = Some(id);
            }
        }
        last_open[scope] = Some(position);

        let (episode, used) = &mut open_episode[scope];
        if *used >= FACTS_PER_EPISODE {
            *episode = v7(at_millis, &mut rng);
            *used = 0;
        }
        *used += 1;
        metas.push(Meta {
            id,
            scope: slot,
            subject,
            object,
            at_millis,
            valid_to_millis: None,
            superseded_by: None,
            episode: *episode,
        });
    }
    if options.shape == "c" {
        println!("hub edges {hub_count} (full size is {HUB_EDGES})");
    }

    // Facts, keeping each episode's text for its row below.
    let mut episodes: Vec<(Uuid, u32, i64, String)> = Vec::with_capacity(total / FACTS_PER_EPISODE + specs.len());
    let mut episode_index = std::collections::HashMap::new();
    let mut copy = Copy::start(
        conn,
        "COPY facts (id, tenant, scope, subject_id, predicate, object_id, statement, embedding, trust, \
         status, valid_from, valid_to, recorded_at, invalidated_at, superseded_by) FROM STDIN",
    )
    .await?;
    let mut statement = String::new();
    let interests: Vec<Vec<usize>> = specs
        .iter()
        .map(|_| (0..20).map(|_| rng.gen_range(0..topics.count())).collect())
        .collect();
    for (position, meta) in metas.iter().enumerate() {
        let spec = &specs[meta.scope as usize];
        statement.clear();
        let words = rng.gen_range(8..=20);
        for i in 0..words {
            if i > 0 {
                statement.push(' ');
            }
            statement.push_str(vocab.sample(&mut rng));
        }
        // Scopes favor a few topics, as a person's memories do, with the rest spread out.
        let topic = if rng.gen_bool(0.7) {
            interests[meta.scope as usize][rng.gen_range(0..20)]
        } else {
            rng.gen_range(0..topics.count())
        };
        let (status, trust) = match rng.gen_range(0..1000) {
            0..=19 => ("quarantined", "external"),
            20..=24 => ("rejected", "external"),
            25..=124 => ("active", "system"),
            125..=424 => ("active", "agent"),
            _ => ("active", "user"),
        };
        let invalidated = rng
            .gen_bool(0.01)
            .then(|| time_text(meta.at_millis + rng.gen_range(1..86_400_000)));
        let at = time_text(meta.at_millis);
        let predicate = vocab.word(rng.gen_range(40..240));
        let _ = write!(
            copy.buffer,
            "{}\t{}\t{}\t{}\t{predicate}\t{}\t{statement}\t",
            meta.id,
            spec.tenant,
            spec.scope,
            meta.subject,
            optional_uuid(meta.object),
        );
        write_vector(&mut copy.buffer, &topics.near(topic, &mut rng));
        let _ = writeln!(
            copy.buffer,
            "\t{trust}\t{status}\t{at}\t{}\t{at}\t{}\t{}",
            meta.valid_to_millis.map_or_else(|| "\\N".to_owned(), time_text),
            invalidated.as_deref().unwrap_or("\\N"),
            optional_uuid(meta.superseded_by),
        );
        copy.row_done().await?;

        let index = *episode_index.entry(meta.episode).or_insert_with(|| {
            episodes.push((meta.episode, meta.scope, meta.at_millis, String::new()));
            episodes.len() - 1
        });
        let text = &mut episodes[index].3;
        if !text.is_empty() {
            text.push_str(". ");
        }
        text.push_str(&statement);
        if position % 200_000 == 0 {
            println!("facts {position}");
        }
    }
    println!("facts {}", copy.finish().await?);

    let mut copy = Copy::start(conn, "COPY fact_sources (fact_id, episode_id, tenant) FROM STDIN").await?;
    for meta in &metas {
        let tenant = &specs[meta.scope as usize].tenant;
        let _ = writeln!(copy.buffer, "{}\t{}\t{tenant}", meta.id, meta.episode);
        copy.row_done().await?;
    }
    println!("fact_sources {}", copy.finish().await?);

    // The last 0.2% are still pending, the rest processed.
    let pending_from = total as f64 * 0.998;
    let mut copy = Copy::start(
        conn,
        "COPY episodes (id, tenant, scope, source, trust, content, occurred_at, created_at, processed_at) FROM STDIN",
    )
    .await?;
    for (id, scope, at_millis, text) in &episodes {
        let spec = &specs[*scope as usize];
        let at = time_text(*at_millis);
        let age = (*at_millis - start.timestamp_millis()) as f64 / span_millis as f64;
        let processed = if age * total as f64 >= pending_from { "\\N" } else { &at };
        let _ = writeln!(
            copy.buffer,
            "{id}\t{}\t{}\tloadtest\tuser\t{text}\t{at}\t{at}\t{processed}",
            spec.tenant, spec.scope
        );
        copy.row_done().await?;
    }
    println!("episodes {}", copy.finish().await?);
    Ok(())
}
