//! Seeds synthetic datasets and measures recall. See docs/loadtest-results.md for the commands.

mod data;
mod measure;
mod seed;

use std::collections::HashMap;

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

const USAGE: &str = "usage:
  memoryos-loadtest seed --db NAME --shape a|b|c [--facts N] [--seed N] [--maintenance-work-mem 2400MB]
  memoryos-loadtest measure --db NAME --shape a|b|c --label TEXT [--ops N] [--seconds N] [--pool N]
                            [--concurrency 1,8,32] [--only TEXT] [--out FILE]
Both take --url postgres://postgres:loadtest@127.0.0.1:55432/postgres";

fn flags(args: &[String]) -> Result<HashMap<String, String>, String> {
    let mut map = HashMap::new();
    let mut items = args.iter();
    while let Some(flag) = items.next() {
        let name = flag.strip_prefix("--").ok_or_else(|| format!("bad flag {flag}"))?;
        let value = items.next().ok_or_else(|| format!("{flag} needs a value"))?;
        map.insert(name.to_owned(), value.clone());
    }
    Ok(map)
}

fn get(map: &HashMap<String, String>, name: &str, default: &str) -> String {
    map.get(name).cloned().unwrap_or_else(|| default.to_owned())
}

fn required(map: &HashMap<String, String>, name: &str) -> Result<String, String> {
    map.get(name).cloned().ok_or_else(|| format!("--{name} is required\n{USAGE}"))
}

#[tokio::main]
async fn main() -> Res<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter("warn")
        .init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((command, rest)) = args.split_first() else {
        return Err(USAGE.into());
    };
    let map = flags(rest)?;
    let admin_url = get(&map, "url", "postgres://postgres:loadtest@127.0.0.1:55432/postgres");
    match command.as_str() {
        "seed" => {
            seed::run(&seed::Options {
                admin_url,
                database: required(&map, "db")?,
                shape: required(&map, "shape")?,
                facts: get(&map, "facts", "1000000").parse()?,
                seed: get(&map, "seed", "42").parse()?,
                maintenance_work_mem: get(&map, "maintenance-work-mem", "2400MB"),
            })
            .await
        }
        "measure" => {
            let concurrency = get(&map, "concurrency", "1,8,32")
                .split(',')
                .map(str::parse)
                .collect::<Result<Vec<usize>, _>>()?;
            measure::run(&measure::Options {
                admin_url,
                database: required(&map, "db")?,
                shape: required(&map, "shape")?,
                ops: get(&map, "ops", "300").parse()?,
                seconds: get(&map, "seconds", "40").parse()?,
                // MEMORYOS_DB_MAX_CONNECTIONS defaults to 20.
                pool: get(&map, "pool", "20").parse()?,
                concurrency,
                only: map.get("only").cloned(),
                label: required(&map, "label")?,
                out: map.get("out").cloned(),
            })
            .await
        }
        _ => Err(USAGE.into()),
    }
}
