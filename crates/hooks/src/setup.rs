//! `memoryos setup`: Docker, Postgres, the server, two keys and the agent's configuration, in one
//! command with nothing for the user to edit. Running it again keeps the data, the password and
//! the keys.
//!
//! Everything lives in `<config>/memoryos`: `credentials.json`, `.env`, the Compose file and the
//! Postgres data. Only the model cache is a Docker volume, because uid 10001 in the container could
//! not write to a directory the user owns on Linux.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::json;

use crate::config::{self, process_env, Env};
use crate::credentials::{self, Credentials};
use crate::install::{self, Agent};
use crate::SetupArgs;

const COMPOSE_FILE: &str = include_str!("../../../deploy/docker-compose.yml");
const DEFAULT_PROJECT: &str = "memoryos";
const DEFAULT_IMAGE: &str = "memoryos:local";
const DEFAULT_PUBLISH: &str = "127.0.0.1:8080";
const TENANT: &str = "local";
const POSTGRES_DIR_IN_CONTAINER: &str = "/var/lib/postgresql/data";
/// Compose gets these from `.env`. A value in the user's shell would override the file.
const ENV_KEYS: [&str; 5] = [
    "POSTGRES_PASSWORD",
    "MEMORYOS_IMAGE",
    "MEMORYOS_PUBLISH",
    "MEMORYOS_PGDATA",
    "MEMORYOS_BUILD_CONTEXT",
];
/// The model download depends on the user's connection, and has been seen at 150 KB/s.
const READY_TIMEOUT: Duration = Duration::from_secs(90 * 60);
const READY_POLL: Duration = Duration::from_secs(3);
const READY_REPORT_EVERY: Duration = Duration::from_secs(15);

/// One key `setup` creates.
pub(crate) struct KeySpec {
    name: &'static str,
    admin: bool,
    max_trust: &'static str,
}

/// Hooks and MCP. Never an admin: an injected agent must not approve its own quarantined facts.
const AGENT_KEY: KeySpec = KeySpec {
    name: "agent",
    admin: false,
    max_trust: "agent",
};
/// `memoryos quarantine` only. An admin action needs trust that reaches its target's, and this one
/// stands for the human, so it may review anything.
const ADMIN_KEY: KeySpec = KeySpec {
    name: "admin",
    admin: true,
    max_trust: "system",
};

impl KeySpec {
    fn args(&self) -> Vec<&'static str> {
        let mut args = vec![
            "key",
            "create",
            "--tenant",
            TENANT,
            "--name",
            self.name,
            "--scopes",
            "agent:*,user:*,shared:*",
            "--max-trust",
            self.max_trust,
        ];
        if self.admin {
            args.push("--admin");
        }
        args
    }
}

// ---- Settings ----------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct Settings {
    dir: PathBuf,
    project: String,
    password: String,
    image: String,
    publish: String,
    build_context: Option<PathBuf>,
}

impl Settings {
    fn compose_file(&self) -> PathBuf {
        self.dir.join("docker-compose.yml")
    }

    fn env_file(&self) -> PathBuf {
        self.dir.join(".env")
    }

    fn pgdata(&self) -> PathBuf {
        self.dir.join("data").join("postgres")
    }

    fn url(&self) -> String {
        let publish = self.publish.replacen("0.0.0.0:", "127.0.0.1:", 1);
        format!("http://{publish}")
    }

    /// `.env`, read by Compose. Single quotes keep a path with spaces in one value.
    fn render(&self) -> String {
        let context = self
            .build_context
            .as_ref()
            .map_or(String::new(), |path| path.display().to_string());
        format!(
            "# Written by `memoryos setup`. The password is only for this machine's database.\n\
             MEMORYOS_PROJECT={}\n\
             POSTGRES_PASSWORD={}\n\
             MEMORYOS_IMAGE={}\n\
             MEMORYOS_PUBLISH={}\n\
             MEMORYOS_PGDATA='{}'\n\
             MEMORYOS_BUILD_CONTEXT='{}'\n",
            self.project,
            self.password,
            self.image,
            self.publish,
            self.pgdata().display(),
            context,
        )
    }

    /// What an earlier run wrote, if any.
    fn load(env: Env<'_>) -> Result<Option<Self>, String> {
        let dir = config::memoryos_dir(env)?;
        let path = dir.join(".env");
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(format!("cannot read {}: {error}", path.display())),
        };
        let value = |key: &str| {
            text.lines()
                .find_map(|line| line.strip_prefix(key)?.strip_prefix('='))
                .map(|value| value.trim().trim_matches('\'').to_owned())
                .filter(|value| !value.is_empty())
        };
        let Some(password) = value("POSTGRES_PASSWORD") else {
            return Err(format!(
                "{} has no POSTGRES_PASSWORD. Restore it, or run `memoryos setup --uninstall \
                 --delete-data` to start over",
                path.display()
            ));
        };
        Ok(Some(Self {
            project: value("MEMORYOS_PROJECT").unwrap_or_else(|| DEFAULT_PROJECT.to_owned()),
            password,
            image: value("MEMORYOS_IMAGE").unwrap_or_else(|| DEFAULT_IMAGE.to_owned()),
            publish: value("MEMORYOS_PUBLISH").unwrap_or_else(|| DEFAULT_PUBLISH.to_owned()),
            build_context: value("MEMORYOS_BUILD_CONTEXT").map(PathBuf::from),
            dir,
        }))
    }

    /// Keeps what an earlier run chose, above all the password, since the data depends on it. The
    /// `MEMORYOS_COMPOSE_PROJECT`, `MEMORYOS_IMAGE` and `MEMORYOS_PUBLISH` variables only count on
    /// a first run.
    fn prepare(env: Env<'_>) -> Result<Self, String> {
        let mut settings = match Self::load(env)? {
            Some(settings) => settings,
            None => Self::first_run(env)?,
        };
        settings.build_context = source_dir(env).or(settings.build_context);
        if settings.dir.display().to_string().contains('\'') {
            return Err(format!(
                "{} has a single quote in its path, which Compose's .env cannot hold",
                settings.dir.display()
            ));
        }
        fs::create_dir_all(settings.pgdata())
            .map_err(|e| format!("cannot create {}: {e}", settings.pgdata().display()))?;
        credentials::write_private(&settings.env_file(), settings.render().as_bytes())?;
        fs::write(settings.compose_file(), COMPOSE_FILE)
            .map_err(|e| format!("cannot write the Compose file: {e}"))?;
        Ok(settings)
    }

    fn first_run(env: Env<'_>) -> Result<Self, String> {
        let dir = config::memoryos_dir(env)?;
        let settings = Self {
            project: env("MEMORYOS_COMPOSE_PROJECT").unwrap_or_else(|| DEFAULT_PROJECT.to_owned()),
            password: generate_password()?,
            image: env("MEMORYOS_IMAGE").unwrap_or_else(|| DEFAULT_IMAGE.to_owned()),
            publish: env("MEMORYOS_PUBLISH").unwrap_or_else(|| DEFAULT_PUBLISH.to_owned()),
            build_context: None,
            dir,
        };
        let has_data =
            fs::read_dir(settings.pgdata()).is_ok_and(|mut entries| entries.next().is_some());
        if has_data {
            return Err(format!(
                "{} holds a database, but its password in .env is gone, so it cannot be opened. \
                 Restore .env, or move the directory away to start over",
                settings.pgdata().display()
            ));
        }
        Ok(settings)
    }
}

fn generate_password() -> Result<String, String> {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).map_err(|e| format!("cannot generate a password: {e}"))?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

/// Where to build the image from: `MEMORYOS_SOURCE_DIR`, or the checkout `cargo install --path`
/// built this binary from, while it still exists.
fn source_dir(env: Env<'_>) -> Option<PathBuf> {
    let built_from = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let candidate = env("MEMORYOS_SOURCE_DIR").map_or(built_from, PathBuf::from);
    if !candidate.join("Dockerfile").is_file() {
        return None;
    }
    fs::canonicalize(candidate).ok()
}

// ---- What setup needs from the outside ---------------------------------------------------------

/// Docker and the running server. Tests use a recording double, so no unit test starts a container.
trait Stack {
    fn check(&self) -> Result<(), String>;
    /// Builds the image if there is none, then starts Postgres, migrations and the server.
    fn start(&self, settings: &Settings) -> Result<(), String>;
    fn wait_ready(&self, settings: &Settings) -> Result<(), String>;
    /// Returns the token. It is printed by no one.
    fn create_key(&self, settings: &Settings, spec: &KeySpec) -> Result<String, String>;
    fn key_works(&self, url: &str, key: &str) -> Result<bool, String>;
    /// Stops the containers and removes their volumes and the Postgres files.
    fn destroy(&self, settings: &Settings) -> Result<(), String>;
}

struct Docker;

impl Docker {
    fn compose(settings: &Settings, args: &[&str]) -> Command {
        let mut command = Command::new("docker");
        command
            .arg("compose")
            .arg("-p")
            .arg(&settings.project)
            .arg("-f")
            .arg(settings.compose_file())
            .arg("--env-file")
            .arg(settings.env_file())
            .args(args);
        for key in ENV_KEYS {
            command.env_remove(key);
        }
        command
    }

    fn run(mut command: Command, what: &str) -> Result<(), String> {
        let status = command
            .status()
            .map_err(|e| format!("cannot run docker: {e}"))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("{what} failed. The output above says why."))
        }
    }

    /// For the progress line only, so a failure just shows as unknown.
    fn model_cache_size(settings: &Settings) -> String {
        let output = Self::compose(
            settings,
            &[
                "exec",
                "-T",
                "memoryos",
                "du",
                "-sm",
                "/var/lib/memoryos/models",
            ],
        )
        .stderr(Stdio::null())
        .output();
        output
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| {
                let text = String::from_utf8_lossy(&output.stdout).into_owned();
                text.split_whitespace().next().map(|mb| format!("{mb} MB"))
            })
            .unwrap_or_else(|| "unknown".to_owned())
    }

    fn ensure_image(settings: &Settings) -> Result<(), String> {
        let present = Command::new("docker")
            .args(["image", "inspect", &settings.image])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success());
        if present {
            return Ok(());
        }
        if settings.build_context.is_none() {
            return Err(format!(
                "there is no {} image, and no source to build it from. Set MEMORYOS_SOURCE_DIR to \
                 a MemoryOS checkout",
                settings.image
            ));
        }
        println!(
            "Building the {} image, once. This takes several minutes.",
            settings.image
        );
        Self::run(
            Self::compose(settings, &["build", "memoryos"]),
            "building the image",
        )
    }
}

impl Stack for Docker {
    fn check(&self) -> Result<(), String> {
        let info = Command::new("docker")
            .args(["info", "--format", "{{.ServerVersion}}"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        match info {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Err(
                "Docker is not installed. Install Docker, start it and run this again.".to_owned(),
            ),
            Err(error) => Err(format!("cannot run docker: {error}")),
            Ok(status) if !status.success() => {
                Err("Docker is not running. Start Docker and run this again.".to_owned())
            }
            Ok(_) => {
                let compose = Command::new("docker")
                    .args(["compose", "version"])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .is_ok_and(|status| status.success());
                if compose {
                    Ok(())
                } else {
                    Err(
                        "Docker Compose is missing. Install the `docker compose` plugin."
                            .to_owned(),
                    )
                }
            }
        }
    }

    fn start(&self, settings: &Settings) -> Result<(), String> {
        Self::ensure_image(settings)?;
        Self::run(
            Self::compose(settings, &["up", "-d"]),
            "starting the containers",
        )
    }

    fn wait_ready(&self, settings: &Settings) -> Result<(), String> {
        let url = settings.url();
        let runtime = runtime()?;
        let http = http_client()?;
        let endpoint = format!("{url}/readyz");
        let started = Instant::now();
        let mut reported = started;
        loop {
            let response = runtime.block_on(async { http.get(&endpoint).send().await });
            if response.is_ok_and(|response| response.status().is_success()) {
                return Ok(());
            }
            if started.elapsed() > READY_TIMEOUT {
                return Err(format!(
                    "the server was not ready after {} minutes. See its log with `docker logs` \
                     on the memoryos container",
                    READY_TIMEOUT.as_secs() / 60
                ));
            }
            if reported.elapsed() >= READY_REPORT_EVERY {
                reported = Instant::now();
                println!(
                    "  still waiting, {}s. Model cache: {} of 481 MB, downloaded once. About 2.5 \
                     minutes on a fast connection.",
                    started.elapsed().as_secs(),
                    Self::model_cache_size(settings),
                );
            }
            std::thread::sleep(READY_POLL);
        }
    }

    fn create_key(&self, settings: &Settings, spec: &KeySpec) -> Result<String, String> {
        let mut args = vec!["exec", "-T", "memoryos", "memoryos"];
        args.extend(spec.args());
        let output = Self::compose(settings, &args)
            .output()
            .map_err(|e| format!("cannot run docker: {e}"))?;
        let token = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        if !output.status.success() || token.is_empty() {
            return Err(format!(
                "cannot create the {} key: {}",
                spec.name,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        Ok(token)
    }

    fn key_works(&self, url: &str, key: &str) -> Result<bool, String> {
        let runtime = runtime()?;
        let http = http_client()?;
        // Inside the runtime: a request with a timeout needs its timer as soon as it is sent.
        let response = runtime.block_on(async {
            http.post(format!("{url}/mcp"))
                .bearer_auth(key)
                .header("content-type", "application/json")
                .body(json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).to_string())
                .send()
                .await
        });
        match response {
            Ok(response) => Ok(response.status().as_u16() != 401),
            Err(error) => Err(format!("cannot check a stored key at {url}: {error}")),
        }
    }

    fn destroy(&self, settings: &Settings) -> Result<(), String> {
        Self::run(
            Self::compose(settings, &["down"]),
            "stopping the containers",
        )?;
        // On Linux these files belong to the container's postgres user, so only a container can
        // remove them.
        Self::run(
            Self::compose(
                settings,
                &[
                    "run",
                    "--rm",
                    "--no-deps",
                    "--entrypoint",
                    "find",
                    "postgres",
                    POSTGRES_DIR_IN_CONTAINER,
                    "-mindepth",
                    "1",
                    "-delete",
                ],
            ),
            "removing the database files",
        )?;
        Self::run(
            Self::compose(settings, &["down", "-v"]),
            "removing the containers and volumes",
        )
    }
}

fn runtime() -> Result<tokio::runtime::Runtime, String> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("cannot start the runtime: {e}"))
}

fn http_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| format!("cannot start the HTTP client: {e}"))
}

// ---- The commands ------------------------------------------------------------------------------

pub fn run(args: &SetupArgs) -> Result<(), String> {
    let project =
        std::env::current_dir().map_err(|e| format!("cannot read the current directory: {e}"))?;
    if args.uninstall {
        let report = uninstall(
            &process_env,
            &Docker,
            args.agent,
            args.global,
            args.delete_data,
            &project,
        )?;
        println!("{report}");
        return Ok(());
    }
    let agent = args.agent.ok_or("--agent is required")?;
    let report = setup(&process_env, &Docker, agent, args.global, &project)?;
    println!("{report}");
    Ok(())
}

fn setup(
    env: Env<'_>,
    stack: &dyn Stack,
    agent: Agent,
    global: bool,
    project: &Path,
) -> Result<String, String> {
    // A configuration file that cannot be merged stops this before anything starts.
    let plans = install::plan_install(agent, global, project, env)?;

    println!("Checking Docker.");
    stack.check()?;

    let settings = Settings::prepare(env)?;
    println!(
        "Starting Postgres and the server, as Compose project {}.",
        settings.project
    );
    stack.start(&settings)?;

    let url = settings.url();
    println!("Waiting for {url}/readyz.");
    stack.wait_ready(&settings)?;
    println!("The server is ready.");

    let keys = ensure_keys(env, stack, &settings)?;
    let written = install::apply(&plans, false)?;

    let mut lines = vec![keys, written];
    lines.extend(agent_notes(agent, env));
    lines.push(format!("Done. Data is in {}.", settings.dir.display()));
    Ok(lines.join("\n"))
}

/// Keeps every stored key the server still accepts. Saves after each one, so a failure between the
/// two cannot lose a token that was already shown once.
fn ensure_keys(env: Env<'_>, stack: &dyn Stack, settings: &Settings) -> Result<String, String> {
    let url = settings.url();
    let mut stored = credentials::load(env)?.unwrap_or_default();
    stored.url.clone_from(&url);

    let agent_state = ensure_key(env, stack, settings, &AGENT_KEY, &mut stored, |c| {
        &mut c.agent_key
    })?;
    let admin_state = ensure_key(env, stack, settings, &ADMIN_KEY, &mut stored, |c| {
        &mut c.admin_key
    })?;
    credentials::save(env, &stored)?;
    Ok(format!(
        "Agent key {agent_state}, admin key {admin_state}, in {}.",
        credentials::path(env)?.display()
    ))
}

fn ensure_key(
    env: Env<'_>,
    stack: &dyn Stack,
    settings: &Settings,
    spec: &KeySpec,
    stored: &mut Credentials,
    slot: fn(&mut Credentials) -> &mut Option<String>,
) -> Result<&'static str, String> {
    if let Some(key) = slot(stored).as_deref() {
        if stack.key_works(&settings.url(), key)? {
            return Ok("kept");
        }
    }
    let token = stack.create_key(settings, spec)?;
    *slot(stored) = Some(token);
    credentials::save(env, stored)?;
    Ok("created")
}

fn agent_notes(agent: Agent, env: Env<'_>) -> Vec<String> {
    let mut notes = Vec::new();
    if agent == Agent::Pi {
        notes.push(
            "Pi reads .pi/mcp.json only in a trusted project: run /trust in Pi here once."
                .to_owned(),
        );
    }
    if !on_path(env) {
        notes.push(
            "`memoryos` is not on PATH, and the agent starts it by that name. Add it to PATH."
                .to_owned(),
        );
    }
    notes
}

fn on_path(env: Env<'_>) -> bool {
    env("PATH")
        .is_some_and(|path| std::env::split_paths(&path).any(|dir| dir.join("memoryos").is_file()))
}

fn uninstall(
    env: Env<'_>,
    stack: &dyn Stack,
    agent: Option<Agent>,
    global: bool,
    delete_data: bool,
    project: &Path,
) -> Result<String, String> {
    let agents = match agent {
        Some(agent) => vec![agent],
        // Pi has no global configuration from `setup`.
        None if global => vec![Agent::ClaudeCode, Agent::Opencode],
        None => vec![Agent::ClaudeCode, Agent::Opencode, Agent::Pi],
    };
    let mut plans = Vec::new();
    for agent in agents {
        plans.extend(install::plan_uninstall(agent, global, project, env)?);
    }
    let settings = Settings::load(env)?;
    if delete_data && settings.is_some() {
        // Docker being down must not leave half of this done.
        stack.check()?;
    }
    let mut lines = vec![install::apply(&plans, false)?];
    let dir = config::memoryos_dir(env)?;

    if !delete_data {
        if let Some(settings) = settings {
            lines.push(format!(
                "Data and credentials are kept in {}. The server is still running. Stop it with: \
                 docker compose -p {} -f {} down\nDelete everything with: memoryos setup \
                 --uninstall --delete-data",
                dir.display(),
                settings.project,
                settings.compose_file().display(),
            ));
        }
        return Ok(lines.join("\n"));
    }

    if let Some(settings) = &settings {
        stack.destroy(settings)?;
    }
    if dir.exists() {
        fs::remove_dir_all(&dir).map_err(|e| format!("cannot remove {}: {e}", dir.display()))?;
    }
    lines.push(format!(
        "Deleted the database, the model cache, the credentials and {}. The image is kept.",
        dir.display()
    ));
    Ok(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    /// Hands out numbered tokens and records every call.
    #[derive(Default)]
    struct FakeStack {
        calls: RefCell<Vec<String>>,
        issued: RefCell<Vec<String>>,
        revoked: RefCell<Vec<String>>,
        docker_down: bool,
    }

    impl FakeStack {
        fn count(&self, call: &str) -> usize {
            self.calls.borrow().iter().filter(|c| *c == call).count()
        }
    }

    impl Stack for FakeStack {
        fn check(&self) -> Result<(), String> {
            self.calls.borrow_mut().push("check".to_owned());
            if self.docker_down {
                return Err("Docker is not running.".to_owned());
            }
            Ok(())
        }

        fn start(&self, _settings: &Settings) -> Result<(), String> {
            self.calls.borrow_mut().push("start".to_owned());
            Ok(())
        }

        fn wait_ready(&self, _settings: &Settings) -> Result<(), String> {
            self.calls.borrow_mut().push("ready".to_owned());
            Ok(())
        }

        fn create_key(&self, _settings: &Settings, spec: &KeySpec) -> Result<String, String> {
            self.calls.borrow_mut().push("create".to_owned());
            let token = format!("mos_secret_{}_{}", spec.name, self.issued.borrow().len());
            self.issued.borrow_mut().push(token.clone());
            Ok(token)
        }

        fn key_works(&self, _url: &str, key: &str) -> Result<bool, String> {
            Ok(!self.revoked.borrow().iter().any(|k| k == key))
        }

        fn destroy(&self, _settings: &Settings) -> Result<(), String> {
            self.calls.borrow_mut().push("destroy".to_owned());
            Ok(())
        }
    }

    struct Sandbox {
        _root: tempfile::TempDir,
        home: PathBuf,
        project: PathBuf,
    }

    impl Sandbox {
        fn new() -> Self {
            let root = tempfile::tempdir().expect("dir");
            let home = root.path().join("home");
            let project = root.path().join("project");
            fs::create_dir_all(&home).expect("home");
            fs::create_dir_all(&project).expect("project");
            Self {
                _root: root,
                home,
                project,
            }
        }

        fn env(&self) -> impl Fn(&str) -> Option<String> {
            let home = self.home.display().to_string();
            move |name: &str| (name == "HOME").then(|| home.clone())
        }

        fn setup(&self, stack: &FakeStack, agent: Agent) -> Result<String, String> {
            setup(&self.env(), stack, agent, false, &self.project)
        }

        fn credentials(&self) -> Credentials {
            credentials::load(&self.env())
                .expect("load")
                .expect("present")
        }

        fn dir(&self) -> PathBuf {
            self.home.join(".config").join("memoryos")
        }

        fn read_all(dir: &Path) -> String {
            let mut out = String::new();
            for entry in fs::read_dir(dir).expect("list").flatten() {
                let path = entry.path();
                if path.is_dir() {
                    out.push_str(&Self::read_all(&path));
                } else {
                    out.push_str(&fs::read_to_string(&path).expect("read"));
                }
            }
            out
        }
    }

    #[test]
    fn the_agent_key_is_never_an_admin_and_the_admin_key_is() {
        assert!(!AGENT_KEY.args().contains(&"--admin"));
        assert!(ADMIN_KEY.args().contains(&"--admin"));
        assert!(ADMIN_KEY
            .args()
            .windows(2)
            .any(|pair| pair == ["--max-trust", "system"]));
        let args = AGENT_KEY.args();
        let at = |flag: &str| args[args.iter().position(|a| *a == flag).expect(flag) + 1];
        assert_eq!(at("--scopes"), "agent:*,user:*,shared:*");
        assert_eq!(at("--max-trust"), "agent");
    }

    #[test]
    fn setup_stores_two_different_keys_in_a_private_file() {
        let sandbox = Sandbox::new();
        let stack = FakeStack::default();
        sandbox.setup(&stack, Agent::ClaudeCode).expect("setup");

        let stored = sandbox.credentials();
        assert_eq!(stored.url, "http://127.0.0.1:8080");
        let agent = stored.agent_key.expect("agent key");
        let admin = stored.admin_key.expect("admin key");
        assert!(agent.contains("agent"), "{agent}");
        assert!(admin.contains("admin"), "{admin}");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for name in ["credentials.json", ".env"] {
                let meta = fs::metadata(sandbox.dir().join(name)).expect("stat");
                assert_eq!(meta.permissions().mode() & 0o777, 0o600, "{name}");
            }
        }
    }

    #[test]
    fn running_it_again_keeps_the_password_and_keys_and_creates_none() {
        let sandbox = Sandbox::new();
        let stack = FakeStack::default();
        sandbox.setup(&stack, Agent::Opencode).expect("first");
        let first_creds = fs::read(sandbox.dir().join("credentials.json")).expect("read");
        let first_env = fs::read(sandbox.dir().join(".env")).expect("read");
        let config = fs::read(sandbox.project.join("opencode.json")).expect("read");

        let report = sandbox.setup(&stack, Agent::Opencode).expect("second");

        assert_eq!(stack.count("create"), 2, "only the first run creates keys");
        assert_eq!(stack.count("start"), 2);
        assert!(
            report.contains("Agent key kept, admin key kept"),
            "{report}"
        );
        assert_eq!(
            fs::read(sandbox.dir().join("credentials.json")).expect("read"),
            first_creds
        );
        assert_eq!(
            fs::read(sandbox.dir().join(".env")).expect("read"),
            first_env
        );
        assert_eq!(
            fs::read(sandbox.project.join("opencode.json")).expect("read"),
            config
        );
    }

    #[test]
    fn a_key_the_server_refuses_is_replaced_and_the_other_is_kept() {
        let sandbox = Sandbox::new();
        let stack = FakeStack::default();
        sandbox.setup(&stack, Agent::ClaudeCode).expect("first");
        let before = sandbox.credentials();
        stack
            .revoked
            .borrow_mut()
            .push(before.agent_key.clone().expect("key"));

        sandbox.setup(&stack, Agent::ClaudeCode).expect("second");

        let after = sandbox.credentials();
        assert_eq!(stack.count("create"), 3);
        assert!(after.agent_key != before.agent_key);
        assert!(after.admin_key == before.admin_key);
    }

    #[test]
    fn no_key_reaches_any_agent_configuration() {
        for agent in [Agent::ClaudeCode, Agent::Opencode, Agent::Pi] {
            let sandbox = Sandbox::new();
            let stack = FakeStack::default();
            let report = sandbox.setup(&stack, agent).expect("setup");
            let written = Sandbox::read_all(&sandbox.project);
            assert!(!written.is_empty());
            for secret in ["mos_secret", "MEMORYOS_API_KEY", "Authorization"] {
                assert!(!written.contains(secret), "{secret} in {written}");
            }
            assert!(!report.contains("mos_secret"), "{report}");
        }
    }

    #[test]
    fn docker_down_stops_setup_before_anything_is_written() {
        let sandbox = Sandbox::new();
        let stack = FakeStack {
            docker_down: true,
            ..FakeStack::default()
        };
        let error = sandbox.setup(&stack, Agent::Opencode).expect_err("stop");
        assert!(error.contains("Docker is not running"), "{error}");
        assert_eq!(fs::read_dir(&sandbox.project).expect("list").count(), 0);
        assert_eq!(fs::read_dir(&sandbox.home).expect("list").count(), 0);
    }

    #[test]
    fn a_config_that_cannot_merge_stops_setup_before_docker() {
        let sandbox = Sandbox::new();
        fs::write(sandbox.project.join("opencode.json"), "{ // mine\n}").expect("seed");
        let stack = FakeStack::default();
        assert!(sandbox.setup(&stack, Agent::Opencode).is_err());
        assert_eq!(stack.count("check"), 0);
    }

    #[test]
    fn a_database_without_its_password_is_refused_not_overwritten() {
        let sandbox = Sandbox::new();
        let stack = FakeStack::default();
        sandbox.setup(&stack, Agent::ClaudeCode).expect("first");
        let pg = sandbox.dir().join("data/postgres/PG_VERSION");
        fs::write(&pg, "17").expect("seed");
        fs::remove_file(sandbox.dir().join(".env")).expect("remove");

        let error = sandbox.setup(&stack, Agent::ClaudeCode).expect_err("stop");
        assert!(error.contains("password"), "{error}");
    }

    #[test]
    fn uninstall_keeps_data_and_says_how_to_stop() {
        let sandbox = Sandbox::new();
        let stack = FakeStack::default();
        sandbox.setup(&stack, Agent::Opencode).expect("setup");

        let report = uninstall(&sandbox.env(), &stack, None, false, false, &sandbox.project)
            .expect("uninstall");

        assert!(!sandbox.project.join("opencode.json").exists());
        assert!(!sandbox
            .project
            .join(".opencode/plugins/memoryos.ts")
            .exists());
        assert!(sandbox.dir().join("credentials.json").exists());
        assert!(sandbox.dir().join("data/postgres").exists());
        assert_eq!(stack.count("destroy"), 0);
        assert!(report.contains("kept"), "{report}");
    }

    #[test]
    fn delete_data_removes_everything_only_when_asked() {
        let sandbox = Sandbox::new();
        let stack = FakeStack::default();
        sandbox.setup(&stack, Agent::ClaudeCode).expect("setup");

        uninstall(
            &sandbox.env(),
            &stack,
            Some(Agent::ClaudeCode),
            false,
            true,
            &sandbox.project,
        )
        .expect("uninstall");

        assert_eq!(stack.count("destroy"), 1);
        assert!(!sandbox.dir().exists());
    }

    #[test]
    fn delete_data_with_docker_down_changes_nothing() {
        let sandbox = Sandbox::new();
        sandbox
            .setup(&FakeStack::default(), Agent::ClaudeCode)
            .expect("setup");
        let down = FakeStack {
            docker_down: true,
            ..FakeStack::default()
        };
        let result = uninstall(
            &sandbox.env(),
            &down,
            Some(Agent::ClaudeCode),
            false,
            true,
            &sandbox.project,
        );
        assert!(result.is_err());
        assert!(sandbox.project.join(".mcp.json").exists());
        assert!(sandbox.dir().join("credentials.json").exists());
    }

    #[test]
    fn the_first_run_variables_are_ignored_later() {
        let sandbox = Sandbox::new();
        let home = sandbox.home.display().to_string();
        let first = |name: &str| match name {
            "HOME" => Some(home.clone()),
            "MEMORYOS_COMPOSE_PROJECT" => Some("throwaway".to_owned()),
            "MEMORYOS_PUBLISH" => Some("127.0.0.1:18080".to_owned()),
            _ => None,
        };
        let settings = Settings::prepare(&first).expect("first");
        assert_eq!(settings.project, "throwaway");
        assert_eq!(settings.url(), "http://127.0.0.1:18080");

        let later = Settings::prepare(&sandbox.env()).expect("later");
        assert_eq!(later.project, "throwaway");
        assert_eq!(later.password, settings.password);
        assert_eq!(later.publish, settings.publish);
    }

    #[test]
    fn a_password_is_long_and_never_repeats() {
        let one = generate_password().expect("one");
        assert_eq!(one.len(), 64);
        assert!(one != generate_password().expect("two"));
    }

    #[test]
    fn the_compose_file_defines_no_key_and_no_provider() {
        for word in ["ANTHROPIC", "OPENAI", "worker"] {
            assert!(!COMPOSE_FILE.contains(word), "{word}");
        }
    }
}
