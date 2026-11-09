// `sqlx::migrate!` embeds the directory at compile time; this rebuilds when it changes.
fn main() {
    println!("cargo:rerun-if-changed=../../migrations/postgres");
    println!("cargo:rerun-if-changed=../../migrations/sqlite");
}
