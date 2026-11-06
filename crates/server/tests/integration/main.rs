//! Black-box tests of the service over a real socket, with real Postgres and WireMock. See common.rs.

#![cfg(feature = "integration")]

mod authorization;
mod common;
mod contract;
mod host_mode;
mod limits;
mod mcp;
mod process;
mod worker_mode;
