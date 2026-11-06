//! Black-box tests of the service over a real socket, with real Postgres. See common.rs.

#![cfg(feature = "integration")]

mod api;
mod authorization;
mod common;
mod contract;
mod limits;
mod mcp;
mod process;
