//! The MemoryOS HTTP service: REST API, MCP endpoint, authorization, limits and telemetry.
//! See docs/SPEC.md sections 6, 9, 10 and 14 and docs/openapi.yaml.

pub mod app;
pub mod auth;
pub mod cli;
pub mod config;
pub mod error;
mod mcp;
mod metered;
mod middleware;
mod rate_limit;
mod readiness;
pub mod routes;
mod service;
pub mod telemetry;
