//! JSON logs, Prometheus metrics, and OTLP traces when `OTEL_EXPORTER_OTLP_ENDPOINT` is set. Metric
//! labels are limited to method, route template, status and job result, so a tenant or a key can
//! never add a time series.

use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::{Protocol, SpanExporter, WithExportConfig};
use opentelemetry_sdk::trace::SdkTracerProvider;
use opentelemetry_sdk::Resource;
use prometheus::{
    Encoder, Histogram, HistogramOpts, HistogramVec, IntCounterVec, Opts, Registry, TextEncoder,
};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

const OTLP_ENDPOINT_ENV: &str = "OTEL_EXPORTER_OTLP_ENDPOINT";
const DEFAULT_LOG_FILTER: &str = "info,opentelemetry=warn,opentelemetry_sdk=warn,hyper=warn";

/// Holds the trace exporter. Call `shutdown` after the runtime has stopped, because the blocking
/// exporter cannot run inside it.
pub struct TelemetryGuard {
    provider: Option<SdkTracerProvider>,
}

impl TelemetryGuard {
    pub fn shutdown(self) {
        if let Some(provider) = self.provider {
            if let Err(error) = provider.shutdown() {
                eprintln!("memoryos: trace flush failed: {error}");
            }
        }
    }
}

/// Call before entering the runtime: the blocking OTLP client must not be created inside one.
pub fn init() -> Result<TelemetryGuard, String> {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_LOG_FILTER));
    let logs = tracing_subscriber::fmt::layer()
        .json()
        .with_current_span(true)
        .with_span_list(false);

    let provider = match std::env::var(OTLP_ENDPOINT_ENV) {
        Ok(endpoint) if !endpoint.trim().is_empty() => Some(otlp_provider()?),
        _ => None,
    };
    let traces = provider
        .as_ref()
        .map(|provider| tracing_opentelemetry::layer().with_tracer(provider.tracer("memoryos")));
    tracing_subscriber::registry()
        .with(filter)
        .with(logs)
        .with(traces)
        .try_init()
        .map_err(|e| format!("cannot start logging: {e}"))?;
    Ok(TelemetryGuard { provider })
}

fn otlp_provider() -> Result<SdkTracerProvider, String> {
    // Reads the endpoint from the environment and appends `/v1/traces`, as the OTLP spec says.
    let exporter = SpanExporter::builder()
        .with_http()
        .with_protocol(Protocol::HttpBinary)
        .build()
        .map_err(|e| format!("cannot start the OTLP exporter: {e}"))?;
    Ok(SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(Resource::builder().with_service_name("memoryos").build())
        .build())
}

/// One registry per server, so tests that run several servers in a process do not share counts.
#[derive(Clone)]
pub struct Metrics {
    registry: Registry,
    requests: IntCounterVec,
    request_seconds: HistogramVec,
    recall_seconds: Histogram,
    jobs: IntCounterVec,
}

impl Metrics {
    pub fn new() -> Result<Self, prometheus::Error> {
        let registry = Registry::new();
        let requests = IntCounterVec::new(
            Opts::new("memoryos_http_requests_total", "HTTP requests served"),
            &["method", "route", "status"],
        )?;
        let request_seconds = HistogramVec::new(
            HistogramOpts::new(
                "memoryos_http_request_duration_seconds",
                "HTTP request latency",
            ),
            &["method", "route"],
        )?;
        let recall_seconds = Histogram::with_opts(HistogramOpts::new(
            "memoryos_recall_duration_seconds",
            "Recall latency, including the query embedding",
        ))?;
        let jobs = IntCounterVec::new(
            Opts::new("memoryos_jobs_processed_total", "Worker jobs finished"),
            &["result"],
        )?;
        registry.register(Box::new(requests.clone()))?;
        registry.register(Box::new(request_seconds.clone()))?;
        registry.register(Box::new(recall_seconds.clone()))?;
        registry.register(Box::new(jobs.clone()))?;
        Ok(Self {
            registry,
            requests,
            request_seconds,
            recall_seconds,
            jobs,
        })
    }

    pub fn request_finished(&self, method: &str, route: &str, status: u16, seconds: f64) {
        self.requests
            .with_label_values(&[method, route, status.to_string().as_str()])
            .inc();
        self.request_seconds
            .with_label_values(&[method, route])
            .observe(seconds);
    }

    pub fn recall_finished(&self, seconds: f64) {
        self.recall_seconds.observe(seconds);
    }

    pub fn job_finished(&self, result: &str) {
        self.jobs.with_label_values(&[result]).inc();
    }

    pub fn render(&self) -> Result<String, String> {
        let mut buffer = Vec::new();
        TextEncoder::new()
            .encode(&self.registry.gather(), &mut buffer)
            .map_err(|e| e.to_string())?;
        String::from_utf8(buffer).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_every_family_once_used() {
        let metrics = Metrics::new().expect("metrics");
        metrics.request_finished("GET", "/v1/facts/{id}", 200, 0.01);
        metrics.recall_finished(0.02);
        metrics.job_finished("completed");
        let text = metrics.render().expect("render");
        assert!(text.contains(
            r#"memoryos_http_requests_total{method="GET",route="/v1/facts/{id}",status="200"} 1"#
        ));
        assert!(text.contains("memoryos_http_request_duration_seconds_bucket"));
        assert!(text.contains("memoryos_recall_duration_seconds_count 1"));
        assert!(text.contains(r#"memoryos_jobs_processed_total{result="completed"} 1"#));
    }
}
