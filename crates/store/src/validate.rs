//! Input checks that run before any SQL, so bad input is `InvalidInput` and not `Storage`.

use memoryos_core::{Error, Result, Scope, EMBEDDING_DIMS};
use pgvector::Vector;

/// Upper bound for any `limit` argument. The API caps are lower.
const MAX_LIMIT: i64 = 1000;

/// Builds the bind value for an embedding. Cosine distance is undefined for the zero vector and
/// HNSW does not index it, so it is rejected along with wrong widths and non-finite values.
pub(crate) fn embedding(values: &[f32]) -> Result<Vector> {
    if values.len() != EMBEDDING_DIMS {
        return Err(Error::InvalidInput(format!(
            "embedding has {} dimensions, expected {EMBEDDING_DIMS}",
            values.len()
        )));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(
            "embedding has a non-finite value".into(),
        ));
    }
    if values.iter().all(|v| *v == 0.0) {
        return Err(Error::InvalidInput("embedding is the zero vector".into()));
    }
    Ok(Vector::from(values.to_vec()))
}

pub(crate) fn limit(value: i64) -> Result<i64> {
    if (1..=MAX_LIMIT).contains(&value) {
        Ok(value)
    } else {
        Err(Error::InvalidInput(format!(
            "limit must be in 1..={MAX_LIMIT}, got {value}"
        )))
    }
}

pub(crate) fn non_empty(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(Error::InvalidInput(format!("{field} must not be empty")));
    }
    Ok(())
}

pub(crate) fn scope_strings(scopes: &[Scope]) -> Vec<String> {
    scopes.iter().map(ToString::to_string).collect()
}

/// `api_keys.key_hash` must be a SHA-256 hex digest, so a raw token can never be stored by mistake.
pub(crate) fn key_hash(value: &str) -> Result<()> {
    let valid = value.len() == 64
        && value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
    if valid {
        Ok(())
    } else {
        Err(Error::InvalidInput(
            "key_hash must be lowercase SHA-256 hex".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_rejects_bad_vectors() {
        assert!(embedding(&[1.0; 3]).is_err());
        assert!(embedding(&vec![0.0; EMBEDDING_DIMS]).is_err());
        let mut nan = vec![1.0; EMBEDDING_DIMS];
        nan[7] = f32::NAN;
        assert!(embedding(&nan).is_err());
        assert!(embedding(&vec![1.0; EMBEDDING_DIMS]).is_ok());
    }

    #[test]
    fn limit_bounds() {
        assert!(limit(0).is_err());
        assert!(limit(1001).is_err());
        assert_eq!(limit(1000).ok(), Some(1000));
    }

    #[test]
    fn key_hash_requires_lowercase_hex_digest() {
        assert!(key_hash(&"a".repeat(64)).is_ok());
        assert!(key_hash(&"A".repeat(64)).is_err());
        assert!(key_hash("mos_rawtoken").is_err());
    }
}
