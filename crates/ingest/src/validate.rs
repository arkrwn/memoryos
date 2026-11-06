//! Text checks and normalization for commits.

/// Maximum episode content, in characters (SPEC section 9).
pub(crate) const MAX_CONTENT_CHARS: usize = 100_000;

/// A value the store would refuse: blank, or holding a NUL, which PostgreSQL cannot store.
pub(crate) fn is_blank_or_unstorable(value: &str) -> bool {
    value.trim().is_empty() || value.contains('\0')
}

/// `Person` and `person` are one kind. The store compares kinds exactly.
pub(crate) fn normalize_kind(kind: &str) -> String {
    kind.trim().to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_and_nul_are_refused() {
        for bad in ["", "   ", "\t\n", "a\0b"] {
            assert!(is_blank_or_unstorable(bad), "{bad:?}");
        }
        assert!(!is_blank_or_unstorable(" a "));
    }

    #[test]
    fn kind_normalization_trims_and_lowers() {
        assert_eq!(normalize_kind(" Person "), "person");
    }
}
